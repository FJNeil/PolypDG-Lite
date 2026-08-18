import os
import csv
import json
import time
import random
import shutil
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError as e:
    raise ImportError("找不到 transformers。請先安裝：pip install transformers") from e


# =========================================================
# 0) 主要設定區
# =========================================================
LOCO_ROOT = os.environ.get("POLYPDG_LOCO_ROOT", "./data/loco_1537_clean")

# 建議改成新資料夾，避免覆蓋你舊版 consistency 結果
OUT_ROOT = os.environ.get("POLYPDG_STAGE2_OUT", "./outputs/stage2_teacher")

RUN_FOLDS = [
    "fold_test_C1",
    "fold_test_C2",
    "fold_test_C3",
    "fold_test_C4",
    "fold_test_C5",
    "fold_test_C6",
]

# 新資料夾通常不用 reset；若只想重跑特定 fold，再把 fold 名稱加進來
RESET_FOLDS = set()

SKIP_COMPLETED_FOLDS = True
DELETE_CHECKPOINT_AFTER_FINISH = False

IMAGE_SIZE = (352, 352)   # (W, H)
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

THRESHOLD = 0.5
EPS = 1e-7
CONSISTENCY_EPS = 1e-4
MAX_LOGIT_ABS = 20.0

NUM_WORKERS = 0
PIN_MEMORY = True
USE_AMP = True
SEED = 42

GRAD_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = USE_AMP and DEVICE == "cuda"

MODEL_NAME = "SegFormer-B2"
USE_PRETRAINED = True
PRETRAINED_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

EVAL_OUT_DIR = os.path.join(OUT_ROOT, "Stage2C_eval_summary")

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)


# =========================================================
# 0-1) Consistency-only 設定（不改研究定義，只強化 self-consistency）
# =========================================================
CONSISTENCY_CFG = {
    "enable": True,
    "weight": 0.08,
    "rampup_epochs": 30,
    "temperature": 1.0,
    "loss_type": "weighted_mse",
    "nan_to_num": True,

    # confidence-aware weighting
    "use_confidence_mask": True,
    "confidence_threshold": 0.25,
    "confidence_power": 2.0,
    "confidence_floor": 0.05,
    "prob_clamp_min": 1e-4,

    # boundary-aware weighting
    "use_boundary_weight": True,
    "boundary_band_radius": 3,
    "boundary_weight": 2.0,

    # disagreement-aware focus
    "use_disagreement_focus": True,
    "disagreement_weight": 0.50,

    # consistency view augmentation（輕量，不等於 PB）
    "view_aug": {
        "brightness_limit": 0.18,
        "contrast_limit": 0.18,
        "gamma_limit": 0.12,
        "color_limit": 0.12,
        "sharpness_limit": 0.10,
        "blur_prob": 0.15,
        "blur_radius_min": 0.30,
        "blur_radius_max": 1.00,
        "noise_prob": 0.20,
        "noise_std": 0.02,
    }
}


# =========================================================
# 1) 固定亂數種子
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2) 印出執行資訊
# =========================================================
print("========================================")
print("DEVICE =", DEVICE)
if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
print("MODEL_NAME =", MODEL_NAME)
print("USE_PRETRAINED =", USE_PRETRAINED)
print("PRETRAINED_MODEL_NAME =", PRETRAINED_MODEL_NAME)
print("OUT_ROOT =", OUT_ROOT)
print("EVAL_OUT_DIR =", EVAL_OUT_DIR)
print("RUN_FOLDS =", RUN_FOLDS)
print("IMAGE_SIZE =", IMAGE_SIZE)
print("BATCH_SIZE =", BATCH_SIZE)
print("NUM_EPOCHS =", NUM_EPOCHS)
print("LR =", LR)
print("WEIGHT_DECAY =", WEIGHT_DECAY)
print("EARLY_STOPPING_PATIENCE =", EARLY_STOPPING_PATIENCE)
print("AMP_ENABLED =", AMP_ENABLED)
print("GRAD_CLIP_NORM =", GRAD_CLIP_NORM)
print("CONSISTENCY_CFG =", json.dumps(CONSISTENCY_CFG, ensure_ascii=False, indent=2))
print("========================================")


# =========================================================
# 3) 同步前處理 / augmentation
# =========================================================
def resize_pair(img, mask, image_size):
    img = img.resize(image_size, Image.BILINEAR)
    mask = mask.resize(image_size, Image.NEAREST)
    return img, mask


def random_hflip_pair(img, mask, p=0.5):
    if random.random() < p:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)
    return img, mask


def random_vflip_pair(img, mask, p=0.5):
    if random.random() < p:
        img = ImageOps.flip(img)
        mask = ImageOps.flip(mask)
    return img, mask


def random_rotate_pair(img, mask, p=0.5, angle_limit=15):
    if random.random() < p:
        angle = random.uniform(-angle_limit, angle_limit)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
    return img, mask


def random_scale_shift_pair(img, mask, p=0.5, scale_low=0.90, scale_high=1.10):
    if random.random() >= p:
        return img, mask

    w, h = img.size
    scale = random.uniform(scale_low, scale_high)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img_scaled = img.resize((new_w, new_h), Image.BILINEAR)
    mask_scaled = mask.resize((new_w, new_h), Image.NEAREST)

    if scale >= 1.0:
        max_left = max(0, new_w - w)
        max_top = max(0, new_h - h)

        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0

        img_out = img_scaled.crop((left, top, left + w, top + h))
        mask_out = mask_scaled.crop((left, top, left + w, top + h))
        return img_out, mask_out

    canvas_img = Image.new("RGB", (w, h), (0, 0, 0))
    canvas_mask = Image.new("L", (w, h), 0)

    max_left = max(0, w - new_w)
    max_top = max(0, h - new_h)

    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0

    canvas_img.paste(img_scaled, (left, top))
    canvas_mask.paste(mask_scaled, (left, top))

    return canvas_img, canvas_mask


def random_brightness_contrast(img, p=0.4, brightness_limit=0.15, contrast_limit=0.15):
    if random.random() < p:
        brightness_factor = random.uniform(1.0 - brightness_limit, 1.0 + brightness_limit)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    if random.random() < p:
        contrast_factor = random.uniform(1.0 - contrast_limit, 1.0 + contrast_limit)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def random_gamma(img, p=0.3, gamma_limit=0.12):
    if random.random() >= p:
        return img

    gamma = random.uniform(1.0 - gamma_limit, 1.0 + gamma_limit)

    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_np = np.clip(img_np, 0.0, 1.0)
    img_np = np.power(img_np, gamma)
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


def random_color_jitter(img, p=0.3, color_limit=0.12):
    if random.random() >= p:
        return img

    factor = random.uniform(1.0 - color_limit, 1.0 + color_limit)
    return ImageEnhance.Color(img).enhance(factor)


def random_sharpness_jitter(img, p=0.2, sharpness_limit=0.10):
    if random.random() >= p:
        return img

    factor = random.uniform(1.0 - sharpness_limit, 1.0 + sharpness_limit)
    return ImageEnhance.Sharpness(img).enhance(factor)


def random_gaussian_blur_pil(img, p=0.15, radius_min=0.30, radius_max=1.00):
    if random.random() >= p:
        return img

    radius = random.uniform(radius_min, radius_max)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def random_gaussian_noise_pil(img, p=0.20, noise_std=0.02):
    if random.random() >= p:
        return img

    img_np = np.asarray(img, dtype=np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=noise_std, size=img_np.shape).astype(np.float32)
    img_np = np.clip(img_np + noise, 0.0, 1.0)
    img_np = (img_np * 255.0).astype(np.uint8)

    return Image.fromarray(img_np)


def apply_consistency_view_aug(img, aug_cfg):
    img = random_brightness_contrast(
        img,
        p=0.4,
        brightness_limit=aug_cfg["brightness_limit"],
        contrast_limit=aug_cfg["contrast_limit"]
    )

    img = random_gamma(
        img,
        p=0.3,
        gamma_limit=aug_cfg["gamma_limit"]
    )

    img = random_color_jitter(
        img,
        p=0.3,
        color_limit=aug_cfg["color_limit"]
    )

    img = random_sharpness_jitter(
        img,
        p=0.2,
        sharpness_limit=aug_cfg["sharpness_limit"]
    )

    img = random_gaussian_blur_pil(
        img,
        p=aug_cfg["blur_prob"],
        radius_min=aug_cfg["blur_radius_min"],
        radius_max=aug_cfg["blur_radius_max"]
    )

    img = random_gaussian_noise_pil(
        img,
        p=aug_cfg["noise_prob"],
        noise_std=aug_cfg["noise_std"]
    )

    return img


# =========================================================
# 4) PIL / NumPy 轉換工具
# =========================================================
def image_to_chw_float(img):
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np


def mask_to_1hw_float(mask):
    mask_np = np.array(mask, dtype=np.uint8)
    mask_np = (mask_np > 127).astype(np.float32)
    mask_np = np.expand_dims(mask_np, axis=0)
    return mask_np


# =========================================================
# 5) 前處理
# =========================================================
def preprocess_single_view(img, mask, image_size=(352, 352), is_train=False):
    img, mask = resize_pair(img, mask, image_size)

    if is_train:
        img, mask = random_hflip_pair(img, mask, p=0.5)
        img, mask = random_vflip_pair(img, mask, p=0.5)
        img, mask = random_rotate_pair(img, mask, p=0.5, angle_limit=15)
        img, mask = random_scale_shift_pair(img, mask, p=0.5, scale_low=0.90, scale_high=1.10)
        img = random_brightness_contrast(img, p=0.4, brightness_limit=0.15, contrast_limit=0.15)

    img_np = image_to_chw_float(img)
    mask_np = mask_to_1hw_float(mask)

    return img_np, mask_np


def preprocess_two_views_for_consistency(img, mask, image_size=(352, 352)):
    """
    仍然是 paired-view self-consistency：
    1. 共享幾何 augmentation
    2. 兩個 views 各自做輕量 photometric augmentation
    3. mask 只保留一份，因為幾何位置對齊
    """
    img, mask = resize_pair(img, mask, image_size)

    img, mask = random_hflip_pair(img, mask, p=0.5)
    img, mask = random_vflip_pair(img, mask, p=0.5)
    img, mask = random_rotate_pair(img, mask, p=0.5, angle_limit=15)
    img, mask = random_scale_shift_pair(img, mask, p=0.5, scale_low=0.90, scale_high=1.10)

    img_view_a = img.copy()
    img_view_b = img.copy()

    img_view_a = apply_consistency_view_aug(img_view_a, CONSISTENCY_CFG["view_aug"])
    img_view_b = apply_consistency_view_aug(img_view_b, CONSISTENCY_CFG["view_aug"])

    img_a_np = image_to_chw_float(img_view_a)
    img_b_np = image_to_chw_float(img_view_b)
    mask_np = mask_to_1hw_float(mask)

    return img_a_np, img_b_np, mask_np


# =========================================================
# 6) Dataset
# =========================================================
class PolypDataset(Dataset):
    def __init__(self, csv_file, image_size=(352, 352), is_train=False, use_consistency=False):
        self.rows = []
        self.image_size = image_size
        self.is_train = is_train
        self.use_consistency = use_consistency

        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")

        if self.is_train and self.use_consistency:
            img_a_np, img_b_np, mask_np = preprocess_two_views_for_consistency(
                img=img,
                mask=mask,
                image_size=self.image_size
            )

            return {
                "image_a": torch.tensor(img_a_np, dtype=torch.float32),
                "image_b": torch.tensor(img_b_np, dtype=torch.float32),
                "mask": torch.tensor(mask_np, dtype=torch.float32),
                "stem": row["stem"],
                "image_path": row["image_path"],
                "mask_path": row["mask_path"],
                "center": row["center"] if "center" in row else "UNKNOWN",
            }

        img_np, mask_np = preprocess_single_view(
            img=img,
            mask=mask,
            image_size=self.image_size,
            is_train=self.is_train
        )

        return {
            "image": torch.tensor(img_np, dtype=torch.float32),
            "mask": torch.tensor(mask_np, dtype=torch.float32),
            "stem": row["stem"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "center": row["center"] if "center" in row else "UNKNOWN",
        }


# =========================================================
# 7) SegFormer-B2 模型
# =========================================================
class SegFormerBinary(nn.Module):
    def __init__(self, use_pretrained=True, pretrained_name="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()

        if use_pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_name,
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                )
                print(f"[SegFormerBinary] loaded pretrained weights from: {pretrained_name}")
            except Exception as e:
                print("[SegFormerBinary] pretrained load failed, fallback to random-init B2 config")
                print("reason:", e)
                self.model = self._build_b2_from_scratch()
        else:
            self.model = self._build_b2_from_scratch()

    def _build_b2_from_scratch(self):
        config = SegformerConfig(
            num_labels=1,
            num_channels=3,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
            num_attention_heads=[1, 2, 5, 8],
            sr_ratios=[8, 4, 2, 1],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout_prob=0.1,
            drop_path_rate=0.1,
        )
        return SegformerForSemanticSegmentation(config)

    def forward(self, x):
        input_h, input_w = x.shape[-2], x.shape[-1]

        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
            )

        return logits


# =========================================================
# 8) Supervised loss：BCE + Dice
# =========================================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return bce + dice_loss


# =========================================================
# 9) Consistency loss（強化版，但仍是 self-consistency）
# =========================================================
def binary_prob_to_two_class(prob, eps=1e-4):
    prob = prob.clamp(eps, 1.0 - eps)
    bg = (1.0 - prob).clamp(eps, 1.0 - eps)
    fg = prob
    return torch.cat([bg, fg], dim=1)


def compute_binary_entropy(prob, eps=1e-4):
    prob = prob.clamp(eps, 1.0 - eps)
    entropy = -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))
    return entropy


def build_boundary_band_from_mask(mask, radius=3):
    if radius <= 0:
        return torch.zeros_like(mask)

    mask = mask.float()
    kernel_size = 2 * radius + 1

    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=radius)

    boundary_band = (dilated - eroded).clamp(0.0, 1.0)
    return boundary_band


def consistency_mse_loss(logits_a, logits_b):
    logits_a = logits_a.float().clamp(-MAX_LOGIT_ABS, MAX_LOGIT_ABS)
    logits_b = logits_b.float().clamp(-MAX_LOGIT_ABS, MAX_LOGIT_ABS)

    prob_a = torch.sigmoid(logits_a)
    prob_b = torch.sigmoid(logits_b)

    loss = F.mse_loss(prob_a, prob_b, reduction="mean")
    return loss


def symmetric_binary_kl_stable(logits_a, logits_b, eps=1e-4):
    logits_a = logits_a.float().clamp(-MAX_LOGIT_ABS, MAX_LOGIT_ABS)
    logits_b = logits_b.float().clamp(-MAX_LOGIT_ABS, MAX_LOGIT_ABS)

    prob_a = torch.sigmoid(logits_a).clamp(eps, 1.0 - eps)
    prob_b = torch.sigmoid(logits_b).clamp(eps, 1.0 - eps)

    dist_a = binary_prob_to_two_class(prob_a, eps=eps)
    dist_b = binary_prob_to_two_class(prob_b, eps=eps)

    log_dist_a = torch.log(dist_a.clamp_min(eps))
    log_dist_b = torch.log(dist_b.clamp_min(eps))

    kl_ab_map = F.kl_div(log_dist_a, dist_b, reduction="none")
    kl_ba_map = F.kl_div(log_dist_b, dist_a, reduction="none")

    kl_ab_map = kl_ab_map.sum(dim=1)
    kl_ba_map = kl_ba_map.sum(dim=1)

    kl_ab = kl_ab_map.mean()
    kl_ba = kl_ba_map.mean()

    loss = 0.5 * (kl_ab + kl_ba)
    return loss


def build_consistency_weight_map(prob_a, prob_b, masks):
    eps = CONSISTENCY_CFG["prob_clamp_min"]

    mean_prob = 0.5 * (prob_a + prob_b)

    entropy = compute_binary_entropy(mean_prob, eps=eps)
    confidence = 1.0 - (entropy / np.log(2.0))
    confidence = confidence.clamp(0.0, 1.0)

    confidence = confidence.pow(CONSISTENCY_CFG["confidence_power"])
    confidence = confidence.clamp_min(CONSISTENCY_CFG["confidence_floor"])

    if CONSISTENCY_CFG["use_confidence_mask"]:
        valid_mask = (confidence >= CONSISTENCY_CFG["confidence_threshold"]).float()
    else:
        valid_mask = torch.ones_like(confidence)

    weight_map = confidence * valid_mask

    if CONSISTENCY_CFG["use_boundary_weight"]:
        boundary_band = build_boundary_band_from_mask(
            masks,
            radius=CONSISTENCY_CFG["boundary_band_radius"]
        )
        weight_map = weight_map * (1.0 + CONSISTENCY_CFG["boundary_weight"] * boundary_band)

    if CONSISTENCY_CFG["use_disagreement_focus"]:
        disagreement = torch.abs(prob_a - prob_b)
        disagreement = disagreement / (disagreement.mean(dim=(2, 3), keepdim=True) + 1e-6)
        disagreement = disagreement.clamp(0.0, 3.0)

        weight_map = weight_map * (1.0 + CONSISTENCY_CFG["disagreement_weight"] * disagreement)

    return weight_map.clamp_min(0.0)


def weighted_consistency_mse_loss(logits_a, logits_b, masks):
    logits_a = logits_a.float().clamp(-MAX_LOGIT_ABS, MAX_LOGIT_ABS)
    logits_b = logits_b.float().clamp(-MAX_LOGIT_ABS, MAX_LOGIT_ABS)

    prob_a = torch.sigmoid(logits_a).clamp(
        CONSISTENCY_CFG["prob_clamp_min"], 1.0 - CONSISTENCY_CFG["prob_clamp_min"]
    )
    prob_b = torch.sigmoid(logits_b).clamp(
        CONSISTENCY_CFG["prob_clamp_min"], 1.0 - CONSISTENCY_CFG["prob_clamp_min"]
    )

    loss_map = (prob_a - prob_b).pow(2)

    weight_map = build_consistency_weight_map(
        prob_a.detach(),
        prob_b.detach(),
        masks
    )

    weighted_sum = (loss_map * weight_map).sum()
    normalizer = weight_map.sum().clamp_min(1.0)

    return weighted_sum / normalizer


def get_consistency_loss(logits_a, logits_b, masks):
    loss_type = CONSISTENCY_CFG["loss_type"]

    if loss_type == "weighted_mse":
        loss = weighted_consistency_mse_loss(logits_a, logits_b, masks)

    elif loss_type == "mse":
        loss = consistency_mse_loss(logits_a, logits_b)

    elif loss_type == "symmetric_kl":
        loss = symmetric_binary_kl_stable(
            logits_a, logits_b, eps=CONSISTENCY_EPS
        )

    else:
        raise ValueError(f"不支援的 consistency loss type: {loss_type}")

    if CONSISTENCY_CFG.get("nan_to_num", True):
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

    return loss


def get_current_consistency_weight(epoch, base_weight, rampup_epochs):
    if rampup_epochs <= 0:
        return base_weight

    factor = min(1.0, epoch / float(rampup_epochs))
    factor = 0.5 - 0.5 * np.cos(np.pi * factor)
    return base_weight * factor


# =========================================================
# 10) Metrics
# =========================================================
def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# =========================================================
# 11) Train / Val / Test epoch
# =========================================================
def run_train_epoch_consistency(model, loader, criterion, optimizer, scaler, epoch):
    model.train()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_cons_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_count = 0
    skipped_batches = 0

    current_cons_weight = get_current_consistency_weight(
        epoch=epoch,
        base_weight=CONSISTENCY_CFG["weight"],
        rampup_epochs=CONSISTENCY_CFG["rampup_epochs"]
    )

    with torch.set_grad_enabled(True):
        for batch in loader:
            images_a = batch["image_a"].to(DEVICE, non_blocking=True)
            images_b = batch["image_b"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=AMP_ENABLED):
                logits_a = model(images_a)
                logits_b = model(images_b)

                sup_loss_a = criterion(logits_a, masks)
                sup_loss_b = criterion(logits_b, masks)
                sup_loss = 0.5 * (sup_loss_a + sup_loss_b)

            cons_loss = get_consistency_loss(logits_a, logits_b, masks)
            loss = sup_loss.float() + current_cons_weight * cons_loss.float()

            if not torch.isfinite(loss):
                skipped_batches += 1
                print(f"[WARN] epoch {epoch} 遇到非有限 loss，跳過此 batch")
                continue

            if AMP_ENABLED:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            batch_size = images_a.size(0)
            metrics = batch_metrics_from_logits(logits_a, masks, THRESHOLD, EPS)

            total_loss += loss.item() * batch_size
            total_sup_loss += sup_loss.item() * batch_size
            total_cons_loss += cons_loss.item() * batch_size
            total_dice += metrics["dice"] * batch_size
            total_iou += metrics["iou"] * batch_size
            total_precision += metrics["precision"] * batch_size
            total_recall += metrics["recall"] * batch_size
            total_count += batch_size

    if total_count == 0:
        return {
            "loss": 999.0,
            "sup_loss": 999.0,
            "cons_loss": 999.0,
            "cons_weight": current_cons_weight,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "skipped_batches": skipped_batches,
        }

    return {
        "loss": total_loss / total_count,
        "sup_loss": total_sup_loss / total_count,
        "cons_loss": total_cons_loss / total_count,
        "cons_weight": current_cons_weight,
        "dice": total_dice / total_count,
        "iou": total_iou / total_count,
        "precision": total_precision / total_count,
        "recall": total_recall / total_count,
        "skipped_batches": skipped_batches,
    }


def run_eval_epoch(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)

            with autocast(enabled=AMP_ENABLED):
                logits = model(images)
                loss = criterion(logits, masks)

            batch_size = images.size(0)
            metrics = batch_metrics_from_logits(logits, masks, THRESHOLD, EPS)

            total_loss += loss.item() * batch_size
            total_dice += metrics["dice"] * batch_size
            total_iou += metrics["iou"] * batch_size
            total_precision += metrics["precision"] * batch_size
            total_recall += metrics["recall"] * batch_size
            total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "dice": total_dice / total_count,
        "iou": total_iou / total_count,
        "precision": total_precision / total_count,
        "recall": total_recall / total_count,
    }


# =========================================================
# 12) 儲存 test prediction
# =========================================================
def save_test_predictions(model, loader, save_dir, threshold=0.5):
    model.eval()

    image_dir = os.path.join(save_dir, "image")
    gt_dir = os.path.join(save_dir, "mask_gt")
    pred_dir = os.path.join(save_dir, "mask_pred")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            stems = batch["stem"]
            image_paths = batch["image_path"]

            with autocast(enabled=AMP_ENABLED):
                logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(images.size(0)):
                stem = stems[i]

                raw_img = Image.open(image_paths[i]).convert("RGB")
                raw_img = raw_img.resize(IMAGE_SIZE, Image.BILINEAR)
                raw_img.save(os.path.join(image_dir, f"{stem}.png"))

                gt_mask = (masks_np[i, 0] * 255).astype(np.uint8)
                Image.fromarray(gt_mask).save(os.path.join(gt_dir, f"{stem}.png"))

                pred_mask = (preds_np[i, 0] * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(os.path.join(pred_dir, f"{stem}.png"))


# =========================================================
# 13) checkpoint 工具
# =========================================================
def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_dice,
    best_val_iou,
    best_epoch,
    history,
    epochs_no_improve
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "history": history,
        "epochs_no_improve": epochs_no_improve,
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_dice"],
        ckpt["best_val_iou"],
        ckpt["best_epoch"],
        ckpt["history"],
        ckpt["epochs_no_improve"]
    )


# =========================================================
# 14) 輸出 Stage2C metadata
# =========================================================
def write_stage2c_metadata(result_bundle_dir):
    config_path = os.path.join(result_bundle_dir, "consistency_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(CONSISTENCY_CFG, f, ensure_ascii=False, indent=2)

    notes_path = os.path.join(result_bundle_dir, "stage2_notes.txt")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("Stage 2-C (Consistency-only, improved self-consistency)\n")
        f.write("Base = SegFormer-B2 + LOCO (same Stage2A baseline augmentation)\n")
        f.write("Change = add paired-view self-consistency learning only\n")
        f.write("Perturbation Bank = OFF\n")
        f.write("Consistency = ON\n")
        f.write("Teacher-Student = OFF\n")
        f.write(f"Consistency Loss Type = {CONSISTENCY_CFG['loss_type']}\n")
        f.write("KD = OFF\n")
        f.write("Quantization = OFF\n")
        f.write("Validation/Test consistency = OFF\n")
        f.write("\n")
        f.write("Consistency Config:\n")
        f.write(json.dumps(CONSISTENCY_CFG, ensure_ascii=False, indent=2))


# =========================================================
# 15) 解析單個 fold 的 result.txt
# =========================================================
def parse_result_txt(result_txt_path):
    data = {}

    with open(result_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            data[key] = value

    required_keys = [
        "fold",
        "best_epoch",
        "best_val_dice",
        "best_val_iou",
        "test_loss",
        "test_dice",
        "test_iou",
        "test_precision",
        "test_recall",
        "delta_shift_dice",
        "delta_shift_iou",
        "time_sec",
    ]

    for k in required_keys:
        if k not in data:
            raise KeyError(f"{result_txt_path} 缺少欄位: {k}")

    return [
        data["fold"],
        int(data["best_epoch"]),
        float(data["best_val_dice"]),
        float(data["best_val_iou"]),
        float(data["test_loss"]),
        float(data["test_dice"]),
        float(data["test_iou"]),
        float(data["test_precision"]),
        float(data["test_recall"]),
        float(data["delta_shift_dice"]),
        float(data["delta_shift_iou"]),
        float(data["time_sec"]),
    ]


def rebuild_all_fold_results_from_disk():
    all_rows = []

    for fold_name in RUN_FOLDS:
        result_txt = os.path.join(OUT_ROOT, fold_name, "stage2C_results", "result.txt")
        if os.path.isfile(result_txt):
            row = parse_result_txt(result_txt)
            all_rows.append(row)
        else:
            print(f"[WARN] 找不到 result.txt，略過：{result_txt}")

    all_rows.sort(key=lambda x: x[0])

    summary_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "best_epoch", "best_val_dice", "best_val_iou",
            "test_loss", "test_dice", "test_iou", "test_precision", "test_recall",
            "delta_shift_dice", "delta_shift_iou", "time_sec"
        ])
        writer.writerows(all_rows)

    summary_txt = os.path.join(OUT_ROOT, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("LOCO Summary\n")
        f.write("========================================\n")

        if len(all_rows) > 0:
            for row in all_rows:
                f.write(
                    f"{row[0]} | best_epoch={row[1]} | best_val_dice={row[2]:.4f} | best_val_iou={row[3]:.4f} | "
                    f"test_loss={row[4]:.4f} | test_dice={row[5]:.4f} | test_iou={row[6]:.4f} | "
                    f"test_precision={row[7]:.4f} | test_recall={row[8]:.4f} | "
                    f"delta_shift_dice={row[9]:.4f} | delta_shift_iou={row[10]:.4f} | time_sec={row[11]:.2f}\n"
                )

            test_dices = [row[5] for row in all_rows]
            test_ious = [row[6] for row in all_rows]

            mean_test_dice = sum(test_dices) / len(test_dices)
            mean_test_iou = sum(test_ious) / len(test_ious)

            worst_row = min(all_rows, key=lambda x: x[5])
            best_row = max(all_rows, key=lambda x: x[5])

            worst_center = worst_row[0]
            worst_test_dice = worst_row[5]
            best_center = best_row[0]
            best_test_dice = best_row[5]
            center_gap_dice = best_test_dice - worst_test_dice

            f.write("\n")
            f.write(f"mean_test_dice: {mean_test_dice:.4f}\n")
            f.write(f"mean_test_iou: {mean_test_iou:.4f}\n")
            f.write(f"best_center: {best_center}\n")
            f.write(f"best_test_dice: {best_test_dice:.4f}\n")
            f.write(f"worst_center: {worst_center}\n")
            f.write(f"worst_test_dice: {worst_test_dice:.4f}\n")
            f.write(f"center_gap_dice: {center_gap_dice:.4f}\n")

        f.write("========================================\n")

    return summary_csv, all_rows


# =========================================================
# 16) Evaluation summary 工具
# =========================================================
def get_gt_pred_dirs(center_name):
    fold_name = f"fold_test_{center_name}"

    gt_dir = os.path.join(
        OUT_ROOT, fold_name, "stage2C_results", "test_predictions", "mask_gt"
    )
    pred_dir = os.path.join(
        OUT_ROOT, fold_name, "stage2C_results", "test_predictions", "mask_pred"
    )

    return fold_name, gt_dir, pred_dir


def list_png_stems(folder_path):
    stems = []
    if not os.path.isdir(folder_path):
        return stems

    for name in os.listdir(folder_path):
        if name.lower().endswith(".png"):
            stems.append(os.path.splitext(name)[0])

    stems.sort()
    return stems


def load_mask_as_bool(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask, dtype=np.uint8)
    return mask_np > 127


def compute_overlap_metrics(gt, pred, eps=1e-7):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(gt, pred).sum(dtype=np.float64)
    tn = np.logical_and(~gt, ~pred).sum(dtype=np.float64)
    fp = np.logical_and(~gt, pred).sum(dtype=np.float64)
    fn = np.logical_and(gt, ~pred).sum(dtype=np.float64)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
    }


def get_surface(mask):
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(mask)
    surface = np.logical_and(mask, np.logical_not(eroded))
    return surface


def compute_hd95_assd(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    gt_sum = gt.sum()
    pred_sum = pred.sum()

    if gt_sum == 0 and pred_sum == 0:
        return 0.0, 0.0

    if gt_sum == 0 or pred_sum == 0:
        return np.nan, np.nan

    gt_surface = get_surface(gt)
    pred_surface = get_surface(pred)

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    pred_to_gt = dt_gt[pred_surface]
    gt_to_pred = dt_pred[gt_surface]

    if pred_to_gt.size == 0 or gt_to_pred.size == 0:
        return np.nan, np.nan

    all_surface_distances = np.concatenate([pred_to_gt, gt_to_pred], axis=0)

    hd95 = np.percentile(all_surface_distances, 95)
    assd = (pred_to_gt.mean() + gt_to_pred.mean()) / 2.0

    return float(hd95), float(assd)


def evaluate_one_image(gt_path, pred_path):
    gt = load_mask_as_bool(gt_path)
    pred = load_mask_as_bool(pred_path)

    if gt.shape != pred.shape:
        raise ValueError(f"GT / Pred 尺寸不同:\n{gt_path}\n{pred_path}")

    overlap = compute_overlap_metrics(gt, pred, EPS)
    hd95, assd = compute_hd95_assd(gt, pred)

    return {
        "dice": overlap["dice"],
        "iou": overlap["iou"],
        "acc": overlap["acc"],
        "precision": overlap["precision"],
        "recall": overlap["recall"],
        "hd95": hd95,
        "assd": assd,
    }


def build_center_level_table():
    center_summary_rows = []
    per_image_rows = []

    run_centers = ["C1", "C2", "C3", "C4", "C5", "C6"]

    for center_name in run_centers:
        fold_name, gt_dir, pred_dir = get_gt_pred_dirs(center_name)

        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"找不到 GT 資料夾: {gt_dir}")
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError(f"找不到 Pred 資料夾: {pred_dir}")

        gt_stems = set(list_png_stems(gt_dir))
        pred_stems = set(list_png_stems(pred_dir))
        common_stems = sorted(gt_stems.intersection(pred_stems))

        if len(common_stems) == 0:
            raise RuntimeError(f"{fold_name} 沒有可配對的 mask_gt / mask_pred png 檔")

        dice_list = []
        iou_list = []
        acc_list = []
        precision_list = []
        recall_list = []
        hd95_list = []
        assd_list = []

        for stem in common_stems:
            gt_path = os.path.join(gt_dir, f"{stem}.png")
            pred_path = os.path.join(pred_dir, f"{stem}.png")

            metrics = evaluate_one_image(gt_path, pred_path)

            dice_list.append(metrics["dice"])
            iou_list.append(metrics["iou"])
            acc_list.append(metrics["acc"])
            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])
            hd95_list.append(metrics["hd95"])
            assd_list.append(metrics["assd"])

            per_image_rows.append([
                fold_name,
                center_name,
                stem,
                metrics["dice"],
                metrics["iou"],
                metrics["acc"],
                metrics["precision"],
                metrics["recall"],
                metrics["hd95"],
                metrics["assd"],
            ])

        summary_row = [
            center_name,
            len(common_stems),
            float(np.mean(dice_list)),
            float(np.mean(iou_list)),
            float(np.mean(acc_list)),
            float(np.mean(precision_list)),
            float(np.mean(recall_list)),
            float(np.nanmean(hd95_list)),
            float(np.nanmean(assd_list)),
        ]
        center_summary_rows.append(summary_row)

    per_image_csv = os.path.join(EVAL_OUT_DIR, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Fold", "Center", "Stem",
            "Dice", "IoU", "Acc", "Precision", "Recall", "HD95", "ASSD"
        ])
        writer.writerows(per_image_rows)

    per_image_txt = os.path.join(EVAL_OUT_DIR, "per_image_metrics.txt")
    with open(per_image_txt, "w", encoding="utf-8") as f:
        f.write("Fold\tCenter\tStem\tDice\tIoU\tAcc\tPrecision\tRecall\tHD95\tASSD\n")
        for row in per_image_rows:
            f.write(
                f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.6f}\t{row[4]:.6f}\t{row[5]:.6f}\t"
                f"{row[6]:.6f}\t{row[7]:.6f}\t{row[8]:.6f}\t{row[9]:.6f}\n"
            )

    center_csv = os.path.join(EVAL_OUT_DIR, "center_level_metrics.csv")
    with open(center_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Center", "Num_images", "Dice", "IoU", "Acc", "Precision", "Recall", "HD95", "ASSD"
        ])
        writer.writerows(center_summary_rows)

    center_txt = os.path.join(EVAL_OUT_DIR, "center_level_metrics.txt")
    with open(center_txt, "w", encoding="utf-8") as f:
        f.write("Center\tNum_images\tDice\tIoU\tAcc\tPrecision\tRecall\tHD95\tASSD\n")
        for row in center_summary_rows:
            f.write(
                f"{row[0]}\t{row[1]}\t{row[2]:.6f}\t{row[3]:.6f}\t{row[4]:.6f}\t"
                f"{row[5]:.6f}\t{row[6]:.6f}\t{row[7]:.6f}\t{row[8]:.6f}\n"
            )

    return center_summary_rows


def build_fold_val_data_table():
    all_fold_results_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")
    if not os.path.isfile(all_fold_results_csv):
        raise FileNotFoundError(f"找不到 all_fold_results.csv: {all_fold_results_csv}")

    fold_rows = []

    with open(all_fold_results_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_rows.append([
                row["fold"],
                float(row["best_val_dice"]),
                float(row["delta_shift_dice"]),
                float(row["best_val_iou"]),
                float(row["delta_shift_iou"]),
            ])

    fold_csv = os.path.join(EVAL_OUT_DIR, "fold_val_data_metrics.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"])
        writer.writerows(fold_rows)

    fold_txt = os.path.join(EVAL_OUT_DIR, "fold_val_data_metrics.txt")
    with open(fold_txt, "w", encoding="utf-8") as f:
        f.write("Fold\tVal_Dice\tΔDice\tVal_IoU\tΔIoU\n")
        for row in fold_rows:
            f.write(f"{row[0]}\t{row[1]:.6f}\t{row[2]:.6f}\t{row[3]:.6f}\t{row[4]:.6f}\n")

    return fold_rows


def export_all_fold_results_into_eval_summary():
    src_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")
    if not os.path.isfile(src_csv):
        raise FileNotFoundError(f"找不到 all_fold_results.csv: {src_csv}")

    dst_csv = os.path.join(EVAL_OUT_DIR, "all_fold_results.csv")
    shutil.copy2(src_csv, dst_csv)

    dst_txt = os.path.join(EVAL_OUT_DIR, "all_fold_results.txt")
    with open(src_csv, "r", encoding="utf-8-sig") as fin, open(dst_txt, "w", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        for row in reader:
            fout.write("\t".join(row) + "\n")


def build_eval_summary():
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

    center_rows = build_center_level_table()
    fold_rows = build_fold_val_data_table()
    export_all_fold_results_into_eval_summary()

    print("=" * 70)
    print("Stage2C_eval_summary 輸出完成")
    print("=" * 70)
    print("1) per_image_metrics.csv / .txt")
    print("2) center_level_metrics.csv / .txt")
    print("3) fold_val_data_metrics.csv / .txt")
    print("4) all_fold_results.csv / .txt")
    print(f"EVAL_OUT_DIR = {EVAL_OUT_DIR}")

    return center_rows, fold_rows


# =========================================================
# 17) 主程式：逐 fold 訓練
# =========================================================
if __name__ == "__main__":
    set_seed(SEED)

    for fold_name in RUN_FOLDS:
        print("========================================")
        print("Training", fold_name)
        print("========================================")

        fold_dir = os.path.join(LOCO_ROOT, fold_name)
        save_dir = os.path.join(OUT_ROOT, fold_name)

        result_bundle_dir = os.path.join(save_dir, "stage2C_results")
        pred_save_dir = os.path.join(result_bundle_dir, "test_predictions")

        result_txt = os.path.join(result_bundle_dir, "result.txt")
        history_csv = os.path.join(result_bundle_dir, "history.csv")
        checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
        best_model_path = os.path.join(save_dir, "best_model.pth")

        if SKIP_COMPLETED_FOLDS and os.path.exists(result_txt) and fold_name not in RESET_FOLDS:
            print(f"[{fold_name}] already completed -> skip")
            print()
            continue

        if fold_name in RESET_FOLDS and os.path.exists(save_dir):
            print(f"[{fold_name}] RESET -> remove old folder")
            shutil.rmtree(save_dir)

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(result_bundle_dir, exist_ok=True)

        write_stage2c_metadata(result_bundle_dir)

        train_csv = os.path.join(fold_dir, "train.csv")
        val_csv = os.path.join(fold_dir, "val.csv")
        test_csv = os.path.join(fold_dir, "test.csv")

        train_dataset = PolypDataset(
            train_csv,
            IMAGE_SIZE,
            is_train=True,
            use_consistency=True
        )
        val_dataset = PolypDataset(
            val_csv,
            IMAGE_SIZE,
            is_train=False,
            use_consistency=False
        )
        test_dataset = PolypDataset(
            test_csv,
            IMAGE_SIZE,
            is_train=False,
            use_consistency=False
        )

        print(f"train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY if DEVICE == "cuda" else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY if DEVICE == "cuda" else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY if DEVICE == "cuda" else False
        )

        model = SegFormerBinary(
            use_pretrained=USE_PRETRAINED,
            pretrained_name=PRETRAINED_MODEL_NAME,
        ).to(DEVICE)

        criterion = DiceBCELoss()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=MIN_LR
        )

        scaler = GradScaler(enabled=AMP_ENABLED)

        best_val_dice = -1.0
        best_val_iou = -1.0
        best_epoch = -1
        history = []
        start_epoch = 1
        epochs_no_improve = 0

        if os.path.exists(checkpoint_path) and fold_name not in RESET_FOLDS:
            start_epoch, best_val_dice, best_val_iou, best_epoch, history, epochs_no_improve = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            print(f"[{fold_name}] resume from epoch {start_epoch}")
        else:
            print(f"[{fold_name}] start from scratch")

        start_time = time.time()

        epoch_pbar = tqdm(
            range(start_epoch, NUM_EPOCHS + 1),
            desc=fold_name,
            leave=True,
            ncols=180
        )

        for epoch in epoch_pbar:
            train_metrics = run_train_epoch_consistency(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch
            )

            val_metrics = run_eval_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion
            )

            current_lr = optimizer.param_groups[0]["lr"]

            history.append([
                epoch,
                current_lr,
                train_metrics["loss"],
                train_metrics["sup_loss"],
                train_metrics["cons_loss"],
                train_metrics["cons_weight"],
                train_metrics["dice"],
                train_metrics["iou"],
                train_metrics["precision"],
                train_metrics["recall"],
                val_metrics["loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"]
            ])

            scheduler.step(val_metrics["dice"])

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_val_iou = val_metrics["iou"]
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_dice=best_val_dice,
                best_val_iou=best_val_iou,
                best_epoch=best_epoch,
                history=history,
                epochs_no_improve=epochs_no_improve
            )

            epoch_pbar.set_postfix({
                "lr": f"{current_lr:.2e}",
                "tr_dice": f"{train_metrics['dice']:.4f}",
                "va_dice": f"{val_metrics['dice']:.4f}",
                "tr_iou": f"{train_metrics['iou']:.4f}",
                "va_iou": f"{val_metrics['iou']:.4f}",
                "c_w": f"{train_metrics['cons_weight']:.3f}",
                "best": f"{best_val_dice:.4f}",
                "skip_b": f"{train_metrics.get('skipped_batches', 0)}",
                "no_imp": f"{epochs_no_improve}/{EARLY_STOPPING_PATIENCE}",
            })

            print(
                f"[{fold_name}] Epoch {epoch:03d}/{NUM_EPOCHS} | "
                f"lr={current_lr:.6e} | "
                f"train_total_loss={train_metrics['loss']:.4f} "
                f"train_sup_loss={train_metrics['sup_loss']:.4f} "
                f"train_cons_loss={train_metrics['cons_loss']:.4f} "
                f"cons_w={train_metrics['cons_weight']:.4f} "
                f"train_dice={train_metrics['dice']:.4f} "
                f"train_iou={train_metrics['iou']:.4f} "
                f"skipped_batches={train_metrics.get('skipped_batches', 0)} | "
                f"valid_loss={val_metrics['loss']:.4f} "
                f"valid_dice={val_metrics['dice']:.4f} "
                f"valid_iou={val_metrics['iou']:.4f} "
                f"valid_precision={val_metrics['precision']:.4f} "
                f"valid_recall={val_metrics['recall']:.4f} | "
                f"best_valid_dice={best_val_dice:.4f} @ epoch {best_epoch}"
            )

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"[{fold_name}] Early stopping triggered.")
                break

        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        test_metrics = run_eval_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion
        )

        save_test_predictions(model, test_loader, pred_save_dir, THRESHOLD)

        elapsed = time.time() - start_time

        delta_shift_dice = best_val_dice - test_metrics["dice"]
        delta_shift_iou = best_val_iou - test_metrics["iou"]

        with open(history_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "lr",
                "train_total_loss", "train_sup_loss", "train_cons_loss", "consistency_weight",
                "train_dice", "train_iou", "train_precision", "train_recall",
                "valid_loss", "valid_dice", "valid_iou", "valid_precision", "valid_recall"
            ])
            writer.writerows(history)

        with open(result_txt, "w", encoding="utf-8") as f:
            f.write(f"model_name: {MODEL_NAME}\n")
            f.write(f"stage: Stage 2-C (Consistency-only, improved self-consistency)\n")
            f.write(f"fold: {fold_name}\n")
            f.write(f"best_epoch: {best_epoch}\n")
            f.write(f"best_val_dice: {best_val_dice:.6f}\n")
            f.write(f"best_val_iou: {best_val_iou:.6f}\n")
            f.write(f"test_loss: {test_metrics['loss']:.6f}\n")
            f.write(f"test_dice: {test_metrics['dice']:.6f}\n")
            f.write(f"test_iou: {test_metrics['iou']:.6f}\n")
            f.write(f"test_precision: {test_metrics['precision']:.6f}\n")
            f.write(f"test_recall: {test_metrics['recall']:.6f}\n")
            f.write(f"delta_shift_dice: {delta_shift_dice:.6f}\n")
            f.write(f"delta_shift_iou: {delta_shift_iou:.6f}\n")
            f.write(f"time_sec: {elapsed:.2f}\n")
            f.write(f"prediction_dir: {pred_save_dir}\n")

        if DELETE_CHECKPOINT_AFTER_FINISH and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        print("------------------------------------------------------------")
        print(f"[{fold_name}] TEST RESULT")
        print(f"best_epoch       = {best_epoch}")
        print(f"best_val_dice    = {best_val_dice:.4f}")
        print(f"best_val_iou     = {best_val_iou:.4f}")
        print(f"test_loss        = {test_metrics['loss']:.4f}")
        print(f"test_dice        = {test_metrics['dice']:.4f}")
        print(f"test_iou         = {test_metrics['iou']:.4f}")
        print(f"test_precision   = {test_metrics['precision']:.4f}")
        print(f"test_recall      = {test_metrics['recall']:.4f}")
        print(f"delta_shift_dice = {delta_shift_dice:.4f}")
        print(f"delta_shift_iou  = {delta_shift_iou:.4f}")
        print(f"result_bundle_dir= {result_bundle_dir}")
        print(f"prediction_dir   = {pred_save_dir}")
        print("------------------------------------------------------------")
        print()

    summary_csv, all_rows = rebuild_all_fold_results_from_disk()

    print("========================================")
    print("Current run finished")
    print("========================================")
    print(f"summary csv: {summary_csv}")

    if len(all_rows) > 0:
        for row in all_rows:
            print(
                f"{row[0]} | best_epoch={row[1]} | best_val_dice={row[2]:.4f} | "
                f"test_dice={row[5]:.4f} | test_iou={row[6]:.4f} | "
                f"test_precision={row[7]:.4f} | test_recall={row[8]:.4f} | "
                f"delta_shift_dice={row[9]:.4f}"
            )
    else:
        print("沒有任何已完成 fold 可供彙總。")

    build_eval_summary()
