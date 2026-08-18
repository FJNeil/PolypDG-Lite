# ============================================================
# Stage 3-E
# SegFormer-B0 + Center-level LOCO + Knowledge Distillation
# ------------------------------------------------------------
# 說明：
# 1. Student：SegFormer-B0
# 2. Teacher：Stage 2C SegFormer-B2 + Consistency Learning
# 3. 使用 Center-level LOCO
# 4. 前處理與 SegFormer-B0 + LOCO baseline 對齊
# 5. 不含 Consistency Learning
# 6. 含 KD：teacher logits -> student logits
# 7. 輸出：
#    - all_fold_results.csv
#    - fold_val_data_metrics.csv
#    - center_level_metrics.csv
#    - per_image_metrics_all.csv
#    - loco_summary.txt
# ============================================================

import os
import csv
import time
import math
import random
import shutil
import traceback
import warnings
import gc
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance

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
    raise ImportError("請先安裝 transformers：pip install transformers") from e


# ============================================================
# 0) Main Config
# ============================================================

# LOCO split 根目錄
LOCO_ROOT = os.environ.get("POLYPDG_LOCO_ROOT", "./data/loco_1537_clean")

# Stage 2 teacher 根目錄
# 這裡要放 SegFormer-B2 + Consistency Learning 的 6-fold teacher
TEACHER_ROOT = os.environ.get("POLYPDG_TEACHER_ROOT", "./outputs/stage2_teacher")

# 每次自動開新資料夾，避免混到舊結果
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")

# 輸出根目錄
OUT_ROOT = os.environ.get("POLYPDG_STAGE3_OUT", f"./outputs/stage3_student_{RUN_STAMP}")

# 最後彙整結果複製到這裡
EVAL_OUT_DIR = os.path.join(OUT_ROOT, "Stage3E_eval_summary")

# 要跑的 folds
RUN_FOLDS = [
    "fold_test_C1",
    "fold_test_C2",
    "fold_test_C3",
    "fold_test_C4",
    "fold_test_C5",
    "fold_test_C6",
]

# 影像尺寸，保持與 SegFormer-B0 + LOCO baseline 一樣
IMAGE_SIZE = (352, 352)

# KD 同時載入 teacher + student，RTX 3050 建議先用 4
BATCH_SIZE = 4

# 訓練 epoch
NUM_EPOCHS = 100

# SegFormer-B0 KD 建議先用 1e-4
LR = 1e-4

# weight decay
WEIGHT_DECAY = 1e-4

# early stopping
EARLY_STOPPING_PATIENCE = 12

# ReduceLROnPlateau
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# dataloader workers
NUM_WORKERS = 0

# pin memory
PIN_MEMORY = True

# segmentation threshold
THRESHOLD = 0.5

# 小常數
EPS = 1e-7

# AMP
USE_AMP = True

# seed
SEED = 42

# gradient clipping
GRAD_CLIP_NORM = 1.0

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 只有 GPU 才開 AMP
AMP_ENABLED = USE_AMP and DEVICE == "cuda"

# 是否儲存 test prediction
SAVE_TEST_PRED = True

# 是否輸出每個 fold 的 center table
SAVE_PER_FOLD_CENTER_TABLE = True

# 不做 resume，避免不同版本混在一起
ENABLE_RESUME = False

# model name
MODEL_NAME = "SegFormer-B0 + LOCO + KD"

# student implementation
STUDENT_IMPL = "segformer_b0"

# teacher checkpoint 名稱
TEACHER_CKPT_NAME = "best_model.pth"

# teacher pretrained name
TEACHER_PRETRAINED_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

# student pretrained name
STUDENT_PRETRAINED_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# 是否載入 student 預訓練權重
LOAD_STUDENT_PRETRAINED = True

# KD 設定
KD_CFG = {
    "enable": True,
    "temperature": 4.0,
    "base_weight": 0.05,
    "warmup_no_kd_epochs": 10,
    "rampup_epochs": 20,
    "teacher_eval_mode": True,
}

# 建立輸出資料夾
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)


# ============================================================
# 1) Seed
# ============================================================

def set_seed(seed=42):
    # 固定 Python random
    random.seed(seed)

    # 固定 NumPy random
    np.random.seed(seed)

    # 固定 PyTorch CPU random
    torch.manual_seed(seed)

    # 固定 CUDA random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 讓 cudnn 更可重現
    torch.backends.cudnn.deterministic = True

    # 關閉 benchmark，避免每次自動選不同 kernel
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Runtime info
# ============================================================

def print_runtime_info():
    print("=" * 100)
    print("Stage 3-E: SegFormer-B0 + LOCO + KD")
    print("=" * 100)
    print("DEVICE =", DEVICE)

    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))
        print("torch.version.cuda =", torch.version.cuda)

    print("MODEL_NAME =", MODEL_NAME)
    print("STUDENT_IMPL =", STUDENT_IMPL)
    print("LOCO_ROOT =", LOCO_ROOT)
    print("TEACHER_ROOT =", TEACHER_ROOT)
    print("OUT_ROOT =", OUT_ROOT)
    print("EVAL_OUT_DIR =", EVAL_OUT_DIR)
    print("RUN_FOLDS =", RUN_FOLDS)
    print("IMAGE_SIZE =", IMAGE_SIZE)
    print("BATCH_SIZE =", BATCH_SIZE)
    print("NUM_EPOCHS =", NUM_EPOCHS)
    print("LR =", LR)
    print("WEIGHT_DECAY =", WEIGHT_DECAY)
    print("EARLY_STOPPING_PATIENCE =", EARLY_STOPPING_PATIENCE)
    print("SCHEDULER_PATIENCE =", SCHEDULER_PATIENCE)
    print("AMP_ENABLED =", AMP_ENABLED)
    print("GRAD_CLIP_NORM =", GRAD_CLIP_NORM)
    print("ENABLE_RESUME =", ENABLE_RESUME)
    print("STUDENT_PRETRAINED_NAME =", STUDENT_PRETRAINED_NAME)
    print("TEACHER_PRETRAINED_NAME =", TEACHER_PRETRAINED_NAME)
    print("KD_CFG =", KD_CFG)
    print("=" * 100)


# ============================================================
# 3) CSV / fold helpers
# ============================================================

def find_split_csv(fold_dir: str, split_name: str) -> str:
    # 先找常見命名
    candidates = [
        os.path.join(fold_dir, f"{split_name}.csv"),
        os.path.join(fold_dir, f"{split_name}_data.csv"),
        os.path.join(fold_dir, f"{split_name}_split.csv"),
        os.path.join(fold_dir, "csv", f"{split_name}.csv"),
        os.path.join(fold_dir, "splits", f"{split_name}.csv"),
    ]

    # 若存在就回傳
    for path in candidates:
        if os.path.isfile(path):
            return path

    # 若不在常見位置，就用 os.walk 搜尋
    for root, _, files in os.walk(fold_dir):
        for file_name in files:
            lower = file_name.lower()

            if lower.endswith(".csv") and split_name in lower:
                return os.path.join(root, file_name)

    raise FileNotFoundError(f"找不到 {split_name}.csv，fold_dir={fold_dir}")


# ============================================================
# 4) Preprocessing / augmentation
#    對齊你 SegFormer-B0 + LOCO baseline / DDRNet LOCO
# ============================================================

def resize_pair(img, mask, image_size):
    # image 使用 bilinear
    img = img.resize(image_size, Image.BILINEAR)

    # mask 使用 nearest，避免 label 被插值污染
    mask = mask.resize(image_size, Image.NEAREST)

    return img, mask


def random_hflip_pair(img, mask, p=0.5):
    # 隨機水平翻轉
    if random.random() < p:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)

    return img, mask


def random_vflip_pair(img, mask, p=0.5):
    # 隨機垂直翻轉
    if random.random() < p:
        img = ImageOps.flip(img)
        mask = ImageOps.flip(mask)

    return img, mask


def random_rotate_pair(img, mask, p=0.5, angle_limit=15):
    # 隨機小角度旋轉
    if random.random() < p:
        angle = random.uniform(-angle_limit, angle_limit)

        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

    return img, mask


def random_scale_shift_pair(img, mask, p=0.5, scale_low=0.90, scale_high=1.10):
    # 若未觸發 augmentation，直接回傳
    if random.random() >= p:
        return img, mask

    # 取得原始寬高
    w, h = img.size

    # 隨機縮放倍率
    scale = random.uniform(scale_low, scale_high)

    # 計算新尺寸
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # image bilinear resize
    img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

    # mask nearest resize
    mask_scaled = mask.resize((new_w, new_h), Image.NEAREST)

    # 放大時 crop 回原尺寸
    if scale >= 1.0:
        max_left = max(0, new_w - w)
        max_top = max(0, new_h - h)

        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0

        img_out = img_scaled.crop((left, top, left + w, top + h))
        mask_out = mask_scaled.crop((left, top, left + w, top + h))

        return img_out, mask_out

    # 縮小時貼回黑底 canvas
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
    # 隨機 brightness
    if random.random() < p:
        brightness_factor = random.uniform(
            1.0 - brightness_limit,
            1.0 + brightness_limit
        )
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 隨機 contrast
    if random.random() < p:
        contrast_factor = random.uniform(
            1.0 - contrast_limit,
            1.0 + contrast_limit
        )
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def image_to_chw_float(img):
    # PIL RGB -> numpy float32, [0,1]
    img_np = np.array(img, dtype=np.float32) / 255.0

    # HWC -> CHW
    img_np = np.transpose(img_np, (2, 0, 1))

    return img_np


def mask_to_1hw_float(mask):
    # mask -> numpy
    mask_np = np.array(mask, dtype=np.uint8)

    # 二值化
    mask_np = (mask_np > 127).astype(np.float32)

    # HxW -> 1xHxW
    mask_np = np.expand_dims(mask_np, axis=0)

    return mask_np


def preprocess_single_view(img, mask, image_size=(352, 352), is_train=False):
    # resize
    img, mask = resize_pair(img, mask, image_size)

    # train 才做 augmentation
    if is_train:
        img, mask = random_hflip_pair(img, mask, p=0.5)
        img, mask = random_vflip_pair(img, mask, p=0.5)
        img, mask = random_rotate_pair(img, mask, p=0.5, angle_limit=15)

        img, mask = random_scale_shift_pair(
            img,
            mask,
            p=0.5,
            scale_low=0.90,
            scale_high=1.10
        )

        img = random_brightness_contrast(
            img,
            p=0.4,
            brightness_limit=0.15,
            contrast_limit=0.15
        )

    # image -> CHW float
    img_np = image_to_chw_float(img)

    # mask -> 1HW float
    mask_np = mask_to_1hw_float(mask)

    return img_np, mask_np


# ============================================================
# 5) Dataset
# ============================================================

class PolypDataset(Dataset):
    def __init__(self, csv_file, image_size=(352, 352), is_train=False):
        # 儲存資料 rows
        self.rows = []

        # 儲存 image size
        self.image_size = image_size

        # 是否為訓練模式
        self.is_train = is_train

        # 讀 csv
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                self.rows.append(row)

    def __len__(self):
        # 回傳資料筆數
        return len(self.rows)

    def __getitem__(self, idx):
        # 取出 row
        row = self.rows[idx]

        # 讀 image
        img = Image.open(row["image_path"]).convert("RGB")

        # 讀 mask
        mask = Image.open(row["mask_path"]).convert("L")

        # 前處理
        img_np, mask_np = preprocess_single_view(
            img=img,
            mask=mask,
            image_size=self.image_size,
            is_train=self.is_train
        )

        # stem 若 csv 沒有，就從檔名取得
        stem = row.get("stem", os.path.splitext(os.path.basename(row["image_path"]))[0])

        # center 若 csv 沒有，就 UNKNOWN
        center = row["center"] if "center" in row else "UNKNOWN"

        return {
            "image": torch.tensor(img_np, dtype=torch.float32),
            "mask": torch.tensor(mask_np, dtype=torch.float32),
            "stem": stem,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "center": center,
        }


# ============================================================
# 6) Teacher model: SegFormer-B2 binary
# ============================================================

class SegFormerB2BinaryTeacher(nn.Module):
    def __init__(self, use_pretrained=True, pretrained_name=TEACHER_PRETRAINED_NAME):
        super().__init__()

        # 是否載入 pretrained
        if use_pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_name,
                    num_labels=1,
                    ignore_mismatched_sizes=True,
                )

                print(f"[Teacher] loaded pretrained weights from: {pretrained_name}")

            except Exception as e:
                print("[Teacher][WARN] pretrained load failed, fallback to scratch B2 config")
                print("reason:", e)

                self.model = self._build_b2_from_scratch()

        else:
            self.model = self._build_b2_from_scratch()

    def _build_b2_from_scratch(self):
        # SegFormer-B2 config
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 記錄輸入尺寸
        input_h, input_w = x.shape[-2], x.shape[-1]

        # forward
        outputs = self.model(pixel_values=x)

        # 取得 logits
        logits = outputs.logits

        # 上採樣回輸入尺寸
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False
            )

        return logits


def normalize_state_dict_keys(state_dict: Dict) -> Dict:
    # 移除 DataParallel 的 module. prefix
    new_state = {}

    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v

    return new_state


def extract_state_dict(ckpt: Dict) -> Dict:
    # 支援多種 checkpoint key
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]

        if "state_dict" in ckpt:
            return ckpt["state_dict"]

        if "student_state_dict" in ckpt:
            return ckpt["student_state_dict"]

    return ckpt


def load_teacher_for_fold(fold_name: str) -> nn.Module:
    # teacher checkpoint path
    ckpt_path = os.path.join(TEACHER_ROOT, fold_name, TEACHER_CKPT_NAME)

    # 檢查 teacher checkpoint 是否存在
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 teacher checkpoint：{ckpt_path}")

    # 建立 teacher
    teacher = SegFormerB2BinaryTeacher(
        use_pretrained=True,
        pretrained_name=TEACHER_PRETRAINED_NAME
    ).to(DEVICE)

    # 載入 checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # 取出 state dict
    state_dict = normalize_state_dict_keys(extract_state_dict(ckpt))

    # 嘗試 strict load
    try:
        teacher.load_state_dict(state_dict, strict=True)
        print(f"[Teacher] strict load success for {fold_name}")

    except RuntimeError as e:
        print(f"[Teacher][WARN] strict load failed for {fold_name}, fallback strict=False")
        print("reason:", e)

        load_result = teacher.load_state_dict(state_dict, strict=False)

        print("[Teacher][WARN] missing_keys:", load_result.missing_keys[:20])
        print("[Teacher][WARN] unexpected_keys:", load_result.unexpected_keys[:20])

    # teacher 設為 eval
    teacher.eval()

    # teacher 不更新
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[Teacher] fold={fold_name} loaded from: {ckpt_path}")

    return teacher


# ============================================================
# 7) Student model: SegFormer-B0 binary
# ============================================================

class SegFormerB0BinaryStudent(nn.Module):
    def __init__(
        self,
        use_pretrained=True,
        pretrained_name=STUDENT_PRETRAINED_NAME
    ):
        super().__init__()

        # 是否載入 pretrained
        if use_pretrained:
            try:
                print(f"[Student] loading pretrained SegFormer-B0 from: {pretrained_name}")

                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_name,
                    num_labels=1,
                    ignore_mismatched_sizes=True
                )

                print("[Student] pretrained SegFormer-B0 loaded successfully.")

            except Exception as e:
                print("[Student][WARN] pretrained load failed, fallback to scratch B0 config")
                print("reason:", e)

                self.model = self._build_b0_from_scratch()

        else:
            print("[Student] using scratch SegFormer-B0 config")
            self.model = self._build_b0_from_scratch()

    def _build_b0_from_scratch(self):
        # SegFormer-B0 config
        config = SegformerConfig(
            num_labels=1,
            num_channels=3,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 取得輸入尺寸
        input_h, input_w = x.shape[-2], x.shape[-1]

        # forward
        outputs = self.model(pixel_values=x)

        # 取得 logits
        logits = outputs.logits

        # 上採樣回輸入解析度
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False
            )

        return logits


def build_student_model():
    # 建立 SegFormer-B0 student
    model = SegFormerB0BinaryStudent(
        use_pretrained=LOAD_STUDENT_PRETRAINED,
        pretrained_name=STUDENT_PRETRAINED_NAME
    )

    return model, STUDENT_IMPL


# ============================================================
# 8) Supervised Loss
# ============================================================

class DiceBCELossWithParts(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()

        # BCE loss
        self.bce = nn.BCEWithLogitsLoss()

        # dice smooth
        self.smooth = smooth

        # BCE 權重
        self.bce_weight = bce_weight

        # Dice 權重
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        # BCE
        bce = self.bce(logits, targets)

        # sigmoid
        probs = torch.sigmoid(logits)

        # flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # intersection
        intersection = (probs * targets).sum(dim=1)

        # Dice
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        # Dice loss
        dice_loss = 1.0 - dice.mean()

        # total supervised loss
        total = self.bce_weight * bce + self.dice_weight * dice_loss

        return total, bce.detach(), dice_loss.detach()


# ============================================================
# 9) KD helpers
# ============================================================

def binary_prob_to_two_class(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 限制 probability 範圍，避免 log(0)
    prob = prob.clamp(eps, 1.0 - eps)

    # background probability
    bg = (1.0 - prob).clamp(eps, 1.0 - eps)

    # foreground probability
    fg = prob

    # [B,1,H,W] + [B,1,H,W] -> [B,2,H,W]
    return torch.cat([bg, fg], dim=1)


def kd_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    eps: float = 1e-6
) -> torch.Tensor:
    # student logits 經 temperature sigmoid
    s_prob = torch.sigmoid(student_logits / temperature)

    # teacher logits 經 temperature sigmoid
    t_prob = torch.sigmoid(teacher_logits / temperature)

    # binary probability 轉成 two-class distribution
    s_dist = binary_prob_to_two_class(s_prob, eps)
    t_dist = binary_prob_to_two_class(t_prob, eps)

    # student log probability
    s_log = torch.log(s_dist.clamp_min(eps))

    # KL divergence map
    kl_map = F.kl_div(
        s_log,
        t_dist,
        reduction="none"
    )

    # sum over class dimension
    kl_map = kl_map.sum(dim=1)

    # mean over B,H,W
    kd = kl_map.mean()

    # temperature scaling
    kd = kd * (temperature ** 2)

    # 避免 nan / inf
    kd = torch.nan_to_num(
        kd,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return kd


def get_current_kd_weight(
    epoch: int,
    base_weight: float,
    warmup_epochs: int,
    rampup_epochs: int
) -> float:
    # warmup 階段不使用 KD
    if epoch <= warmup_epochs:
        return 0.0

    # 若 rampup_epochs <= 0，直接使用 base weight
    if rampup_epochs <= 0:
        return base_weight

    # 計算 rampup progress
    effective_epoch = epoch - warmup_epochs

    # 限制在 [0,1]
    factor = min(1.0, effective_epoch / float(rampup_epochs))

    # cosine rampup
    factor = 0.5 - 0.5 * math.cos(math.pi * factor)

    return base_weight * factor


# ============================================================
# 10) Metrics
# ============================================================

def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    # logits -> probs
    probs = torch.sigmoid(logits)

    # probs -> binary mask
    preds = (probs > threshold).float()

    # flatten
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # TP / FP / FN
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    # pred / target 前景數量
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)

    # pred 和 gt 都空的情況
    both_empty = (pred_sum == 0) & (target_sum == 0)

    # dice / iou / precision / recall
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    # 空白影像特例視為 perfect
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    iou = torch.where(both_empty, torch.ones_like(iou), iou)
    precision = torch.where(both_empty, torch.ones_like(precision), precision)
    recall = torch.where(both_empty, torch.ones_like(recall), recall)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


def safe_div(n: float, d: float) -> float:
    # 安全除法
    return float(n) / float(d) if d > 0 else 0.0


def binary_surface(mask: np.ndarray) -> np.ndarray:
    # 若沒有前景，回傳全 False
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    # erosion
    eroded = binary_erosion(mask)

    # surface = mask XOR eroded
    return mask.astype(bool) ^ eroded.astype(bool)


def hd95_and_assd(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, bool]:
    # 轉 bool
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 兩邊都空，視為 perfect
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0, True

    # 一空一不空，邊界距離不可用
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan, False

    # 取 surface
    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)

    # surface 都空，視為 perfect
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0, 0.0, True

    # 任一 surface 空，距離不可用
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return np.nan, np.nan, False

    # distance transform
    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    # pred surface 到 gt surface
    dist_pred_to_gt = dt_gt[pred_surface]

    # gt surface 到 pred surface
    dist_gt_to_pred = dt_pred[gt_surface]

    # 若距離空，回傳 nan
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan, np.nan, False

    # 合併雙向距離
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)

    # HD95
    hd95 = np.percentile(all_dists, 95)

    # ASSD
    assd = np.mean(all_dists)

    return float(hd95), float(assd), True


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # 轉 uint8
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # 前景數量
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    # 兩邊都空
    if pred_sum == 0 and gt_sum == 0:
        return {
            "Dice": 1.0,
            "IoU": 1.0,
            "Acc": 1.0,
            "Precision": 1.0,
            "Recall": 1.0,
            "HD95": 0.0,
            "ASSD": 0.0,
            "Valid_Boundary_Case": 1,
        }

    # confusion matrix
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    # metrics
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    # boundary metrics
    hd95, assd, valid_boundary = hd95_and_assd(pred, gt)

    return {
        "Dice": dice,
        "IoU": iou,
        "Acc": acc,
        "Precision": precision,
        "Recall": recall,
        "HD95": hd95,
        "ASSD": assd,
        "Valid_Boundary_Case": int(valid_boundary),
    }


# ============================================================
# 11) Checkpoint utils
# ============================================================

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
    epochs_no_improve,
    student_impl,
):
    # 儲存完整 checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "history": history,
            "epochs_no_improve": epochs_no_improve,
            "student_impl": student_impl,
        },
        checkpoint_path
    )


# ============================================================
# 12) Train / Eval
# ============================================================

def run_train_epoch_kd(model, teacher, loader, criterion, optimizer, scaler, epoch):
    # student train mode
    model.train()

    # teacher eval mode
    if KD_CFG["teacher_eval_mode"]:
        teacher.eval()

    # 累加變數
    total_loss = 0.0
    total_sup_loss = 0.0
    total_kd_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_count = 0
    skipped_batches = 0

    # 取得目前 epoch 的 KD 權重
    current_kd_weight = get_current_kd_weight(
        epoch=epoch,
        base_weight=KD_CFG["base_weight"],
        warmup_epochs=KD_CFG["warmup_no_kd_epochs"],
        rampup_epochs=KD_CFG["rampup_epochs"],
    )

    for batch in loader:
        # 取 image / mask
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        # 清空 gradient
        optimizer.zero_grad(set_to_none=True)

        # AMP forward
        with autocast(enabled=AMP_ENABLED):
            # student logits
            student_logits = model(images)

            # supervised loss
            sup_loss, batch_bce, batch_dice_loss = criterion(student_logits, masks)

            # 預設 kd loss = 0
            kd_loss = torch.tensor(0.0, device=DEVICE, dtype=sup_loss.dtype)

            # 若 KD 啟動且 KD weight > 0
            if KD_CFG["enable"] and current_kd_weight > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(images)

                kd_loss = kd_kl_from_logits(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=KD_CFG["temperature"],
                )

            # total loss = supervised + KD
            loss = sup_loss + current_kd_weight * kd_loss

        # 若 loss 非有限，跳過該 batch
        if not torch.isfinite(loss):
            skipped_batches += 1
            print("[WARN] 遇到非有限 loss，跳過此 batch")
            continue

        # backward
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

        # batch size
        batch_size = images.size(0)

        # 計算 batch metrics
        metrics = batch_metrics_from_logits(student_logits, masks, THRESHOLD, EPS)

        # 累加
        total_loss += loss.item() * batch_size
        total_sup_loss += sup_loss.item() * batch_size
        total_kd_loss += kd_loss.item() * batch_size
        total_bce += batch_bce.item() * batch_size
        total_dice_loss += batch_dice_loss.item() * batch_size
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size
        total_precision += metrics["precision"] * batch_size
        total_recall += metrics["recall"] * batch_size
        total_count += batch_size

    # 防止全部 batch 都跳過
    if total_count == 0:
        return {
            "loss": 999.0,
            "sup_loss": 999.0,
            "kd_loss": 0.0,
            "bce": 0.0,
            "dice_loss": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "kd_weight": current_kd_weight,
            "skipped_batches": skipped_batches,
        }

    return {
        "loss": total_loss / total_count,
        "sup_loss": total_sup_loss / total_count,
        "kd_loss": total_kd_loss / total_count,
        "bce": total_bce / total_count,
        "dice_loss": total_dice_loss / total_count,
        "dice": total_dice / total_count,
        "iou": total_iou / total_count,
        "precision": total_precision / total_count,
        "recall": total_recall / total_count,
        "kd_weight": current_kd_weight,
        "skipped_batches": skipped_batches,
    }


@torch.no_grad()
def run_eval_epoch(model, loader, criterion):
    # eval mode
    model.eval()

    # 累加變數
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_count = 0

    for batch in loader:
        # 取 image / mask
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        # AMP forward
        with autocast(enabled=AMP_ENABLED):
            logits = model(images)
            loss, bce, dice_loss = criterion(logits, masks)

        # batch size
        batch_size = images.size(0)

        # batch metrics
        metrics = batch_metrics_from_logits(logits, masks, THRESHOLD, EPS)

        # 累加
        total_loss += loss.item() * batch_size
        total_bce += bce.item() * batch_size
        total_dice_loss += dice_loss.item() * batch_size
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size
        total_precision += metrics["precision"] * batch_size
        total_recall += metrics["recall"] * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "bce": total_bce / max(total_count, 1),
        "dice_loss": total_dice_loss / max(total_count, 1),
        "dice": total_dice / max(total_count, 1),
        "iou": total_iou / max(total_count, 1),
        "precision": total_precision / max(total_count, 1),
        "recall": total_recall / max(total_count, 1),
    }


# ============================================================
# 13) Save predictions / per-image eval
# ============================================================

@torch.no_grad()
def save_test_predictions(model, loader, save_dir, threshold=0.5):
    # eval mode
    model.eval()

    # output dirs
    image_dir = os.path.join(save_dir, "image")
    gt_dir = os.path.join(save_dir, "mask_gt")
    pred_dir = os.path.join(save_dir, "mask_pred")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    for batch in loader:
        # 取資料
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        stems = batch["stem"]
        image_paths = batch["image_path"]

        # forward
        with autocast(enabled=AMP_ENABLED):
            logits = model(images)

        # logits -> pred
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # cpu numpy
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy()

        for i in range(images.size(0)):
            stem = stems[i]

            # 儲存 resize 後原圖
            raw_img = Image.open(image_paths[i]).convert("RGB")
            raw_img = raw_img.resize(IMAGE_SIZE, Image.BILINEAR)
            raw_img.save(os.path.join(image_dir, f"{stem}.png"))

            # 儲存 GT mask
            gt_mask = (masks_np[i, 0] * 255).astype(np.uint8)
            Image.fromarray(gt_mask).save(os.path.join(gt_dir, f"{stem}.png"))

            # 儲存 pred mask
            pred_mask = (preds_np[i, 0] * 255).astype(np.uint8)
            Image.fromarray(pred_mask).save(os.path.join(pred_dir, f"{stem}.png"))


@torch.no_grad()
def evaluate_test_per_image(model, loader, fold_name, threshold=0.5):
    # eval mode
    model.eval()

    # 儲存每張圖結果
    rows = []

    for batch in loader:
        # 取資料
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        stems = batch["stem"]
        image_paths = batch["image_path"]
        centers = batch["center"]

        # forward
        with autocast(enabled=AMP_ENABLED):
            logits = model(images)

        # logits -> pred
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # cpu numpy
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(images.size(0)):
            # pred / gt
            pred_bin = preds_np[i, 0].astype(np.uint8)
            gt_bin = masks_np[i, 0].astype(np.uint8)

            # metrics
            metrics = compute_binary_metrics(pred_bin, gt_bin)

            rows.append(
                {
                    "Fold": fold_name,
                    "Center": centers[i],
                    "Stem": stems[i],
                    "Image_Path": image_paths[i],
                    "Dice": metrics["Dice"],
                    "IoU": metrics["IoU"],
                    "Acc": metrics["Acc"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "HD95": metrics["HD95"],
                    "ASSD": metrics["ASSD"],
                    "Valid_Boundary_Case": metrics["Valid_Boundary_Case"],
                }
            )

    return rows


def summarize_center_level(df: pd.DataFrame) -> pd.DataFrame:
    # 儲存 center-level rows
    rows = []

    for center_name, g in df.groupby("Center"):
        # HD95 / ASSD 去除 nan
        hd_vals = g["HD95"].dropna().values
        assd_vals = g["ASSD"].dropna().values

        rows.append(
            {
                "Center": center_name,
                "Num_images": len(g),
                "Dice": g["Dice"].mean(),
                "IoU": g["IoU"].mean(),
                "Acc": g["Acc"].mean(),
                "Precision": g["Precision"].mean(),
                "Recall": g["Recall"].mean(),
                "HD95": float(np.mean(hd_vals)) if len(hd_vals) > 0 else np.nan,
                "ASSD": float(np.mean(assd_vals)) if len(assd_vals) > 0 else np.nan,
            }
        )

    out_df = pd.DataFrame(rows).sort_values("Center").reset_index(drop=True)

    return out_df


def aggregate_per_image_metrics(per_image_df: pd.DataFrame) -> Dict[str, float]:
    # 從 per-image 統計 test metrics
    return {
        "dice": float(per_image_df["Dice"].mean()),
        "iou": float(per_image_df["IoU"].mean()),
        "precision": float(per_image_df["Precision"].mean()),
        "recall": float(per_image_df["Recall"].mean()),
        "acc": float(per_image_df["Acc"].mean()),
        "hd95": float(per_image_df["HD95"].dropna().mean()) if per_image_df["HD95"].notna().any() else np.nan,
        "assd": float(per_image_df["ASSD"].dropna().mean()) if per_image_df["ASSD"].notna().any() else np.nan,
    }


# ============================================================
# 14) Parse result.txt and rebuild summary
# ============================================================

def parse_result_txt(result_txt_path):
    # 儲存 result.txt key-value
    data = {}

    with open(result_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

    # 必要欄位
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

    # 檢查欄位
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
    # 儲存所有 fold row
    all_rows = []

    for fold_name in RUN_FOLDS:
        result_txt = os.path.join(
            OUT_ROOT,
            fold_name,
            "stage3E_results",
            "result.txt"
        )

        if os.path.isfile(result_txt):
            row = parse_result_txt(result_txt)
            all_rows.append(row)
        else:
            print(f"[WARN] 找不到 result.txt，略過：{result_txt}")

    # 依 fold name 排序
    all_rows.sort(key=lambda x: x[0])

    # 輸出 all_fold_results.csv
    summary_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
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
        )

        writer.writerows(all_rows)

    return summary_csv, all_rows


def build_fold_val_data_table():
    # all_fold_results 路徑
    all_fold_results_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")

    # 檢查存在
    if not os.path.isfile(all_fold_results_csv):
        raise FileNotFoundError(f"找不到 all_fold_results.csv: {all_fold_results_csv}")

    # 儲存 fold rows
    fold_rows = []

    with open(all_fold_results_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            fold_rows.append(
                [
                    row["fold"],
                    float(row["best_val_dice"]),
                    float(row["delta_shift_dice"]),
                    float(row["best_val_iou"]),
                    float(row["delta_shift_iou"]),
                ]
            )

    # 輸出 fold_val_data_metrics.csv
    fold_csv = os.path.join(OUT_ROOT, "fold_val_data_metrics.csv")

    with open(fold_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["Fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"])
        writer.writerows(fold_rows)

    return fold_rows


def build_global_center_level_table():
    # 收集所有 fold 的 per_image_metrics
    per_image_all = []

    for fold_name in RUN_FOLDS:
        per_image_csv = os.path.join(
            OUT_ROOT,
            fold_name,
            "stage3E_results",
            "per_image_metrics.csv"
        )

        if os.path.isfile(per_image_csv):
            per_image_all.append(pd.read_csv(per_image_csv))

    # 若沒有任何 per-image csv
    if len(per_image_all) == 0:
        print("[WARN] 找不到任何 per_image_metrics.csv，略過全域中心表重建")
        return None

    # 合併所有 fold
    per_image_all_df = pd.concat(per_image_all, axis=0, ignore_index=True)

    # 輸出總 per-image metrics
    per_image_all_df.to_csv(
        os.path.join(OUT_ROOT, "per_image_metrics_all.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 建立 center-level table
    center_level_df = summarize_center_level(per_image_all_df)

    # 輸出 center-level table
    center_level_df.to_csv(
        os.path.join(OUT_ROOT, "center_level_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    return center_level_df


def export_eval_summary():
    # 建立 summary dir
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

    # 要複製的檔案
    files_to_copy = [
        "all_fold_results.csv",
        "fold_val_data_metrics.csv",
        "center_level_metrics.csv",
        "per_image_metrics_all.csv",
        "loco_summary.txt",
    ]

    for name in files_to_copy:
        src = os.path.join(OUT_ROOT, name)
        dst = os.path.join(EVAL_OUT_DIR, name)

        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print("=" * 70)
    print("Stage3E_eval_summary 輸出完成")
    print("=" * 70)
    print("1) per_image_metrics_all.csv")
    print("2) center_level_metrics.csv")
    print("3) fold_val_data_metrics.csv")
    print("4) all_fold_results.csv")
    print("5) loco_summary.txt")
    print(f"EVAL_OUT_DIR = {EVAL_OUT_DIR}")


def rebuild_global_reports():
    # 重建 all fold results
    summary_csv, all_rows = rebuild_all_fold_results_from_disk()

    # 建立 fold val/delta table
    fold_rows = build_fold_val_data_table()

    # 建立 center level table
    center_level_df = build_global_center_level_table()

    # 寫 loco summary
    if center_level_df is not None and len(center_level_df) > 0:
        loco_mean_dice = float(center_level_df["Dice"].mean())
        loco_mean_iou = float(center_level_df["IoU"].mean())

        worst_center_row = center_level_df.loc[center_level_df["Dice"].idxmin()]
        best_center_row = center_level_df.loc[center_level_df["Dice"].idxmax()]

        mean_delta_dice = float(np.mean([row[9] for row in all_rows])) if len(all_rows) > 0 else np.nan
        mean_delta_iou = float(np.mean([row[10] for row in all_rows])) if len(all_rows) > 0 else np.nan

        with open(os.path.join(OUT_ROOT, "loco_summary.txt"), "w", encoding="utf-8") as f:
            f.write("========================================\n")
            f.write("Stage 3-E: SegFormer-B0 + LOCO + KD\n")
            f.write("========================================\n")
            f.write(f"mean_dice: {loco_mean_dice:.6f}\n")
            f.write(f"mean_iou: {loco_mean_iou:.6f}\n")
            f.write(f"worst_center: {worst_center_row['Center']}\n")
            f.write(f"worst_center_dice: {worst_center_row['Dice']:.6f}\n")
            f.write(f"best_center: {best_center_row['Center']}\n")
            f.write(f"best_center_dice: {best_center_row['Dice']:.6f}\n")
            f.write(f"mean_delta_dice: {mean_delta_dice:.6f}\n")
            f.write(f"mean_delta_iou: {mean_delta_iou:.6f}\n")
            f.write("========================================\n")

    # 複製到 eval summary
    export_eval_summary()

    return summary_csv, all_rows, fold_rows, center_level_df


# ============================================================
# 15) Main training loop
# ============================================================

def run_single_fold(fold_name):
    print("=" * 100)
    print("Training", fold_name)
    print("=" * 100)

    # fold dir
    fold_dir = os.path.join(LOCO_ROOT, fold_name)

    # save dir
    save_dir = os.path.join(OUT_ROOT, fold_name)

    # result dir
    result_bundle_dir = os.path.join(save_dir, "stage3E_results")

    # prediction dir
    pred_save_dir = os.path.join(result_bundle_dir, "test_predictions")

    # output files
    result_txt = os.path.join(result_bundle_dir, "result.txt")
    history_csv = os.path.join(result_bundle_dir, "history.csv")
    checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # 建立資料夾
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_bundle_dir, exist_ok=True)

    # 找 train / val / test csv
    train_csv = find_split_csv(fold_dir, "train")
    val_csv = find_split_csv(fold_dir, "val")
    test_csv = find_split_csv(fold_dir, "test")

    # 建立 dataset
    train_dataset = PolypDataset(train_csv, IMAGE_SIZE, is_train=True)
    val_dataset = PolypDataset(val_csv, IMAGE_SIZE, is_train=False)
    test_dataset = PolypDataset(test_csv, IMAGE_SIZE, is_train=False)

    print(f"train={len(train_dataset)} | val={len(val_dataset)} | test={len(test_dataset)}")

    # train loader
    # drop_last=True 避免最後一個 batch 太小造成訓練不穩
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if DEVICE == "cuda" else False,
        drop_last=True
    )

    # val loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if DEVICE == "cuda" else False,
        drop_last=False
    )

    # test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if DEVICE == "cuda" else False,
        drop_last=False
    )

    # 建立 student
    student, student_impl = build_student_model()
    student = student.to(DEVICE)

    # 載入該 fold 的 teacher
    teacher = load_teacher_for_fold(fold_name)

    # loss
    criterion = DiceBCELossWithParts(
        smooth=1.0,
        bce_weight=1.0,
        dice_weight=1.0
    )

    # optimizer
    optimizer = optim.AdamW(
        student.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR
    )

    # AMP scaler
    scaler = GradScaler(enabled=AMP_ENABLED)

    # train state
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = -1
    history = []
    start_epoch = 1
    epochs_no_improve = 0

    # resume
    if ENABLE_RESUME and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)

        student.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_val_dice = ckpt["best_val_dice"]
        best_val_iou = ckpt["best_val_iou"]
        best_epoch = ckpt["best_epoch"]
        history = ckpt["history"]
        epochs_no_improve = ckpt["epochs_no_improve"]

        print(f"[{fold_name}] resume from epoch {start_epoch}")

    else:
        print(f"[{fold_name}] start from scratch")

    print(f"[{fold_name}] student_impl = {student_impl}")

    # 開始計時
    start_time = time.time()

    # training loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # train one epoch with KD
        train_metrics = run_train_epoch_kd(
            model=student,
            teacher=teacher,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch
        )

        # validation
        val_metrics = run_eval_epoch(
            model=student,
            loader=val_loader,
            criterion=criterion
        )

        # current lr
        current_lr = optimizer.param_groups[0]["lr"]

        # 儲存 history row
        history.append(
            [
                epoch,
                current_lr,
                train_metrics["loss"],
                train_metrics["sup_loss"],
                train_metrics["kd_loss"],
                train_metrics["kd_weight"],
                train_metrics["bce"],
                train_metrics["dice_loss"],
                train_metrics["dice"],
                train_metrics["iou"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["skipped_batches"],
                val_metrics["loss"],
                val_metrics["bce"],
                val_metrics["dice_loss"],
                val_metrics["dice"],
                val_metrics["iou"],
                val_metrics["precision"],
                val_metrics["recall"],
            ]
        )

        # scheduler step
        scheduler.step(val_metrics["dice"])

        # update best
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_no_improve = 0

            # best model 只存 student state_dict
            torch.save(student.state_dict(), best_model_path)

        else:
            epochs_no_improve += 1

        # 儲存 checkpoint_latest
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_dice=best_val_dice,
            best_val_iou=best_val_iou,
            best_epoch=best_epoch,
            history=history,
            epochs_no_improve=epochs_no_improve,
            student_impl=student_impl,
        )

        # 印 log
        print(
            f"[{fold_name}] Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_sup={train_metrics['sup_loss']:.4f} "
            f"train_kd={train_metrics['kd_loss']:.4f} "
            f"kd_w={train_metrics['kd_weight']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"train_precision={train_metrics['precision']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} "
            f"skipped_batches={train_metrics['skipped_batches']} | "
            f"valid_loss={val_metrics['loss']:.4f} "
            f"valid_dice={val_metrics['dice']:.4f} "
            f"valid_iou={val_metrics['iou']:.4f} "
            f"valid_precision={val_metrics['precision']:.4f} "
            f"valid_recall={val_metrics['recall']:.4f} | "
            f"best_valid_dice={best_val_dice:.4f} @ epoch {best_epoch}"
        )

        # early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"[{fold_name}] Early stopping triggered.")
            break

    # 檢查 best model
    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"[{fold_name}] 找不到 best model: {best_model_path}")

    # 載入 best student model
    student.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # batch-level test
    test_metrics_batch = run_eval_epoch(
        model=student,
        loader=test_loader,
        criterion=criterion
    )

    # save predictions
    if SAVE_TEST_PRED:
        save_test_predictions(
            model=student,
            loader=test_loader,
            save_dir=pred_save_dir,
            threshold=THRESHOLD
        )

    # per-image evaluation
    per_image_rows = evaluate_test_per_image(
        model=student,
        loader=test_loader,
        fold_name=fold_name,
        threshold=THRESHOLD
    )

    per_image_df = pd.DataFrame(per_image_rows)

    per_image_csv_path = os.path.join(result_bundle_dir, "per_image_metrics.csv")

    per_image_df.to_csv(
        per_image_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    # per-fold center table
    center_df = summarize_center_level(per_image_df)

    if SAVE_PER_FOLD_CENTER_TABLE:
        center_df.to_csv(
            os.path.join(result_bundle_dir, "center_level_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # final test metrics from per-image
    test_metrics_final = aggregate_per_image_metrics(per_image_df)

    # elapsed time
    elapsed = time.time() - start_time

    # delta shift
    delta_shift_dice = best_val_dice - test_metrics_final["dice"]
    delta_shift_iou = best_val_iou - test_metrics_final["iou"]

    # batch vs per-image check
    compare_txt = os.path.join(result_bundle_dir, "batch_vs_per_image_check.txt")

    with open(compare_txt, "w", encoding="utf-8") as f:
        f.write(f"fold: {fold_name}\n")
        f.write(f"batch_test_dice: {test_metrics_batch['dice']:.6f}\n")
        f.write(f"batch_test_iou: {test_metrics_batch['iou']:.6f}\n")
        f.write(f"batch_test_precision: {test_metrics_batch['precision']:.6f}\n")
        f.write(f"batch_test_recall: {test_metrics_batch['recall']:.6f}\n")
        f.write(f"per_image_test_dice: {test_metrics_final['dice']:.6f}\n")
        f.write(f"per_image_test_iou: {test_metrics_final['iou']:.6f}\n")
        f.write(f"per_image_test_precision: {test_metrics_final['precision']:.6f}\n")
        f.write(f"per_image_test_recall: {test_metrics_final['recall']:.6f}\n")
        f.write(f"per_image_test_acc: {test_metrics_final['acc']:.6f}\n")
        f.write(f"per_image_test_hd95: {test_metrics_final['hd95']:.6f}\n")
        f.write(f"per_image_test_assd: {test_metrics_final['assd']:.6f}\n")

    # history.csv
    with open(history_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_sup_loss",
                "train_kd_loss",
                "train_kd_weight",
                "train_bce",
                "train_dice_loss",
                "train_dice",
                "train_iou",
                "train_precision",
                "train_recall",
                "skipped_batches",
                "valid_loss",
                "valid_bce",
                "valid_dice_loss",
                "valid_dice",
                "valid_iou",
                "valid_precision",
                "valid_recall",
            ]
        )

        writer.writerows(history)

    # result.txt
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"model_name: {MODEL_NAME}\n")
        f.write("stage: Stage 3-E (SegFormer-B0 + LOCO + KD, consistent metrics)\n")
        f.write(f"student_impl: {student_impl}\n")
        f.write(f"student_pretrained: {STUDENT_PRETRAINED_NAME}\n")
        f.write(f"teacher_root: {TEACHER_ROOT}\n")
        f.write(f"teacher_pretrained: {TEACHER_PRETRAINED_NAME}\n")
        f.write(f"kd_temperature: {KD_CFG['temperature']}\n")
        f.write(f"kd_base_weight: {KD_CFG['base_weight']}\n")
        f.write(f"kd_warmup_no_kd_epochs: {KD_CFG['warmup_no_kd_epochs']}\n")
        f.write(f"kd_rampup_epochs: {KD_CFG['rampup_epochs']}\n")
        f.write(f"fold: {fold_name}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_dice: {best_val_dice:.6f}\n")
        f.write(f"best_val_iou: {best_val_iou:.6f}\n")
        f.write(f"test_loss: {test_metrics_batch['loss']:.6f}\n")
        f.write(f"test_dice: {test_metrics_final['dice']:.6f}\n")
        f.write(f"test_iou: {test_metrics_final['iou']:.6f}\n")
        f.write(f"test_precision: {test_metrics_final['precision']:.6f}\n")
        f.write(f"test_recall: {test_metrics_final['recall']:.6f}\n")
        f.write(f"delta_shift_dice: {delta_shift_dice:.6f}\n")
        f.write(f"delta_shift_iou: {delta_shift_iou:.6f}\n")
        f.write(f"time_sec: {elapsed:.2f}\n")
        f.write(f"prediction_dir: {pred_save_dir}\n")
        f.write(f"per_image_csv: {per_image_csv_path}\n")
        f.write(f"batch_vs_per_image_check: {compare_txt}\n")

    # print final fold result
    print("-" * 80)
    print(f"[{fold_name}] TEST RESULT")
    print(f"student_impl      = {student_impl}")
    print(f"best_epoch        = {best_epoch}")
    print(f"best_val_dice     = {best_val_dice:.4f}")
    print(f"best_val_iou      = {best_val_iou:.4f}")
    print(f"test_loss(batch)  = {test_metrics_batch['loss']:.4f}")
    print(f"test_dice(final)  = {test_metrics_final['dice']:.4f}")
    print(f"test_iou(final)   = {test_metrics_final['iou']:.4f}")
    print(f"test_precision    = {test_metrics_final['precision']:.4f}")
    print(f"test_recall       = {test_metrics_final['recall']:.4f}")
    print(f"test_acc          = {test_metrics_final['acc']:.4f}")
    print(f"test_hd95         = {test_metrics_final['hd95']:.4f}")
    print(f"test_assd         = {test_metrics_final['assd']:.4f}")
    print(f"delta_shift_dice  = {delta_shift_dice:.4f}")
    print(f"delta_shift_iou   = {delta_shift_iou:.4f}")
    print(f"result_bundle_dir = {result_bundle_dir}")
    print(f"prediction_dir    = {pred_save_dir}")
    print("-" * 80)
    print()

    # 清理 GPU 記憶體
    del student
    del teacher
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del val_loader
    del test_loader
    del train_dataset
    del val_dataset
    del test_dataset

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 16) Main
# ============================================================

if __name__ == "__main__":
    # 印出設定
    print_runtime_info()

    # 固定 seed
    set_seed(SEED)

    try:
        # 逐 fold 跑
        for fold_name in RUN_FOLDS:
            run_single_fold(fold_name)

        # 重建全域報告
        summary_csv, all_rows, fold_rows, center_level_df = rebuild_global_reports()

        print("=" * 100)
        print("Current run finished")
        print("=" * 100)
        print(f"summary csv: {summary_csv}")

        if len(all_rows) > 0:
            for row in all_rows:
                print(
                    f"{row[0]} | "
                    f"best_epoch={row[1]} | "
                    f"best_val_dice={row[2]:.4f} | "
                    f"test_dice={row[5]:.4f} | "
                    f"test_iou={row[6]:.4f} | "
                    f"test_precision={row[7]:.4f} | "
                    f"test_recall={row[8]:.4f} | "
                    f"delta_shift_dice={row[9]:.4f}"
                )
        else:
            print("沒有任何已完成 fold 可供彙總。")

        if center_level_df is not None and not center_level_df.empty:
            print("\n[Center-level Metrics]")
            print(center_level_df.to_string(index=False))

        print("=" * 100)
        print("[FINAL OUTPUT FILES]")
        print(os.path.join(OUT_ROOT, "all_fold_results.csv"))
        print(os.path.join(OUT_ROOT, "fold_val_data_metrics.csv"))
        print(os.path.join(OUT_ROOT, "center_level_metrics.csv"))
        print(os.path.join(OUT_ROOT, "per_image_metrics_all.csv"))
        print(os.path.join(OUT_ROOT, "loco_summary.txt"))
        print("=" * 100)

    except Exception as e:
        print("\n[ERROR] Training crashed.")
        print("reason:", str(e))
        traceback.print_exc()
