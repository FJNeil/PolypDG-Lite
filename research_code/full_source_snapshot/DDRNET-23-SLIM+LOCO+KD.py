import os
import csv
import time
import math
import random
import shutil
import traceback
import warnings
from typing import List, Dict, Tuple

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

# =========================================================
# Teacher: SegFormer
# =========================================================
try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError as e:
    raise ImportError("請先安裝 transformers：pip install transformers") from e


# =========================================================
# 0) Main Config
# =========================================================
LOCO_ROOT = r"./data\loco_1537_clean"
TEACHER_ROOT = r"./data\stage2C_segformer_b2_consistency_improved_selfcons_official"

# 每次自動開新資料夾，避免混檔
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_ROOT = rf"./data\stage3C_ddrnet23slim_loco_kd_stage3Afair_fix_{RUN_STAMP}"
EVAL_OUT_DIR = os.path.join(OUT_ROOT, "Stage3C_eval_summary")

RUN_FOLDS = [
    "fold_test_C1",
    "fold_test_C2",
    "fold_test_C3",
    "fold_test_C4",
    "fold_test_C5",
    "fold_test_C6",
]

# 若你只想先驗證 C4，改成：
# RUN_FOLDS = ["fold_test_C4"]

IMAGE_SIZE = (352, 352)
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 12
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

NUM_WORKERS = 0
PIN_MEMORY = True

THRESHOLD = 0.5
EPS = 1e-7

USE_AMP = True
SEED = 42
GRAD_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = USE_AMP and DEVICE == "cuda"

SAVE_TEST_PRED = True
SAVE_PER_FOLD_CENTER_TABLE = True

MODEL_NAME = "DDRNet-23-slim + LOCO + KD"
STUDENT_IMPL = "custom_ddrnet23slim"

TEACHER_CKPT_NAME = "best_model.pth"
TEACHER_PRETRAINED_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

KD_CFG = {
    "enable": True,
    "temperature": 4.0,
    "base_weight": 0.05,
    "warmup_no_kd_epochs": 10,
    "rampup_epochs": 20,
    "teacher_eval_mode": True,
}

# 不做 resume，避免混版
ENABLE_RESUME = False

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)


# =========================================================
# 1) Seed
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
# 2) Runtime info
# =========================================================
def print_runtime_info():
    print("========================================")
    print("DEVICE =", DEVICE)
    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))
        print("torch.version.cuda =", torch.version.cuda)
    print("MODEL_NAME =", MODEL_NAME)
    print("STUDENT_IMPL =", STUDENT_IMPL)
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
    print("========================================")


# =========================================================
# 3) CSV / fold helpers
# =========================================================
def find_split_csv(fold_dir: str, split_name: str) -> str:
    candidates = [
        os.path.join(fold_dir, f"{split_name}.csv"),
        os.path.join(fold_dir, f"{split_name}_data.csv"),
        os.path.join(fold_dir, f"{split_name}_split.csv"),
        os.path.join(fold_dir, "csv", f"{split_name}.csv"),
        os.path.join(fold_dir, "splits", f"{split_name}.csv"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    for root, _, files in os.walk(fold_dir):
        for file_name in files:
            lower = file_name.lower()
            if lower.endswith(".csv") and split_name in lower:
                return os.path.join(root, file_name)

    raise FileNotFoundError(f"找不到 {split_name}.csv，fold_dir={fold_dir}")


# =========================================================
# 4) Preprocessing / augmentation
#    對齊你原本 Stage3A
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


def image_to_chw_float(img):
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    return img_np


def mask_to_1hw_float(mask):
    mask_np = np.array(mask, dtype=np.uint8)
    mask_np = (mask_np > 127).astype(np.float32)
    mask_np = np.expand_dims(mask_np, axis=0)
    return mask_np


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


# =========================================================
# 5) Dataset
# =========================================================
class PolypDataset(Dataset):
    def __init__(self, csv_file, image_size=(352, 352), is_train=False):
        self.rows = []
        self.image_size = image_size
        self.is_train = is_train

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

        img_np, mask_np = preprocess_single_view(
            img=img,
            mask=mask,
            image_size=self.image_size,
            is_train=self.is_train
        )

        return {
            "image": torch.tensor(img_np, dtype=torch.float32),
            "mask": torch.tensor(mask_np, dtype=torch.float32),
            "stem": row.get("stem", os.path.splitext(os.path.basename(row["image_path"]))[0]),
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "center": row["center"] if "center" in row else "UNKNOWN",
        }


# =========================================================
# 6) Teacher model
# =========================================================
class SegFormerBinary(nn.Module):
    def __init__(self, use_pretrained=True, pretrained_name=TEACHER_PRETRAINED_NAME):
        super().__init__()
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
        input_h, input_w = x.shape[-2], x.shape[-1]
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(logits, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return logits


def normalize_state_dict_keys(state_dict: Dict) -> Dict:
    new_state = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    return new_state


def extract_state_dict(ckpt: Dict) -> Dict:
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "student_state_dict" in ckpt:
            return ckpt["student_state_dict"]
    return ckpt


def load_teacher_for_fold(fold_name: str) -> nn.Module:
    ckpt_path = os.path.join(TEACHER_ROOT, fold_name, TEACHER_CKPT_NAME)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到 teacher checkpoint：{ckpt_path}")

    teacher = SegFormerBinary(use_pretrained=True, pretrained_name=TEACHER_PRETRAINED_NAME).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = normalize_state_dict_keys(extract_state_dict(ckpt))

    try:
        teacher.load_state_dict(state_dict, strict=True)
        print(f"[Teacher] strict load success for {fold_name}")
    except RuntimeError as e:
        print(f"[Teacher][WARN] strict load failed for {fold_name}, fallback strict=False")
        print("reason:", e)
        load_result = teacher.load_state_dict(state_dict, strict=False)
        print("[Teacher][WARN] missing_keys:", load_result.missing_keys[:20])
        print("[Teacher][WARN] unexpected_keys:", load_result.unexpected_keys[:20])

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[Teacher] fold={fold_name} loaded from: {ckpt_path}")
    return teacher


# =========================================================
# 7) Student model: custom DDRNet-23-slim
# =========================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, no_relu=False):
        super().__init__()
        self.no_relu = no_relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = out + identity
        if self.no_relu:
            return out
        return self.relu(out)


def make_layer(in_channels, out_channels, num_blocks, stride=1, last_no_relu=False):
    layers = []
    for i in range(num_blocks):
        cur_stride = stride if i == 0 else 1
        cur_in_channels = in_channels if i == 0 else out_channels
        no_relu = last_no_relu and (i == num_blocks - 1)
        layers.append(BasicBlock(cur_in_channels, out_channels, cur_stride, no_relu))
    return nn.Sequential(*layers)


class DAPPM(nn.Module):
    def __init__(self, in_channels, branch_channels, out_channels):
        super().__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False)
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(5, 2, 2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(9, 4, 4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(17, 8, 8),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False)
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False)
        )

        self.process1 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)
        self.process2 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)
        self.process3 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)
        self.process4 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels * 5, out_channels, 1, bias=False)
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        x0 = self.scale0(x)
        x1 = self.process1(F.interpolate(self.scale1(x), size=(h, w), mode="bilinear", align_corners=False) + x0)
        x2 = self.process2(F.interpolate(self.scale2(x), size=(h, w), mode="bilinear", align_corners=False) + x1)
        x3 = self.process3(F.interpolate(self.scale3(x), size=(h, w), mode="bilinear", align_corners=False) + x2)
        x4 = self.process4(F.interpolate(self.scale4(x), size=(h, w), mode="bilinear", align_corners=False) + x3)
        out = torch.cat([x0, x1, x2, x3, x4], dim=1)
        out = self.compression(out) + self.shortcut(x)
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class CustomDDRNet23SlimBackbone(nn.Module):
    def __init__(self, planes=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvBNReLU(3, planes, 3, 2, 1),
            ConvBNReLU(planes, planes, 3, 2, 1)
        )
        self.layer1 = make_layer(planes, planes, 2, 1)
        self.layer2 = make_layer(planes, planes * 2, 2, 2)
        self.layer3 = make_layer(planes * 2, planes * 4, 2, 2)
        self.layer4 = make_layer(planes * 4, planes * 8, 2, 2)
        self.layer5 = make_layer(planes * 8, planes * 8, 1, 1, True)

        self.layer3_ = make_layer(planes * 2, planes * 2, 2, 1)
        self.layer4_ = make_layer(planes * 2, planes * 2, 2, 1)
        self.layer5_ = make_layer(planes * 2, planes * 2, 1, 1, True)

        self.compression3 = nn.Sequential(
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 2, 1, bias=False)
        )
        self.compression4 = nn.Sequential(
            nn.BatchNorm2d(planes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 8, planes * 2, 1, bias=False)
        )

        self.down3 = nn.Sequential(
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 2, planes * 4, 3, 2, 1, bias=False)
        )
        self.down4 = nn.Sequential(
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 2, planes * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, 3, 2, 1, bias=False),
        )

        self.dappm = DAPPM(planes * 8, planes, planes * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x_high = self.layer3_(x)
        x_low = self.layer3(x)

        x_high = x_high + F.interpolate(self.compression3(x_low), size=x_high.shape[-2:], mode="bilinear", align_corners=False)
        x_low = x_low + self.down3(x_high)

        x_high = self.layer4_(F.relu(x_high))
        x_low = self.layer4(F.relu(x_low))

        x_high = x_high + F.interpolate(self.compression4(x_low), size=x_high.shape[-2:], mode="bilinear", align_corners=False)
        x_low = x_low + self.down4(x_high)

        aux_feature = F.relu(x_high)
        x_high = self.layer5_(aux_feature)
        x_low = self.layer5(F.relu(x_low))

        x_low_context = self.dappm(F.relu(x_low))
        x_low_up = F.interpolate(x_low_context, size=x_high.shape[-2:], mode="bilinear", align_corners=False)

        fused_feature = torch.cat([x_high, x_low_up], dim=1)
        return fused_feature, aux_feature


class CustomDDRNet23SlimBinarySeg(nn.Module):
    def __init__(self, num_classes=1, planes=32, use_aux=True):
        super().__init__()
        self.use_aux = use_aux
        self.backbone = CustomDDRNet23SlimBackbone(planes=planes)
        self.seg_head = SegHead(planes * 4, planes * 2, num_classes)
        self.aux_head = SegHead(planes * 2, planes, num_classes)

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        fused_feature, aux_feature = self.backbone(x)

        main_logits = self.seg_head(fused_feature)
        main_logits = F.interpolate(main_logits, size=(input_h, input_w), mode="bilinear", align_corners=False)

        if self.training and self.use_aux:
            aux_logits = self.aux_head(aux_feature)
            aux_logits = F.interpolate(aux_logits, size=(input_h, input_w), mode="bilinear", align_corners=False)
            return main_logits, aux_logits

        return main_logits


def build_student_model():
    model = CustomDDRNet23SlimBinarySeg(num_classes=1, planes=32, use_aux=True)
    return model, STUDENT_IMPL


# =========================================================
# 8) Losses
# =========================================================
class DiceBCELossWithParts(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

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

        total = self.bce_weight * bce + self.dice_weight * dice_loss
        return total, bce.detach(), dice_loss.detach()


# =========================================================
# 9) KD helpers
# =========================================================
def binary_prob_to_two_class(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    bg = (1.0 - prob).clamp(eps, 1.0 - eps)
    fg = prob
    return torch.cat([bg, fg], dim=1)


def kd_kl_from_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    s_prob = torch.sigmoid(student_logits / temperature)
    t_prob = torch.sigmoid(teacher_logits / temperature)

    s_dist = binary_prob_to_two_class(s_prob, eps)
    t_dist = binary_prob_to_two_class(t_prob, eps)

    s_log = torch.log(s_dist.clamp_min(eps))
    kl_map = F.kl_div(s_log, t_dist, reduction="none")
    kl_map = kl_map.sum(dim=1)
    kd = kl_map.mean()
    kd = kd * (temperature ** 2)
    kd = torch.nan_to_num(kd, nan=0.0, posinf=1.0, neginf=0.0)
    return kd


def get_current_kd_weight(epoch: int, base_weight: float, warmup_epochs: int, rampup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if rampup_epochs <= 0:
        return base_weight

    effective_epoch = epoch - warmup_epochs
    factor = min(1.0, effective_epoch / float(rampup_epochs))
    factor = 0.5 - 0.5 * math.cos(math.pi * factor)
    return base_weight * factor


# =========================================================
# 10) Metrics
#    關鍵修正：
#    both-empty case 在 batch-level 和 per-image-level 都統一定義為 perfect match
# =========================================================
def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    both_empty = (pred_sum == 0) & (target_sum == 0)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

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
    return float(n) / float(d) if d > 0 else 0.0


def binary_surface(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask.astype(bool) ^ eroded.astype(bool)


def hd95_and_assd(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, bool]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0, True
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan, False

    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)

    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0, 0.0, True
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return np.nan, np.nan, False

    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    dist_pred_to_gt = dt_gt[pred_surface]
    dist_gt_to_pred = dt_pred[gt_surface]

    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan, np.nan, False

    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)
    hd95 = np.percentile(all_dists, 95)
    assd = np.mean(all_dists)
    return float(hd95), float(assd), True


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    # 關鍵修正：both-empty 統一定義為 perfect match
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

    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

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


# =========================================================
# 11) Checkpoint utils
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
    epochs_no_improve,
    student_impl,
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
        "student_impl": student_impl,
    }, checkpoint_path)


# =========================================================
# 12) Train / Eval
# =========================================================
def run_train_epoch_kd(model, teacher, loader, criterion, optimizer, scaler, epoch):
    model.train()
    if KD_CFG["teacher_eval_mode"]:
        teacher.eval()

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

    current_kd_weight = get_current_kd_weight(
        epoch=epoch,
        base_weight=KD_CFG["base_weight"],
        warmup_epochs=KD_CFG["warmup_no_kd_epochs"],
        rampup_epochs=KD_CFG["rampup_epochs"],
    )

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=AMP_ENABLED):
            outputs = model(images)

            if isinstance(outputs, tuple):
                main_logits, aux_logits = outputs
                main_sup_loss, main_bce, main_dice_loss = criterion(main_logits, masks)
                aux_sup_loss, _, _ = criterion(aux_logits, masks)
                sup_loss = main_sup_loss + 0.4 * aux_sup_loss
                metric_logits = main_logits
                batch_bce = main_bce
                batch_dice_loss = main_dice_loss
            else:
                main_logits = outputs
                sup_loss, batch_bce, batch_dice_loss = criterion(main_logits, masks)
                metric_logits = main_logits

            kd_loss = torch.tensor(0.0, device=DEVICE, dtype=sup_loss.dtype)
            if KD_CFG["enable"] and current_kd_weight > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                kd_loss = kd_kl_from_logits(
                    student_logits=main_logits,
                    teacher_logits=teacher_logits,
                    temperature=KD_CFG["temperature"],
                )

            loss = sup_loss + current_kd_weight * kd_loss

        if not torch.isfinite(loss):
            skipped_batches += 1
            print("[WARN] 遇到非有限 loss，跳過此 batch")
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

        batch_size = images.size(0)
        metrics = batch_metrics_from_logits(metric_logits, masks, THRESHOLD, EPS)

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
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_count = 0

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        with autocast(enabled=AMP_ENABLED):
            logits = model(images)
            loss, bce, dice_loss = criterion(logits, masks)

        batch_size = images.size(0)
        metrics = batch_metrics_from_logits(logits, masks, THRESHOLD, EPS)

        total_loss += loss.item() * batch_size
        total_bce += bce.item() * batch_size
        total_dice_loss += dice_loss.item() * batch_size
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size
        total_precision += metrics["precision"] * batch_size
        total_recall += metrics["recall"] * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "bce": total_bce / total_count,
        "dice_loss": total_dice_loss / total_count,
        "dice": total_dice / total_count,
        "iou": total_iou / total_count,
        "precision": total_precision / total_count,
        "recall": total_recall / total_count,
    }


# =========================================================
# 13) Save predictions / per-image eval
# =========================================================
@torch.no_grad()
def save_test_predictions(model, loader, save_dir, threshold=0.5):
    model.eval()

    image_dir = os.path.join(save_dir, "image")
    gt_dir = os.path.join(save_dir, "mask_gt")
    pred_dir = os.path.join(save_dir, "mask_pred")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

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


@torch.no_grad()
def evaluate_test_per_image(model, loader, fold_name, threshold=0.5):
    model.eval()
    rows = []

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        stems = batch["stem"]
        image_paths = batch["image_path"]
        centers = batch["center"]

        with autocast(enabled=AMP_ENABLED):
            logits = model(images)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i in range(images.size(0)):
            pred_bin = preds_np[i, 0].astype(np.uint8)
            gt_bin = masks_np[i, 0].astype(np.uint8)
            metrics = compute_binary_metrics(pred_bin, gt_bin)

            rows.append({
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
            })

    return rows


def summarize_center_level(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for center_name, g in df.groupby("Center"):
        hd_vals = g["HD95"].dropna().values
        assd_vals = g["ASSD"].dropna().values

        rows.append({
            "Center": center_name,
            "Num_images": len(g),
            "Dice": g["Dice"].mean(),
            "IoU": g["IoU"].mean(),
            "Acc": g["Acc"].mean(),
            "Precision": g["Precision"].mean(),
            "Recall": g["Recall"].mean(),
            "HD95": float(np.mean(hd_vals)) if len(hd_vals) > 0 else np.nan,
            "ASSD": float(np.mean(assd_vals)) if len(assd_vals) > 0 else np.nan,
        })

    out_df = pd.DataFrame(rows).sort_values("Center").reset_index(drop=True)
    return out_df


def aggregate_per_image_metrics(per_image_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "dice": float(per_image_df["Dice"].mean()),
        "iou": float(per_image_df["IoU"].mean()),
        "precision": float(per_image_df["Precision"].mean()),
        "recall": float(per_image_df["Recall"].mean()),
        "acc": float(per_image_df["Acc"].mean()),
        "hd95": float(per_image_df["HD95"].dropna().mean()) if per_image_df["HD95"].notna().any() else np.nan,
        "assd": float(per_image_df["ASSD"].dropna().mean()) if per_image_df["ASSD"].notna().any() else np.nan,
    }


# =========================================================
# 14) Parse result.txt and rebuild summary
# =========================================================
def parse_result_txt(result_txt_path):
    data = {}

    with open(result_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

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
        result_txt = os.path.join(OUT_ROOT, fold_name, "stage3C_results", "result.txt")
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

    return summary_csv, all_rows


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

    fold_csv = os.path.join(OUT_ROOT, "fold_val_data_metrics.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"])
        writer.writerows(fold_rows)

    return fold_rows


def build_global_center_level_table():
    per_image_all = []

    for fold_name in RUN_FOLDS:
        per_image_csv = os.path.join(OUT_ROOT, fold_name, "stage3C_results", "per_image_metrics.csv")
        if os.path.isfile(per_image_csv):
            per_image_all.append(pd.read_csv(per_image_csv))

    if len(per_image_all) == 0:
        print("[WARN] 找不到任何 per_image_metrics.csv，略過全域中心表重建")
        return None

    per_image_all_df = pd.concat(per_image_all, axis=0, ignore_index=True)
    per_image_all_df.to_csv(os.path.join(OUT_ROOT, "per_image_metrics_all.csv"), index=False, encoding="utf-8-sig")

    center_level_df = summarize_center_level(per_image_all_df)
    center_level_df.to_csv(os.path.join(OUT_ROOT, "center_level_metrics.csv"), index=False, encoding="utf-8-sig")

    return center_level_df


def export_eval_summary():
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

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
    print("Stage3C_eval_summary 輸出完成")
    print("=" * 70)
    print("1) per_image_metrics_all.csv")
    print("2) center_level_metrics.csv")
    print("3) fold_val_data_metrics.csv")
    print("4) all_fold_results.csv")
    print("5) loco_summary.txt")
    print(f"EVAL_OUT_DIR = {EVAL_OUT_DIR}")


def rebuild_global_reports():
    summary_csv, all_rows = rebuild_all_fold_results_from_disk()
    fold_rows = build_fold_val_data_table()
    center_level_df = build_global_center_level_table()

    if center_level_df is not None and len(center_level_df) > 0:
        loco_mean_dice = float(center_level_df["Dice"].mean())
        loco_mean_iou = float(center_level_df["IoU"].mean())
        worst_center_row = center_level_df.loc[center_level_df["Dice"].idxmin()]
        best_center_row = center_level_df.loc[center_level_df["Dice"].idxmax()]
        mean_delta_dice = float(np.mean([row[9] for row in all_rows])) if len(all_rows) > 0 else np.nan
        mean_delta_iou = float(np.mean([row[10] for row in all_rows])) if len(all_rows) > 0 else np.nan

        with open(os.path.join(OUT_ROOT, "loco_summary.txt"), "w", encoding="utf-8") as f:
            f.write("========================================\n")
            f.write("LOCO Summary\n")
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

    export_eval_summary()
    return summary_csv, all_rows, fold_rows, center_level_df


# =========================================================
# 15) Main training loop
# =========================================================
def run_single_fold(fold_name):
    print("========================================")
    print("Training", fold_name)
    print("========================================")

    fold_dir = os.path.join(LOCO_ROOT, fold_name)
    save_dir = os.path.join(OUT_ROOT, fold_name)

    result_bundle_dir = os.path.join(save_dir, "stage3C_results")
    pred_save_dir = os.path.join(result_bundle_dir, "test_predictions")

    result_txt = os.path.join(result_bundle_dir, "result.txt")
    history_csv = os.path.join(result_bundle_dir, "history.csv")
    checkpoint_path = os.path.join(save_dir, "checkpoint_latest.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_bundle_dir, exist_ok=True)

    train_csv = find_split_csv(fold_dir, "train")
    val_csv = find_split_csv(fold_dir, "val")
    test_csv = find_split_csv(fold_dir, "test")

    train_dataset = PolypDataset(train_csv, IMAGE_SIZE, is_train=True)
    val_dataset = PolypDataset(val_csv, IMAGE_SIZE, is_train=False)
    test_dataset = PolypDataset(test_csv, IMAGE_SIZE, is_train=False)

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

    student, student_impl = build_student_model()
    student = student.to(DEVICE)

    teacher = load_teacher_for_fold(fold_name)

    criterion = DiceBCELossWithParts(smooth=1.0, bce_weight=1.0, dice_weight=1.0)

    optimizer = optim.AdamW(
        student.parameters(),
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

    start_time = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_metrics = run_train_epoch_kd(
            model=student,
            teacher=teacher,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch
        )

        val_metrics = run_eval_epoch(
            model=student,
            loader=val_loader,
            criterion=criterion
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history.append([
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
            val_metrics["recall"]
        ])

        scheduler.step(val_metrics["dice"])

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(student.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

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
            student_impl=student_impl
        )

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

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"[{fold_name}] Early stopping triggered.")
            break

    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(f"[{fold_name}] 找不到 best model: {best_model_path}")

    student.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # batch-level test metrics
    test_metrics_batch = run_eval_epoch(
        model=student,
        loader=test_loader,
        criterion=criterion
    )

    if SAVE_TEST_PRED:
        save_test_predictions(student, test_loader, pred_save_dir, THRESHOLD)

    # per-image metrics
    per_image_rows = evaluate_test_per_image(student, test_loader, fold_name, THRESHOLD)
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_csv_path = os.path.join(result_bundle_dir, "per_image_metrics.csv")
    per_image_df.to_csv(per_image_csv_path, index=False, encoding="utf-8-sig")

    # per-fold center-level metrics
    center_df = summarize_center_level(per_image_df)
    if SAVE_PER_FOLD_CENTER_TABLE:
        center_df.to_csv(os.path.join(result_bundle_dir, "center_level_metrics.csv"), index=False, encoding="utf-8-sig")

    # 關鍵：fold summary 一律採用 per-image 聚合結果，保證和 center-level 一致
    test_metrics_final = aggregate_per_image_metrics(per_image_df)

    elapsed = time.time() - start_time

    delta_shift_dice = best_val_dice - test_metrics_final["dice"]
    delta_shift_iou = best_val_iou - test_metrics_final["iou"]

    # 額外存一份 batch vs per-image 對照，方便你檢查是否還有不一致
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

    with open(history_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "lr",
            "train_loss", "train_sup_loss", "train_kd_loss", "train_kd_weight",
            "train_bce", "train_dice_loss",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "skipped_batches",
            "valid_loss", "valid_bce", "valid_dice_loss",
            "valid_dice", "valid_iou", "valid_precision", "valid_recall"
        ])
        writer.writerows(history)

    # 關鍵：result.txt 也改成寫 per-image 聚合結果
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"model_name: {MODEL_NAME}\n")
        f.write("stage: Stage 3C (DDRNet-23-slim + LOCO + KD, Stage3A-fair settings, consistent metrics)\n")
        f.write(f"student_impl: {student_impl}\n")
        f.write(f"teacher_root: {TEACHER_ROOT}\n")
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

    print("------------------------------------------------------------")
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
    print(f"delta_shift_dice  = {delta_shift_dice:.4f}")
    print(f"delta_shift_iou   = {delta_shift_iou:.4f}")
    print(f"result_bundle_dir = {result_bundle_dir}")
    print(f"prediction_dir    = {pred_save_dir}")
    print("------------------------------------------------------------")
    print()


# =========================================================
# 16) Main
# =========================================================
if __name__ == "__main__":
    print_runtime_info()
    set_seed(SEED)

    try:
        for fold_name in RUN_FOLDS:
            run_single_fold(fold_name)

        summary_csv, all_rows, fold_rows, center_level_df = rebuild_global_reports()

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

    except Exception as e:
        print("\n[ERROR] Training crashed.")
        print("reason:", str(e))
        traceback.print_exc()