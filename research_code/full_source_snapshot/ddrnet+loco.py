# ============================================================
# Stage 3-A (Clean Baseline Version)
# DDRNet-23-slim + Center-level LOCO for Polyp Binary Segmentation
# ------------------------------------------------------------
# 說明：
# 1. 這是 Student Baseline（可當正式 baseline）
# 2. 不含 KD
# 3. 不含 Consistency Learning
# 4. 前處理對齊你前面單視圖 supervised pipeline
# 5. 額外修正：
#    - 強制乾淨重跑，避免舊 checkpoint 汙染
#    - 調整 LR / scheduler / early stopping
#    - 加入 validation threshold search
#    - test / per-image / prediction saving 全部改用最佳 threshold
# 6. 輸出：
#    - all_fold_results.csv
#    - fold_val_data_metrics.csv
#    - center_level_metrics.csv
#    - per_image_metrics.csv
# ============================================================

import os
import csv
import time
import math
import random
import shutil
from pathlib import Path

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


# ============================================================
# 0) Main Config
# ============================================================

# LOCO split 根目錄
LOCO_ROOT = r"./data\loco_1537_clean"

# 輸出根目錄
OUT_ROOT = r"./data\stage3A_ddrnet23slim_loco_baseline_clean"

# 要跑的 folds
RUN_FOLDS = [
    "fold_test_C1",
    "fold_test_C2",
    "fold_test_C3",
    "fold_test_C4",
    "fold_test_C5",
    "fold_test_C6",
]

# 影像尺寸，保持與你 SegFormer 單視圖前處理一致
IMAGE_SIZE = (352, 352)  # (W, H)

# Batch size
BATCH_SIZE = 8

# 訓練 epoch
NUM_EPOCHS = 100

# learning rate
# 原本 1e-4 對 scratch student backbone 偏保守，改成 3e-4
LR = 3e-4

# weight decay
WEIGHT_DECAY = 1e-4

# early stopping
# 原本 12 稍微偏早，改成 18 讓 model 有更完整收斂機會
EARLY_STOPPING_PATIENCE = 18

# ReduceLROnPlateau 設定
# 原本 4 稍快，改成 6
SCHEDULER_PATIENCE = 6
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# segmentation threshold
# 注意：train metric 仍先用 0.5，
# 但 val/test 最終報告改成由 validation search 找最佳 threshold
THRESHOLD = 0.5

# validation threshold search 候選值
VAL_THRESHOLD_CANDIDATES = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]

# 小常數，避免除零
EPS = 1e-7

# dataloader workers
NUM_WORKERS = 0

# pin memory
PIN_MEMORY = True

# AMP
USE_AMP = True

# seed
SEED = 42

# auxiliary head 權重
AUX_LOSS_WEIGHT = 0.4

# DDRNet-23-slim 常用 base channels
PLANES = 32

# 儲存 test 預測圖
SAVE_TEST_PRED = True

# 是否刪除 checkpoint
DELETE_CHECKPOINT_AFTER_FINISH = False

# ============================================================
# Clean-run / Resume control
# ============================================================

# True：強制刪掉舊 fold 輸出，重新跑乾淨 baseline
FORCE_CLEAN_RUN = True

# False：不自動接舊 checkpoint，避免你改參數卻沿用舊 optimizer/scheduler 狀態
AUTO_RESUME = False

# 強制乾淨跑時，把所有 folds 都列進 reset
RESET_FOLDS = set(RUN_FOLDS) if FORCE_CLEAN_RUN else set()

# 強制乾淨跑時，不要因為已有 result.txt 就跳過
SKIP_COMPLETED_FOLDS = False if FORCE_CLEAN_RUN else True

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 只有在 GPU 才啟動 AMP
AMP_ENABLED = USE_AMP and DEVICE == "cuda"

# 建立輸出資料夾
os.makedirs(OUT_ROOT, exist_ok=True)


# ============================================================
# 1) Seed
# ============================================================

def set_seed(seed: int = 42) -> None:
    # 固定 Python random
    random.seed(seed)

    # 固定 NumPy random
    np.random.seed(seed)

    # 固定 PyTorch CPU
    torch.manual_seed(seed)

    # 若有 GPU，也固定 GPU 亂數
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cudnn 設為 deterministic，讓結果更可重現
    torch.backends.cudnn.deterministic = True

    # 關閉 benchmark，避免不同路徑造成不穩定
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Preprocessing
#    與你之前單視圖 supervised 前處理對齊
# ============================================================

def resize_pair(img: Image.Image, mask: Image.Image, image_size=(352, 352)):
    # image 用 bilinear
    img = img.resize(image_size, Image.BILINEAR)

    # mask 用 nearest
    mask = mask.resize(image_size, Image.NEAREST)

    return img, mask


def random_hflip_pair(img: Image.Image, mask: Image.Image, p=0.5):
    # 以機率 p 做水平翻轉
    if random.random() < p:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)

    return img, mask


def random_vflip_pair(img: Image.Image, mask: Image.Image, p=0.5):
    # 以機率 p 做垂直翻轉
    if random.random() < p:
        img = ImageOps.flip(img)
        mask = ImageOps.flip(mask)

    return img, mask


def random_rotate_pair(img: Image.Image, mask: Image.Image, p=0.5, angle_limit=15):
    # 以機率 p 做旋轉
    if random.random() < p:
        # 隨機角度落在 [-angle_limit, angle_limit]
        angle = random.uniform(-angle_limit, angle_limit)

        # image 用 bilinear
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        # mask 用 nearest
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

    return img, mask


def random_scale_shift_pair(
    img: Image.Image,
    mask: Image.Image,
    p=0.5,
    scale_low=0.90,
    scale_high=1.10
):
    # 若未觸發 augmentation，直接回傳
    if random.random() >= p:
        return img, mask

    # 原始寬高
    w, h = img.size

    # 隨機縮放倍率
    scale = random.uniform(scale_low, scale_high)

    # 新尺寸
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # image 用 bilinear resize
    img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

    # mask 用 nearest resize
    mask_scaled = mask.resize((new_w, new_h), Image.NEAREST)

    # 若放大，隨機 crop 回原尺寸
    if scale >= 1.0:
        max_left = max(0, new_w - w)
        max_top = max(0, new_h - h)

        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0

        img_out = img_scaled.crop((left, top, left + w, top + h))
        mask_out = mask_scaled.crop((left, top, left + w, top + h))

        return img_out, mask_out

    # 若縮小，貼回黑色 canvas
    canvas_img = Image.new("RGB", (w, h), (0, 0, 0))
    canvas_mask = Image.new("L", (w, h), 0)

    max_left = max(0, w - new_w)
    max_top = max(0, h - new_h)

    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0

    canvas_img.paste(img_scaled, (left, top))
    canvas_mask.paste(mask_scaled, (left, top))

    return canvas_img, canvas_mask


def random_brightness_contrast(
    img: Image.Image,
    p=0.4,
    brightness_limit=0.15,
    contrast_limit=0.15
):
    # brightness augmentation
    if random.random() < p:
        brightness_factor = random.uniform(1.0 - brightness_limit, 1.0 + brightness_limit)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # contrast augmentation
    if random.random() < p:
        contrast_factor = random.uniform(1.0 - contrast_limit, 1.0 + contrast_limit)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def image_to_chw_float(img: Image.Image):
    # 轉成 float32 並正規化到 [0,1]
    img_np = np.array(img, dtype=np.float32) / 255.0

    # HWC -> CHW
    img_np = np.transpose(img_np, (2, 0, 1))

    return img_np


def mask_to_1hw_float(mask: Image.Image):
    # 先轉成 uint8
    mask_np = np.array(mask, dtype=np.uint8)

    # mask > 127 視為前景
    mask_np = (mask_np > 127).astype(np.float32)

    # HxW -> 1xHxW
    mask_np = np.expand_dims(mask_np, axis=0)

    return mask_np


def preprocess_single_view(
    img: Image.Image,
    mask: Image.Image,
    image_size=(352, 352),
    is_train=False
):
    # 先 resize
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
# 3) Dataset
# ============================================================

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

        # 讀 image，轉 RGB
        img = Image.open(row["image_path"]).convert("RGB")

        # 讀 mask，轉灰階
        mask = Image.open(row["mask_path"]).convert("L")

        # 做前處理
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


# ============================================================
# 4) DDRNet-23-slim Modules
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # 卷積層
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        # BN
        self.bn = nn.BatchNorm2d(out_channels)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Conv -> BN -> ReLU
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, no_relu=False):
        super().__init__()

        # 是否跳過最後一個 ReLU
        self.no_relu = no_relu

        # 第一層 3x3 conv
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二層 3x3 conv
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

        # shortcut 對齊
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 保存 identity
        identity = x

        # 第一層
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二層
        out = self.conv2(out)
        out = self.bn2(out)

        # 若 shortcut 尺寸不同，先做 downsample
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 殘差相加
        out = out + identity

        # 根據設定決定是否加最後 ReLU
        if self.no_relu:
            return out

        return self.relu(out)


def make_layer(in_channels, out_channels, num_blocks, stride=1, last_no_relu=False):
    layers = []

    for i in range(num_blocks):
        cur_stride = stride if i == 0 else 1
        cur_in_channels = in_channels if i == 0 else out_channels
        no_relu = last_no_relu and (i == num_blocks - 1)

        layers.append(
            BasicBlock(
                in_channels=cur_in_channels,
                out_channels=out_channels,
                stride=cur_stride,
                no_relu=no_relu
            )
        )

    return nn.Sequential(*layers)


class DAPPM(nn.Module):
    def __init__(self, in_channels, branch_channels, out_channels):
        super().__init__()

        # 原始尺度分支
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        # 5x5 average pool 分支
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        # 9x9 average pool 分支
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        # 17x17 average pool 分支
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        # 全域池化分支
        # 這裡不放 BatchNorm2d，避免 batch=1 時 [1, C, 1, 1] BN crash
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False)
        )

        # 分支後處理
        self.process1 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)
        self.process2 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)
        self.process3 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)
        self.process4 = ConvBNReLU(branch_channels, branch_channels, 3, 1, 1)

        # concat 後壓回 out_channels
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels * 5, out_channels, kernel_size=1, bias=False)
        )

        # shortcut
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # 原始空間尺寸
        h, w = x.shape[-2:]

        # 原尺度分支
        x0 = self.scale0(x)

        # scale1
        x1 = self.scale1(x)
        x1 = F.interpolate(x1, size=(h, w), mode="bilinear", align_corners=False)
        x1 = self.process1(x1 + x0)

        # scale2
        x2 = self.scale2(x)
        x2 = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
        x2 = self.process2(x2 + x1)

        # scale3
        x3 = self.scale3(x)
        x3 = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=False)
        x3 = self.process3(x3 + x2)

        # scale4
        x4 = self.scale4(x)
        x4 = F.interpolate(x4, size=(h, w), mode="bilinear", align_corners=False)
        x4 = self.process4(x4 + x3)

        # concat
        out = torch.cat([x0, x1, x2, x3, x4], dim=1)

        # compression + shortcut
        out = self.compression(out) + self.shortcut(x)

        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
        )

    def forward(self, x):
        return self.block(x)


class DDRNet23SlimBackbone(nn.Module):
    def __init__(self, planes=32):
        super().__init__()

        # stem：先降到 1/4 解析度
        self.conv1 = nn.Sequential(
            ConvBNReLU(3, planes, 3, 2, 1),
            ConvBNReLU(planes, planes, 3, 2, 1)
        )

        # 主幹分支
        self.layer1 = make_layer(planes, planes, num_blocks=2, stride=1)
        self.layer2 = make_layer(planes, planes * 2, num_blocks=2, stride=2)
        self.layer3 = make_layer(planes * 2, planes * 4, num_blocks=2, stride=2)
        self.layer4 = make_layer(planes * 4, planes * 8, num_blocks=2, stride=2)
        self.layer5 = make_layer(planes * 8, planes * 8, num_blocks=1, stride=1, last_no_relu=True)

        # 高解析度分支
        self.layer3_ = make_layer(planes * 2, planes * 2, num_blocks=2, stride=1)
        self.layer4_ = make_layer(planes * 2, planes * 2, num_blocks=2, stride=1)
        self.layer5_ = make_layer(planes * 2, planes * 2, num_blocks=1, stride=1, last_no_relu=True)

        # low -> high 壓縮
        self.compression3 = nn.Sequential(
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False)
        )

        self.compression4 = nn.Sequential(
            nn.BatchNorm2d(planes * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False)
        )

        # high -> low 下採樣
        self.down3 = nn.Sequential(
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.down4 = nn.Sequential(
            nn.BatchNorm2d(planes * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False)
        )

        # DAPPM
        self.dappm = DAPPM(
            in_channels=planes * 8,
            branch_channels=planes,
            out_channels=planes * 2
        )

    def forward(self, x):
        # stem
        x = self.conv1(x)

        # 主幹前兩層
        x = self.layer1(x)
        x = self.layer2(x)

        # 分成高、低解析分支
        x_high = self.layer3_(x)
        x_low = self.layer3(x)

        # 第一次雙路融合
        x_high = x_high + F.interpolate(
            self.compression3(x_low),
            size=x_high.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x_low = x_low + self.down3(x_high)

        # 第二層
        x_high = self.layer4_(F.relu(x_high))
        x_low = self.layer4(F.relu(x_low))

        # 第二次雙路融合
        x_high = x_high + F.interpolate(
            self.compression4(x_low),
            size=x_high.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x_low = x_low + self.down4(x_high)

        # 保存 aux 特徵
        aux_feature = F.relu(x_high)

        # 最後一層
        x_high = self.layer5_(aux_feature)
        x_low = self.layer5(F.relu(x_low))

        # low branch 走 DAPPM
        x_low_context = self.dappm(F.relu(x_low))

        # 上採樣到 high branch 尺寸
        x_low_up = F.interpolate(
            x_low_context,
            size=x_high.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # 拼接融合
        fused_feature = torch.cat([x_high, x_low_up], dim=1)

        return fused_feature, aux_feature


class DDRNet23SlimBinarySeg(nn.Module):
    def __init__(self, num_classes=1, planes=32, use_aux=True):
        super().__init__()

        # 是否使用 auxiliary head
        self.use_aux = use_aux

        # backbone
        self.backbone = DDRNet23SlimBackbone(planes=planes)

        # 主輸出 head
        self.seg_head = SegHead(
            in_channels=planes * 4,
            mid_channels=planes * 2,
            num_classes=num_classes
        )

        # 輔助輸出 head
        self.aux_head = SegHead(
            in_channels=planes * 2,
            mid_channels=planes,
            num_classes=num_classes
        )

    def forward(self, x):
        # 記錄輸入尺寸
        input_h, input_w = x.shape[-2:]

        # backbone forward
        fused_feature, aux_feature = self.backbone(x)

        # main logits
        main_logits = self.seg_head(fused_feature)
        main_logits = F.interpolate(
            main_logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False
        )

        # 訓練階段才回傳 aux
        if self.training and self.use_aux:
            aux_logits = self.aux_head(aux_feature)
            aux_logits = F.interpolate(
                aux_logits,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False
            )
            return main_logits, aux_logits

        # eval/test 只回主輸出
        return main_logits


# ============================================================
# 5) Loss
# ============================================================

def dice_loss_from_logits(logits, targets, eps=1e-6):
    # logits -> probability
    probs = torch.sigmoid(logits)

    # flatten 成 [B, -1]
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    # intersection
    intersection = (probs * targets).sum(dim=1)

    # union
    union = probs.sum(dim=1) + targets.sum(dim=1)

    # Dice
    dice = (2.0 * intersection + eps) / (union + eps)

    # Dice loss
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        # BCE with logits
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE
        bce = self.bce(logits, targets)

        # Dice
        dice = dice_loss_from_logits(logits, targets)

        # 總 loss
        total = bce + dice

        # 回傳 total / bce / dice，方便紀錄
        return total, bce.detach(), dice.detach()


# ============================================================
# 6) Metrics
# ============================================================

def safe_div(n, d):
    return float(n) / float(d) if d > 0 else 0.0


def binary_surface(mask: np.ndarray):
    # 若整張 mask 都沒有前景，回傳全 False
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    # 先做 erosion
    eroded = binary_erosion(mask)

    # surface = 原 mask XOR erosion 後 mask
    surface = mask.astype(bool) ^ eroded.astype(bool)

    return surface


def hd95_and_assd(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 兩者都空白，視為 perfect
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0, True

    # 一空一不空，邊界距離不可用
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan, False

    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)

    # 若兩邊 surface 都沒有點，也視為 perfect
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0, 0.0, True

    # 若任一側 surface 沒有點，則不可用
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return np.nan, np.nan, False

    # 分別計算距離 transform
    dt_gt = distance_transform_edt(~gt_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    # pred surface 到 gt surface 的距離
    dist_pred_to_gt = dt_gt[pred_surface]

    # gt surface 到 pred surface 的距離
    dist_gt_to_pred = dt_pred[gt_surface]

    # 合併雙向距離
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)

    # HD95
    hd95 = np.percentile(all_dists, 95)

    # ASSD
    assd = np.mean(all_dists)

    return float(hd95), float(assd), True


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # confusion matrix
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    # overlap metrics
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    # 空白特例：若 pred/gt 都是空白，precision/recall 視為 1
    if pred.sum() == 0 and gt.sum() == 0:
        precision = 1.0
        recall = 1.0
    else:
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


def aggregate_metrics(rows):
    # rows 是 list[dict]
    if len(rows) == 0:
        return {}

    out = {}
    out["Dice"] = float(np.mean([r["Dice"] for r in rows]))
    out["IoU"] = float(np.mean([r["IoU"] for r in rows]))
    out["Acc"] = float(np.mean([r["Acc"] for r in rows]))
    out["Precision"] = float(np.mean([r["Precision"] for r in rows]))
    out["Recall"] = float(np.mean([r["Recall"] for r in rows]))

    hd_values = [r["HD95"] for r in rows if not np.isnan(r["HD95"])]
    assd_values = [r["ASSD"] for r in rows if not np.isnan(r["ASSD"])]

    out["HD95"] = float(np.mean(hd_values)) if len(hd_values) > 0 else np.nan
    out["ASSD"] = float(np.mean(assd_values)) if len(assd_values) > 0 else np.nan
    out["Valid_Boundary_Cases"] = int(np.sum([r["Valid_Boundary_Case"] for r in rows]))
    out["Num_Images"] = len(rows)

    return out


def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    # logits -> probs
    probs = torch.sigmoid(logits)

    # threshold 成二值輸出
    preds = (probs >= threshold).float()

    # flatten
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # confusion parts
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    # metrics
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


def batch_metrics_from_probs(probs, targets, threshold=0.5, eps=1e-7):
    # 直接從 probability 計算，不再重新 forward
    preds = (probs >= threshold).float()

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


def collect_probs_and_targets(model, loader):
    # 把整個 loader 的 sigmoid probs 與 targets 收集起來
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)

            with autocast(enabled=AMP_ENABLED):
                logits = model(images)

            probs = torch.sigmoid(logits).detach().cpu()
            targets = masks.detach().cpu()

            all_probs.append(probs)
            all_targets.append(targets)

    if len(all_probs) == 0:
        return None, None

    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return all_probs, all_targets


def select_best_threshold_on_loader(model, loader, candidate_thresholds):
    # 在 validation set 上搜尋最佳 threshold
    probs, targets = collect_probs_and_targets(model, loader)

    # 若沒資料，退回 0.5
    if probs is None or targets is None or probs.size(0) == 0:
        return 0.5, {
            "dice": 0.0,
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    best_threshold = 0.5
    best_metrics = None

    for th in candidate_thresholds:
        metrics = batch_metrics_from_probs(probs, targets, threshold=th, eps=EPS)

        # 排序規則：
        # 1. 先比 Dice
        # 2. Dice 相同時比 IoU
        # 3. 再相同時偏好更接近 0.5 的 threshold
        if best_metrics is None:
            best_threshold = th
            best_metrics = metrics
            continue

        better = False

        if metrics["dice"] > best_metrics["dice"]:
            better = True
        elif metrics["dice"] == best_metrics["dice"] and metrics["iou"] > best_metrics["iou"]:
            better = True
        elif (
            metrics["dice"] == best_metrics["dice"]
            and metrics["iou"] == best_metrics["iou"]
            and abs(th - 0.5) < abs(best_threshold - 0.5)
        ):
            better = True

        if better:
            best_threshold = th
            best_metrics = metrics

    return best_threshold, best_metrics


# ============================================================
# 7) Train / Eval
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    # 切到 train mode
    model.train()

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

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=AMP_ENABLED):
            outputs = model(images)

            # 訓練模式下，model 會回傳 (main_logits, aux_logits)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs

                main_loss, main_bce, main_dice_loss = criterion(logits, masks)
                aux_loss, _, _ = criterion(aux_logits, masks)

                # main + aux
                loss = main_loss + AUX_LOSS_WEIGHT * aux_loss

                batch_bce = main_bce
                batch_dice_loss = main_dice_loss
                metric_logits = logits

            else:
                # 理論上 train 時通常不會走這裡，但保險留著
                logits = outputs
                loss, batch_bce, batch_dice_loss = criterion(logits, masks)
                metric_logits = logits

        # 反向傳播
        if AMP_ENABLED:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        metrics = batch_metrics_from_logits(metric_logits, masks, THRESHOLD, EPS)

        total_loss += loss.item() * batch_size
        total_bce += batch_bce.item() * batch_size
        total_dice_loss += batch_dice_loss.item() * batch_size
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


def eval_one_epoch(model, loader, criterion):
    # 切到 eval mode
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
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
                loss, batch_bce, batch_dice_loss = criterion(logits, masks)

            batch_size = images.size(0)
            metrics = batch_metrics_from_logits(logits, masks, THRESHOLD, EPS)

            total_loss += loss.item() * batch_size
            total_bce += batch_bce.item() * batch_size
            total_dice_loss += batch_dice_loss.item() * batch_size
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
# 8) Save Test Predictions
# ============================================================

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
            preds = (probs >= threshold).float()

            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(images.size(0)):
                stem = stems[i]

                # 儲存 resize 後的原圖，方便與 pred 對齊
                raw_img = Image.open(image_paths[i]).convert("RGB")
                raw_img = raw_img.resize(IMAGE_SIZE, Image.BILINEAR)
                raw_img.save(os.path.join(image_dir, f"{stem}.png"))

                # 儲存 gt mask
                gt_mask = (masks_np[i, 0] * 255).astype(np.uint8)
                Image.fromarray(gt_mask).save(os.path.join(gt_dir, f"{stem}.png"))

                # 儲存 pred mask
                pred_mask = (preds_np[i, 0] * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(os.path.join(pred_dir, f"{stem}.png"))


# ============================================================
# 9) Checkpoint
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
    best_threshold,
    history,
    epochs_no_improve
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "history": history,
            "epochs_no_improve": epochs_no_improve,
        },
        checkpoint_path
    )


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
        ckpt.get("best_threshold", 0.5),
        ckpt["history"],
        ckpt["epochs_no_improve"]
    )


# ============================================================
# 10) Per-image Evaluation on Test Set
# ============================================================

def evaluate_test_per_image(model, loader, fold_name, threshold=0.5):
    model.eval()

    all_rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)
            stems = batch["stem"]
            image_paths = batch["image_path"]
            centers = batch["center"]

            with autocast(enabled=AMP_ENABLED):
                logits = model(images)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(images.size(0)):
                pred_bin = preds_np[i, 0].astype(np.uint8)
                gt_bin = masks_np[i, 0].astype(np.uint8)

                metrics = compute_binary_metrics(pred_bin, gt_bin)

                row = {
                    "Fold": fold_name,
                    "Center": centers[i],
                    "Stem": stems[i],
                    "Image_Path": image_paths[i],
                    "Threshold": float(threshold),
                    "Dice": metrics["Dice"],
                    "IoU": metrics["IoU"],
                    "Acc": metrics["Acc"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "HD95": metrics["HD95"],
                    "ASSD": metrics["ASSD"],
                    "Valid_Boundary_Case": metrics["Valid_Boundary_Case"],
                }
                all_rows.append(row)

    return all_rows


# ============================================================
# 11) Result Rebuild Helpers
# ============================================================

def parse_result_txt(result_txt_path):
    data = {}

    with open(result_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

    # 新版 result.txt 會有這些欄位
    # 舊版沒有的話，做 fallback
    return {
        "fold": data["fold"],
        "best_epoch": int(data["best_epoch"]),
        "best_val_dice": float(data["best_val_dice"]),
        "best_val_iou": float(data["best_val_iou"]),
        "report_val_dice": float(data.get("report_val_dice", data["best_val_dice"])),
        "report_val_iou": float(data.get("report_val_iou", data["best_val_iou"])),
        "best_threshold": float(data.get("best_threshold", 0.5)),
        "test_loss": float(data["test_loss"]),
        "test_dice": float(data["test_dice"]),
        "test_iou": float(data["test_iou"]),
        "test_precision": float(data["test_precision"]),
        "test_recall": float(data["test_recall"]),
        "delta_shift_dice": float(data["delta_shift_dice"]),
        "delta_shift_iou": float(data["delta_shift_iou"]),
        "time_sec": float(data["time_sec"]),
    }


def rebuild_all_fold_results_from_disk():
    rows = []

    for fold_name in RUN_FOLDS:
        result_txt = os.path.join(OUT_ROOT, fold_name, "stage3A_results", "result.txt")

        if os.path.isfile(result_txt):
            rows.append(parse_result_txt(result_txt))
        else:
            print(f"[WARN] 找不到 result.txt，略過：{result_txt}")

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("fold").reset_index(drop=True)

    out_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return df


def rebuild_fold_val_data_metrics(all_fold_df: pd.DataFrame):
    if all_fold_df.empty:
        return pd.DataFrame()

    df = all_fold_df.copy()

    # 改用 report_val_dice / report_val_iou
    # 也就是 validation threshold search 後真正拿來報告的值
    df["Val_Dice"] = df["report_val_dice"]
    df["ΔDice"] = df["delta_shift_dice"]
    df["Val_IoU"] = df["report_val_iou"]
    df["ΔIoU"] = df["delta_shift_iou"]

    out_df = df[["fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"]].copy()
    out_df.columns = ["Fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"]

    out_csv = os.path.join(OUT_ROOT, "fold_val_data_metrics.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return out_df


def rebuild_per_image_metrics_from_disk():
    all_rows = []

    for fold_name in RUN_FOLDS:
        per_image_csv = os.path.join(OUT_ROOT, fold_name, "stage3A_results", "per_image_metrics.csv")

        if os.path.isfile(per_image_csv):
            df = pd.read_csv(per_image_csv)
            all_rows.append(df)
        else:
            print(f"[WARN] 找不到 per_image_metrics.csv，略過：{per_image_csv}")

    if len(all_rows) == 0:
        return pd.DataFrame()

    all_df = pd.concat(all_rows, axis=0, ignore_index=True)

    out_csv = os.path.join(OUT_ROOT, "per_image_metrics.csv")
    all_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return all_df


def rebuild_center_level_metrics(per_image_df: pd.DataFrame):
    if per_image_df.empty:
        return pd.DataFrame()

    rows = []

    for center in sorted(per_image_df["Center"].unique()):
        sub = per_image_df[per_image_df["Center"] == center]

        metric_rows = []
        for _, r in sub.iterrows():
            metric_rows.append(
                {
                    "Dice": float(r["Dice"]),
                    "IoU": float(r["IoU"]),
                    "Acc": float(r["Acc"]),
                    "Precision": float(r["Precision"]),
                    "Recall": float(r["Recall"]),
                    "HD95": float(r["HD95"]) if not pd.isna(r["HD95"]) else np.nan,
                    "ASSD": float(r["ASSD"]) if not pd.isna(r["ASSD"]) else np.nan,
                    "Valid_Boundary_Case": int(r["Valid_Boundary_Case"]),
                }
            )

        agg = aggregate_metrics(metric_rows)

        rows.append(
            {
                "Center": center,
                "Num_images": agg["Num_Images"],
                "Dice": agg["Dice"],
                "IoU": agg["IoU"],
                "Acc": agg["Acc"],
                "Precision": agg["Precision"],
                "Recall": agg["Recall"],
                "HD95": agg["HD95"],
                "ASSD": agg["ASSD"],
            }
        )

    center_df = pd.DataFrame(rows).sort_values("Center").reset_index(drop=True)

    out_csv = os.path.join(OUT_ROOT, "center_level_metrics.csv")
    center_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return center_df


def write_loco_summary(all_fold_df: pd.DataFrame, center_df: pd.DataFrame):
    summary_path = os.path.join(OUT_ROOT, "loco_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 3-A: DDRNet-23-slim + LOCO (Student Baseline, Clean Version)\n")
        f.write("=" * 70 + "\n\n")

        if not all_fold_df.empty:
            mean_test_dice = all_fold_df["test_dice"].mean()
            mean_test_iou = all_fold_df["test_iou"].mean()
            mean_delta_dice = all_fold_df["delta_shift_dice"].mean()
            mean_delta_iou = all_fold_df["delta_shift_iou"].mean()

            f.write(f"LOCO mean test Dice: {mean_test_dice:.6f}\n")
            f.write(f"LOCO mean test IoU : {mean_test_iou:.6f}\n")
            f.write(f"Mean ΔDice         : {mean_delta_dice:.6f}\n")
            f.write(f"Mean ΔIoU          : {mean_delta_iou:.6f}\n\n")

        if not center_df.empty:
            best_idx = center_df["Dice"].idxmax()
            worst_idx = center_df["Dice"].idxmin()

            f.write("Best center by Dice:\n")
            f.write(center_df.loc[best_idx].to_string() + "\n\n")

            f.write("Worst center by Dice:\n")
            f.write(center_df.loc[worst_idx].to_string() + "\n\n")


# ============================================================
# 12) Single Fold Runner
# ============================================================

def run_single_fold(fold_name):
    print("\n" + "=" * 100)
    print(f"[START] {fold_name}")
    print("=" * 100)

    fold_dir = os.path.join(LOCO_ROOT, fold_name)
    result_root = os.path.join(OUT_ROOT, fold_name, "stage3A_results")

    train_csv = os.path.join(fold_dir, "train.csv")
    val_csv = os.path.join(fold_dir, "val.csv")
    test_csv = os.path.join(fold_dir, "test.csv")

    history_csv_path = os.path.join(result_root, "history.csv")
    checkpoint_path = os.path.join(result_root, "best_model.pth")
    result_txt_path = os.path.join(result_root, "result.txt")
    per_image_csv_path = os.path.join(result_root, "per_image_metrics.csv")
    test_pred_dir = os.path.join(result_root, "test_predictions")

    # 若指定 reset，此 fold 先刪掉
    if fold_name in RESET_FOLDS and os.path.isdir(os.path.join(OUT_ROOT, fold_name)):
        print(f"[RESET] 刪除舊結果：{os.path.join(OUT_ROOT, fold_name)}")
        shutil.rmtree(os.path.join(OUT_ROOT, fold_name), ignore_errors=True)

    # 若已完成且選擇 skip
    if SKIP_COMPLETED_FOLDS and os.path.isfile(result_txt_path):
        print(f"[SKIP] 已完成：{fold_name}")
        return

    os.makedirs(result_root, exist_ok=True)

    # 準備資料集
    train_dataset = PolypDataset(train_csv, image_size=IMAGE_SIZE, is_train=True)
    val_dataset = PolypDataset(val_csv, image_size=IMAGE_SIZE, is_train=False)
    test_dataset = PolypDataset(test_csv, image_size=IMAGE_SIZE, is_train=False)

    print(f"[DATA] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    # train loader:
    # drop_last=True 可避免最後一個 batch 只有 1 張，讓 BN 更穩定
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    # 建模
    model = DDRNet23SlimBinarySeg(
        num_classes=1,
        planes=PLANES,
        use_aux=True
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

    # 訓練狀態
    start_epoch = 0
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = -1
    best_threshold = 0.5
    epochs_no_improve = 0
    history = []

    # 若允許 resume 且 checkpoint 存在，才續跑
    if AUTO_RESUME and os.path.isfile(checkpoint_path):
        print(f"[RESUME] 載入 checkpoint：{checkpoint_path}")
        (
            start_epoch,
            best_val_dice,
            best_val_iou,
            best_epoch,
            best_threshold,
            history,
            epochs_no_improve
        ) = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    t0 = time.time()

    # 訓練 loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        # train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler)

        # val
        val_metrics = eval_one_epoch(model, val_loader, criterion)

        # scheduler 根據 val dice 調 lr
        scheduler.step(val_metrics["dice"])

        # 記錄當前 lr
        current_lr = optimizer.param_groups[0]["lr"]

        # 寫進 history
        epoch_record = {
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_bce": train_metrics["bce"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_bce": val_metrics["bce"],
            "val_dice_loss": val_metrics["dice_loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(epoch_record)

        # 顯示 log
        print(
            f"[{fold_name}] "
            f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
            f"lr={current_lr:.6e} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        # 是否更新 best（仍用原本 val_dice 做 checkpoint 選模）
        improved = val_metrics["dice"] > best_val_dice

        if improved:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch + 1

            # 在最佳 epoch 上，用 validation set 搜最佳 threshold
            best_threshold, searched_val_metrics = select_best_threshold_on_loader(
                model=model,
                loader=val_loader,
                candidate_thresholds=VAL_THRESHOLD_CANDIDATES
            )

            epochs_no_improve = 0

            # 存最佳模型 checkpoint
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_dice=best_val_dice,
                best_val_iou=best_val_iou,
                best_epoch=best_epoch,
                best_threshold=best_threshold,
                history=history,
                epochs_no_improve=epochs_no_improve
            )

            print(
                f"[BEST] {fold_name} "
                f"best_val_dice={best_val_dice:.4f} "
                f"best_val_iou={best_val_iou:.4f} "
                f"best_threshold={best_threshold:.2f} "
                f"searched_val_dice={searched_val_metrics['dice']:.4f} "
                f"@ epoch {best_epoch}"
            )

        else:
            epochs_no_improve += 1

        # 每個 epoch 都存 history.csv
        pd.DataFrame(history).to_csv(history_csv_path, index=False, encoding="utf-8-sig")

        # early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"[EARLY STOP] {fold_name} at epoch {epoch + 1}")
            break

    # 載入最佳權重
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        best_threshold = ckpt.get("best_threshold", 0.5)

    # ========================================================
    # 最終 validation threshold-aware metrics
    # ========================================================
    report_val_threshold, val_threshold_metrics = select_best_threshold_on_loader(
        model=model,
        loader=val_loader,
        candidate_thresholds=VAL_THRESHOLD_CANDIDATES
    )

    # 理論上應與 checkpoint 內 best_threshold 相同；
    # 這裡直接採用重新計算的結果，確保與當前最佳模型一致
    best_threshold = report_val_threshold

    report_val_dice = val_threshold_metrics["dice"]
    report_val_iou = val_threshold_metrics["iou"]

    # ========================================================
    # test evaluation
    # loss 仍用 eval_one_epoch 計算
    # overlap metrics 改用最佳 validation threshold 計算
    # ========================================================
    test_metrics = eval_one_epoch(model, test_loader, criterion)

    test_probs, test_targets = collect_probs_and_targets(model, test_loader)
    best_th_test_overlap = batch_metrics_from_probs(
        test_probs,
        test_targets,
        threshold=best_threshold,
        eps=EPS
    )

    # 用最佳 threshold 覆蓋 test overlap metrics
    test_metrics["dice"] = best_th_test_overlap["dice"]
    test_metrics["iou"] = best_th_test_overlap["iou"]
    test_metrics["precision"] = best_th_test_overlap["precision"]
    test_metrics["recall"] = best_th_test_overlap["recall"]

    # 計算 Δshift
    delta_shift_dice = report_val_dice - test_metrics["dice"]
    delta_shift_iou = report_val_iou - test_metrics["iou"]

    # 每張圖詳細評估（改用最佳 threshold）
    per_image_rows = evaluate_test_per_image(
        model=model,
        loader=test_loader,
        fold_name=fold_name,
        threshold=best_threshold
    )
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(per_image_csv_path, index=False, encoding="utf-8-sig")

    # 儲存 test prediction（改用最佳 threshold）
    if SAVE_TEST_PRED:
        save_test_predictions(
            model=model,
            loader=test_loader,
            save_dir=test_pred_dir,
            threshold=best_threshold
        )

    total_time = time.time() - t0

    # 寫 result.txt
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(f"fold: {fold_name}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_dice: {best_val_dice:.10f}\n")
        f.write(f"best_val_iou: {best_val_iou:.10f}\n")
        f.write(f"report_val_dice: {report_val_dice:.10f}\n")
        f.write(f"report_val_iou: {report_val_iou:.10f}\n")
        f.write(f"best_threshold: {best_threshold:.10f}\n")
        f.write(f"test_loss: {test_metrics['loss']:.10f}\n")
        f.write(f"test_dice: {test_metrics['dice']:.10f}\n")
        f.write(f"test_iou: {test_metrics['iou']:.10f}\n")
        f.write(f"test_precision: {test_metrics['precision']:.10f}\n")
        f.write(f"test_recall: {test_metrics['recall']:.10f}\n")
        f.write(f"delta_shift_dice: {delta_shift_dice:.10f}\n")
        f.write(f"delta_shift_iou: {delta_shift_iou:.10f}\n")
        f.write(f"time_sec: {total_time:.4f}\n")

    print(f"[DONE] {fold_name}")
    print(f"  best_epoch       = {best_epoch}")
    print(f"  best_val_dice    = {best_val_dice:.4f}")
    print(f"  best_val_iou     = {best_val_iou:.4f}")
    print(f"  report_val_dice  = {report_val_dice:.4f}")
    print(f"  report_val_iou   = {report_val_iou:.4f}")
    print(f"  best_threshold   = {best_threshold:.2f}")
    print(f"  test_dice        = {test_metrics['dice']:.4f}")
    print(f"  test_iou         = {test_metrics['iou']:.4f}")
    print(f"  test_precision   = {test_metrics['precision']:.4f}")
    print(f"  test_recall      = {test_metrics['recall']:.4f}")
    print(f"  delta_shift_dice = {delta_shift_dice:.4f}")
    print(f"  delta_shift_iou  = {delta_shift_iou:.4f}")
    print(f"  time_sec         = {total_time:.2f}")

    # 是否刪 checkpoint
    if DELETE_CHECKPOINT_AFTER_FINISH and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)


# ============================================================
# 13) Main
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 100)
    print("Stage 3-A: DDRNet-23-slim + LOCO (Student Baseline, Clean Version)")
    print("=" * 100)
    print("DEVICE =", DEVICE)
    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))
    print("LOCO_ROOT =", LOCO_ROOT)
    print("OUT_ROOT =", OUT_ROOT)
    print("RUN_FOLDS =", RUN_FOLDS)
    print("IMAGE_SIZE =", IMAGE_SIZE)
    print("BATCH_SIZE =", BATCH_SIZE)
    print("NUM_EPOCHS =", NUM_EPOCHS)
    print("LR =", LR)
    print("WEIGHT_DECAY =", WEIGHT_DECAY)
    print("EARLY_STOPPING_PATIENCE =", EARLY_STOPPING_PATIENCE)
    print("SCHEDULER_PATIENCE =", SCHEDULER_PATIENCE)
    print("VAL_THRESHOLD_CANDIDATES =", VAL_THRESHOLD_CANDIDATES)
    print("FORCE_CLEAN_RUN =", FORCE_CLEAN_RUN)
    print("AUTO_RESUME =", AUTO_RESUME)
    print("AMP_ENABLED =", AMP_ENABLED)
    print("=" * 100)

    # 逐 fold 跑
    for fold_name in RUN_FOLDS:
        run_single_fold(fold_name)

    # 重建總表
    all_fold_df = rebuild_all_fold_results_from_disk()
    fold_val_df = rebuild_fold_val_data_metrics(all_fold_df)
    per_image_df = rebuild_per_image_metrics_from_disk()
    center_df = rebuild_center_level_metrics(per_image_df)
    write_loco_summary(all_fold_df, center_df)

    print("\n" + "=" * 100)
    print("[FINAL SUMMARY FILES]")
    print("=" * 100)
    print(os.path.join(OUT_ROOT, "all_fold_results.csv"))
    print(os.path.join(OUT_ROOT, "fold_val_data_metrics.csv"))
    print(os.path.join(OUT_ROOT, "per_image_metrics.csv"))
    print(os.path.join(OUT_ROOT, "center_level_metrics.csv"))
    print(os.path.join(OUT_ROOT, "loco_summary.txt"))
    print("=" * 100)

    # 額外印出 center-level 結果，方便你先看
    if not center_df.empty:
        print("\n[Center-level Metrics]")
        print(center_df.to_string(index=False))

    if not fold_val_df.empty:
        print("\n[Fold Val Data Metrics]")
        print(fold_val_df.to_string(index=False))


if __name__ == "__main__":
    main()