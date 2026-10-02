# ============================================================
# Stage 3-B
# SegFormer-B0 + Center-level LOCO for Polyp Binary Segmentation
# ------------------------------------------------------------
# 說明：
# 1. 這是 Student Baseline
# 2. 不含 KD
# 3. 不含 Consistency Learning
# 4. 前處理、LOCO split、augmentation、metrics 輸出格式
#    對齊你 DDRNet-23-slim + LOCO baseline
# 5. 輸出：
#    - all_fold_results.csv
#    - fold_val_data_metrics.csv
#    - center_level_metrics.csv
#    - per_image_metrics.csv
#    - loco_summary.txt
# ============================================================

import os
import csv
import time
import random
import gc
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

from transformers import SegformerForSemanticSegmentation, SegformerConfig


# ============================================================
# 0) Main Config
# ============================================================

# LOCO split 根目錄
LOCO_ROOT = r"./data\loco_1537_clean"

# 輸出根目錄
OUT_ROOT = r"./data\stage3B_segformer_b0_loco_baseline"

# 每個 fold 內部結果資料夾名稱
RESULT_DIR_NAME = "stage3B_results"

# 要跑的 folds
RUN_FOLDS = [
    "fold_test_C1",
    "fold_test_C2",
    "fold_test_C3",
    "fold_test_C4",
    "fold_test_C5",
    "fold_test_C6",
]

# SegFormer-B0 pretrained 權重
# 若第一次執行需要網路下載；下載後會快取到本機
PRETRAINED_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# 是否載入 ImageNet/ADE 預訓練權重
# 建議 True，因為從頭訓練通常比較不穩
LOAD_PRETRAINED = True

# 影像尺寸，保持與你 DDRNet/SegFormer 單視圖前處理一致
IMAGE_SIZE = (352, 352)  # (W, H)

# Batch size
BATCH_SIZE = 8

# 訓練 epoch
NUM_EPOCHS = 100

# learning rate
# SegFormer-B0 用 1e-4 可先跑 baseline；若震盪可改 6e-5
LR = 1e-4

# weight decay
WEIGHT_DECAY = 1e-4

# early stopping
EARLY_STOPPING_PATIENCE = 12

# ReduceLROnPlateau 設定
SCHEDULER_PATIENCE = 4
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# segmentation threshold
THRESHOLD = 0.5

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

# 儲存 test 預測圖
SAVE_TEST_PRED = True

# 是否刪除 checkpoint
DELETE_CHECKPOINT_AFTER_FINISH = False

# reset 某些 fold 時可放進來，例如 {"fold_test_C4"}
RESET_FOLDS = set()

# 是否跳過已完成 fold
SKIP_COMPLETED_FOLDS = True

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

    # 固定 PyTorch CPU random
    torch.manual_seed(seed)

    # 若有 GPU，固定 CUDA random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 開啟 deterministic，讓結果較可重現
    torch.backends.cudnn.deterministic = True

    # 關閉 benchmark，避免 cudnn 自動選不同演算法
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Preprocessing
#    與 DDRNet-23-slim + LOCO baseline 對齊
# ============================================================

def resize_pair(img: Image.Image, mask: Image.Image, image_size=(352, 352)):
    # image 用 bilinear resize
    img = img.resize(image_size, Image.BILINEAR)

    # mask 用 nearest resize，避免標籤被插值污染
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
    # 以機率 p 做小角度旋轉
    if random.random() < p:
        angle = random.uniform(-angle_limit, angle_limit)

        # image 使用 bilinear
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        # mask 使用 nearest
        mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

    return img, mask


def random_scale_shift_pair(
    img: Image.Image,
    mask: Image.Image,
    p=0.5,
    scale_low=0.90,
    scale_high=1.10
):
    # 未觸發 augmentation 就直接回傳
    if random.random() >= p:
        return img, mask

    # 取得原圖寬高
    w, h = img.size

    # 隨機縮放倍率
    scale = random.uniform(scale_low, scale_high)

    # 計算縮放後尺寸
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    # image 用 bilinear resize
    img_scaled = img.resize((new_w, new_h), Image.BILINEAR)

    # mask 用 nearest resize
    mask_scaled = mask.resize((new_w, new_h), Image.NEAREST)

    # 若放大，隨機裁切回原尺寸
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
    # 隨機亮度
    if random.random() < p:
        brightness_factor = random.uniform(
            1.0 - brightness_limit,
            1.0 + brightness_limit
        )
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 隨機對比
    if random.random() < p:
        contrast_factor = random.uniform(
            1.0 - contrast_limit,
            1.0 + contrast_limit
        )
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def image_to_chw_float(img: Image.Image):
    # PIL image -> numpy float32，並正規化到 [0,1]
    img_np = np.array(img, dtype=np.float32) / 255.0

    # HWC -> CHW，符合 PyTorch Conv2d 輸入格式
    img_np = np.transpose(img_np, (2, 0, 1))

    return img_np


def mask_to_1hw_float(mask: Image.Image):
    # mask 轉成 numpy
    mask_np = np.array(mask, dtype=np.uint8)

    # 大於 127 視為前景
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
#    讀取 fold 內 train.csv / val.csv / test.csv
# ============================================================

class PolypDataset(Dataset):
    def __init__(self, csv_file, image_size=(352, 352), is_train=False):
        # 儲存 csv rows
        self.rows = []

        # 儲存 image size
        self.image_size = image_size

        # 是否為 train mode
        self.is_train = is_train

        # 讀取 csv
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        # 回傳資料筆數
        return len(self.rows)

    def __getitem__(self, idx):
        # 取得第 idx 筆 row
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
# 4) SegFormer-B0 Binary Segmentation Model
# ============================================================

class SegFormerB0BinarySeg(nn.Module):
    def __init__(
        self,
        pretrained_name="nvidia/segformer-b0-finetuned-ade-512-512",
        load_pretrained=True
    ):
        super().__init__()

        self.pretrained_name = pretrained_name
        self.load_pretrained = load_pretrained

        if load_pretrained:
            try:
                print(f"[MODEL] Loading pretrained SegFormer-B0 from: {pretrained_name}")

                # 載入 SegFormer-B0 預訓練模型
                # num_labels=1 表示輸出單通道 binary logits
                # ignore_mismatched_sizes=True 讓 ADE 150 類 head 被替換成 1 類 head
                self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                    pretrained_name,
                    num_labels=1,
                    ignore_mismatched_sizes=True
                )

                print("[MODEL] Pretrained SegFormer-B0 loaded successfully.")

            except Exception as e:
                print("[WARN] Failed to load pretrained SegFormer-B0.")
                print("[WARN] Error:", str(e))
                print("[WARN] Falling back to randomly initialized SegFormer-B0 config.")

                # 若本機沒有 cache 或下載失敗，使用 B0 config 從頭初始化
                config = SegformerConfig(
                    num_labels=1,
                    num_channels=3,
                    depths=[2, 2, 2, 2],
                    sr_ratios=[8, 4, 2, 1],
                    hidden_sizes=[32, 64, 160, 256],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    num_attention_heads=[1, 2, 5, 8],
                    mlp_ratios=[4, 4, 4, 4],
                    decoder_hidden_size=256
                )

                self.segformer = SegformerForSemanticSegmentation(config)

        else:
            print("[MODEL] Using randomly initialized SegFormer-B0.")

            # SegFormer-B0 config
            config = SegformerConfig(
                num_labels=1,
                num_channels=3,
                depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[32, 64, 160, 256],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                decoder_hidden_size=256
            )

            self.segformer = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        # 記錄輸入尺寸，例如 352x352
        input_h, input_w = x.shape[-2:]

        # SegFormer forward
        outputs = self.segformer(pixel_values=x)

        # HuggingFace SegFormer 輸出 logits 通常是低解析度
        logits = outputs.logits

        # 將 logits 上採樣回原輸入尺寸
        logits = F.interpolate(
            logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False
        )

        return logits


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

    # Dice score
    dice = (2.0 * intersection + eps) / (union + eps)

    # Dice loss
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

        # BCE with logits
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE loss
        bce = self.bce(logits, targets)

        # Dice loss
        dice = dice_loss_from_logits(logits, targets)

        # total loss
        total = bce + dice

        # 回傳 total / bce / dice，方便紀錄
        return total, bce.detach(), dice.detach()


# ============================================================
# 6) Metrics
# ============================================================

def safe_div(n, d):
    # 安全除法，避免 denominator 為 0
    return float(n) / float(d) if d > 0 else 0.0


def binary_surface(mask: np.ndarray):
    # 若整張 mask 沒有前景，回傳全 False
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    # erosion 後取得內部區域
    eroded = binary_erosion(mask)

    # surface = 原 mask XOR erosion 後 mask
    surface = mask.astype(bool) ^ eroded.astype(bool)

    return surface


def hd95_and_assd(pred: np.ndarray, gt: np.ndarray):
    # 轉成 bool
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # pred 和 gt 都空，視為 perfect
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0, True

    # 一空一不空，邊界距離不可用
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan, False

    # 取得邊界
    pred_surface = binary_surface(pred)
    gt_surface = binary_surface(gt)

    # 兩邊 surface 都沒有點，視為 perfect
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0, 0.0, True

    # 任一側沒有 surface，距離不可用
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return np.nan, np.nan, False

    # 計算到 gt surface 的距離 transform
    dt_gt = distance_transform_edt(~gt_surface)

    # 計算到 pred surface 的距離 transform
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
    # 轉成 uint8
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # TP / TN / FP / FN
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())

    # overlap metrics
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    # pred/gt 都空時，precision/recall 視為 1
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
    # 若沒有資料，回傳空 dict
    if len(rows) == 0:
        return {}

    out = {}

    # 一般 metrics 直接平均
    out["Dice"] = float(np.mean([r["Dice"] for r in rows]))
    out["IoU"] = float(np.mean([r["IoU"] for r in rows]))
    out["Acc"] = float(np.mean([r["Acc"] for r in rows]))
    out["Precision"] = float(np.mean([r["Precision"] for r in rows]))
    out["Recall"] = float(np.mean([r["Recall"] for r in rows]))

    # HD95 / ASSD 略過 nan
    hd_values = [r["HD95"] for r in rows if not np.isnan(r["HD95"])]
    assd_values = [r["ASSD"] for r in rows if not np.isnan(r["ASSD"])]

    out["HD95"] = float(np.mean(hd_values)) if len(hd_values) > 0 else np.nan
    out["ASSD"] = float(np.mean(assd_values)) if len(assd_values) > 0 else np.nan
    out["Valid_Boundary_Cases"] = int(np.sum([r["Valid_Boundary_Case"] for r in rows]))
    out["Num_Images"] = len(rows)

    return out


def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    # logits -> probability
    probs = torch.sigmoid(logits)

    # threshold 成二值 mask
    preds = (probs > threshold).float()

    # flatten
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # TP / FP / FN
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


# ============================================================
# 7) Train / Eval
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    # 切成 train mode
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
        # 取出 image / mask
        images = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        # 清空梯度
        optimizer.zero_grad(set_to_none=True)

        # AMP forward
        with autocast(enabled=AMP_ENABLED):
            logits = model(images)
            loss, batch_bce, batch_dice_loss = criterion(logits, masks)

        # backward + update
        if AMP_ENABLED:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # batch size
        batch_size = images.size(0)

        # 計算 batch metrics
        metrics = batch_metrics_from_logits(logits, masks, THRESHOLD, EPS)

        # 累加
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
    # 切成 eval mode
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
            # 取出 image / mask
            images = batch["image"].to(DEVICE, non_blocking=True)
            masks = batch["mask"].to(DEVICE, non_blocking=True)

            # AMP forward
            with autocast(enabled=AMP_ENABLED):
                logits = model(images)
                loss, batch_bce, batch_dice_loss = criterion(logits, masks)

            # batch size
            batch_size = images.size(0)

            # 計算 batch metrics
            metrics = batch_metrics_from_logits(logits, masks, THRESHOLD, EPS)

            # 累加
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
    # 切成 eval mode
    model.eval()

    # 建立輸出資料夾
    image_dir = os.path.join(save_dir, "image")
    gt_dir = os.path.join(save_dir, "mask_gt")
    pred_dir = os.path.join(save_dir, "mask_pred")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            # 取出資料
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            stems = batch["stem"]
            image_paths = batch["image_path"]

            # forward
            with autocast(enabled=AMP_ENABLED):
                logits = model(images)

            # logits -> probability -> binary mask
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # 搬回 CPU
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
    history,
    epochs_no_improve
):
    # 儲存 checkpoint
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
        },
        checkpoint_path
    )


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    # 載入 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    # 載入 model 權重
    model.load_state_dict(ckpt["model_state_dict"])

    # 載入 optimizer 狀態
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # 載入 scheduler 狀態
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return (
        ckpt["epoch"] + 1,
        ckpt["best_val_dice"],
        ckpt["best_val_iou"],
        ckpt["best_epoch"],
        ckpt["history"],
        ckpt["epochs_no_improve"]
    )


# ============================================================
# 10) Per-image Evaluation on Test Set
# ============================================================

def evaluate_test_per_image(model, loader, fold_name):
    # 切成 eval mode
    model.eval()

    # 儲存每張影像的結果
    all_rows = []

    with torch.no_grad():
        for batch in loader:
            # 取出 batch
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
            preds = (probs > THRESHOLD).float()

            # 搬到 CPU
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(images.size(0)):
                # 取單張 pred / gt
                pred_bin = preds_np[i, 0].astype(np.uint8)
                gt_bin = masks_np[i, 0].astype(np.uint8)

                # 計算完整 metrics
                metrics = compute_binary_metrics(pred_bin, gt_bin)

                # 建立 row
                row = {
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

                all_rows.append(row)

    return all_rows


# ============================================================
# 11) Result Rebuild Helpers
# ============================================================

def parse_result_txt(result_txt_path):
    # 儲存 result.txt 讀到的 key-value
    data = {}

    # 讀 result.txt
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

    # 檢查欄位是否存在
    for k in required_keys:
        if k not in data:
            raise KeyError(f"{result_txt_path} 缺少欄位: {k}")

    return {
        "fold": data["fold"],
        "best_epoch": int(data["best_epoch"]),
        "best_val_dice": float(data["best_val_dice"]),
        "best_val_iou": float(data["best_val_iou"]),
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
    # 儲存所有 fold 結果
    rows = []

    for fold_name in RUN_FOLDS:
        result_txt = os.path.join(
            OUT_ROOT,
            fold_name,
            RESULT_DIR_NAME,
            "result.txt"
        )

        if os.path.isfile(result_txt):
            rows.append(parse_result_txt(result_txt))
        else:
            print(f"[WARN] 找不到 result.txt，略過：{result_txt}")

    if len(rows) == 0:
        return pd.DataFrame()

    # 建立 DataFrame
    df = pd.DataFrame(rows)

    # 依 fold 排序
    df = df.sort_values("fold").reset_index(drop=True)

    # 輸出總表
    out_csv = os.path.join(OUT_ROOT, "all_fold_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return df


def rebuild_fold_val_data_metrics(all_fold_df: pd.DataFrame):
    # 若沒有資料，回傳空表
    if all_fold_df.empty:
        return pd.DataFrame()

    # 複製一份
    df = all_fold_df.copy()

    # 對齊你原本表格欄位
    df["Val_Dice"] = df["best_val_dice"]
    df["ΔDice"] = df["delta_shift_dice"]
    df["Val_IoU"] = df["best_val_iou"]
    df["ΔIoU"] = df["delta_shift_iou"]

    # 只保留需要欄位
    out_df = df[["fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"]].copy()

    # 欄位重新命名
    out_df.columns = ["Fold", "Val_Dice", "ΔDice", "Val_IoU", "ΔIoU"]

    # 輸出
    out_csv = os.path.join(OUT_ROOT, "fold_val_data_metrics.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return out_df


def rebuild_per_image_metrics_from_disk():
    # 儲存所有 fold 的 per-image metrics
    all_rows = []

    for fold_name in RUN_FOLDS:
        per_image_csv = os.path.join(
            OUT_ROOT,
            fold_name,
            RESULT_DIR_NAME,
            "per_image_metrics.csv"
        )

        if os.path.isfile(per_image_csv):
            df = pd.read_csv(per_image_csv)
            all_rows.append(df)
        else:
            print(f"[WARN] 找不到 per_image_metrics.csv，略過：{per_image_csv}")

    if len(all_rows) == 0:
        return pd.DataFrame()

    # 合併所有 fold
    all_df = pd.concat(all_rows, axis=0, ignore_index=True)

    # 輸出總 per-image metrics
    out_csv = os.path.join(OUT_ROOT, "per_image_metrics.csv")
    all_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return all_df


def rebuild_center_level_metrics(per_image_df: pd.DataFrame):
    # 若沒有資料，回傳空表
    if per_image_df.empty:
        return pd.DataFrame()

    # 儲存每個 center 結果
    rows = []

    for center in sorted(per_image_df["Center"].unique()):
        # 取該 center
        sub = per_image_df[per_image_df["Center"] == center]

        # 轉成 aggregate_metrics 需要的格式
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

        # 聚合
        agg = aggregate_metrics(metric_rows)

        # 建立 center row
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

    # 建立 DataFrame
    center_df = pd.DataFrame(rows).sort_values("Center").reset_index(drop=True)

    # 輸出
    out_csv = os.path.join(OUT_ROOT, "center_level_metrics.csv")
    center_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return center_df


def write_loco_summary(all_fold_df: pd.DataFrame, center_df: pd.DataFrame):
    # summary path
    summary_path = os.path.join(OUT_ROOT, "loco_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 3-B: SegFormer-B0 + LOCO (Student Baseline)\n")
        f.write("=" * 70 + "\n\n")

        if not all_fold_df.empty:
            # LOCO mean
            mean_test_dice = all_fold_df["test_dice"].mean()
            mean_test_iou = all_fold_df["test_iou"].mean()
            mean_delta_dice = all_fold_df["delta_shift_dice"].mean()
            mean_delta_iou = all_fold_df["delta_shift_iou"].mean()

            f.write(f"LOCO mean test Dice: {mean_test_dice:.6f}\n")
            f.write(f"LOCO mean test IoU : {mean_test_iou:.6f}\n")
            f.write(f"Mean ΔDice         : {mean_delta_dice:.6f}\n")
            f.write(f"Mean ΔIoU          : {mean_delta_iou:.6f}\n\n")

        if not center_df.empty:
            # best / worst center
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

    # fold dir
    fold_dir = os.path.join(LOCO_ROOT, fold_name)

    # result root
    result_root = os.path.join(OUT_ROOT, fold_name, RESULT_DIR_NAME)

    # csv paths
    train_csv = os.path.join(fold_dir, "train.csv")
    val_csv = os.path.join(fold_dir, "val.csv")
    test_csv = os.path.join(fold_dir, "test.csv")

    # output paths
    history_csv_path = os.path.join(result_root, "history.csv")
    checkpoint_path = os.path.join(result_root, "best_model.pth")
    result_txt_path = os.path.join(result_root, "result.txt")
    per_image_csv_path = os.path.join(result_root, "per_image_metrics.csv")
    test_pred_dir = os.path.join(result_root, "test_predictions")

    # 若指定 reset，此 fold 先刪掉
    if fold_name in RESET_FOLDS and os.path.isdir(os.path.join(OUT_ROOT, fold_name)):
        print(f"[RESET] 刪除舊結果：{os.path.join(OUT_ROOT, fold_name)}")
        import shutil
        shutil.rmtree(os.path.join(OUT_ROOT, fold_name), ignore_errors=True)

    # 若已完成且選擇 skip
    if SKIP_COMPLETED_FOLDS and os.path.isfile(result_txt_path):
        print(f"[SKIP] 已完成：{fold_name}")
        return

    # 建立結果資料夾
    os.makedirs(result_root, exist_ok=True)

    # 檢查 csv 是否存在
    for p in [train_csv, val_csv, test_csv]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 CSV：{p}")

    # 建立 dataset
    train_dataset = PolypDataset(train_csv, image_size=IMAGE_SIZE, is_train=True)
    val_dataset = PolypDataset(val_csv, image_size=IMAGE_SIZE, is_train=False)
    test_dataset = PolypDataset(test_csv, image_size=IMAGE_SIZE, is_train=False)

    print(f"[DATA] train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    # train loader
    # 保持與 DDRNet baseline 一樣 drop_last=True
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )

    # val loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    # test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    # 建立 SegFormer-B0 模型
    model = SegFormerB0BinarySeg(
        pretrained_name=PRETRAINED_NAME,
        load_pretrained=LOAD_PRETRAINED
    ).to(DEVICE)

    # loss
    criterion = DiceBCELoss()

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
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

    # training state
    start_epoch = 0
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    # 若 checkpoint 存在，則續跑
    if os.path.isfile(checkpoint_path):
        print(f"[RESUME] 載入 checkpoint：{checkpoint_path}")
        (
            start_epoch,
            best_val_dice,
            best_val_iou,
            best_epoch,
            history,
            epochs_no_improve
        ) = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    # 計時
    t0 = time.time()

    # train loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        # 訓練一個 epoch
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler)

        # 驗證一個 epoch
        val_metrics = eval_one_epoch(model, val_loader, criterion)

        # scheduler 根據 val dice 調整 LR
        scheduler.step(val_metrics["dice"])

        # 取得目前 LR
        current_lr = optimizer.param_groups[0]["lr"]

        # 紀錄 history
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

        # 印出 log
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

        # 判斷是否刷新 best
        improved = val_metrics["dice"] > best_val_dice

        if improved:
            best_val_dice = val_metrics["dice"]
            best_val_iou = val_metrics["iou"]
            best_epoch = epoch + 1
            epochs_no_improve = 0

            # 儲存 best checkpoint
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

            print(
                f"[BEST] {fold_name} "
                f"best_val_dice={best_val_dice:.4f} "
                f"best_val_iou={best_val_iou:.4f} "
                f"@ epoch {best_epoch}"
            )

        else:
            epochs_no_improve += 1

        # 每個 epoch 都寫 history.csv
        pd.DataFrame(history).to_csv(history_csv_path, index=False, encoding="utf-8-sig")

        # early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"[EARLY STOP] {fold_name} at epoch {epoch + 1}")
            break

    # 載入最佳權重
    if os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

    # test evaluation
    test_metrics = eval_one_epoch(model, test_loader, criterion)

    # 計算 Δshift
    delta_shift_dice = best_val_dice - test_metrics["dice"]
    delta_shift_iou = best_val_iou - test_metrics["iou"]

    # 每張圖詳細評估
    per_image_rows = evaluate_test_per_image(model, test_loader, fold_name)
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(per_image_csv_path, index=False, encoding="utf-8-sig")

    # 儲存 test prediction
    if SAVE_TEST_PRED:
        save_test_predictions(
            model=model,
            loader=test_loader,
            save_dir=test_pred_dir,
            threshold=THRESHOLD
        )

    # 計算總時間
    total_time = time.time() - t0

    # 寫 result.txt
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(f"fold: {fold_name}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_dice: {best_val_dice:.10f}\n")
        f.write(f"best_val_iou: {best_val_iou:.10f}\n")
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

    # 清理記憶體
    del model
    del optimizer
    del scheduler
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
# 13) Main
# ============================================================

def main():
    # 固定 seed
    set_seed(SEED)

    print("=" * 100)
    print("Stage 3-B: SegFormer-B0 + LOCO (Student Baseline)")
    print("=" * 100)
    print("DEVICE =", DEVICE)

    if torch.cuda.is_available():
        print("GPU =", torch.cuda.get_device_name(0))

    print("LOCO_ROOT =", LOCO_ROOT)
    print("OUT_ROOT =", OUT_ROOT)
    print("RUN_FOLDS =", RUN_FOLDS)
    print("PRETRAINED_NAME =", PRETRAINED_NAME)
    print("LOAD_PRETRAINED =", LOAD_PRETRAINED)
    print("IMAGE_SIZE =", IMAGE_SIZE)
    print("BATCH_SIZE =", BATCH_SIZE)
    print("NUM_EPOCHS =", NUM_EPOCHS)
    print("LR =", LR)
    print("WEIGHT_DECAY =", WEIGHT_DECAY)
    print("EARLY_STOPPING_PATIENCE =", EARLY_STOPPING_PATIENCE)
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

    # 印 center-level 結果
    if not center_df.empty:
        print("\n[Center-level Metrics]")
        print(center_df.to_string(index=False))

    # 印 fold val / delta shift 結果
    if not fold_val_df.empty:
        print("\n[Fold Val Data Metrics]")
        print(fold_val_df.to_string(index=False))


if __name__ == "__main__":
    main()