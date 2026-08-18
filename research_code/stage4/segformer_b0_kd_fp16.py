# ============================================================
# Stage 4
# SegFormer-B0 + LOCO + KD FP16 LOCO Re-evaluation
# ------------------------------------------------------------
# 功能：
# 1. 不重新訓練
# 2. 載入 Stage3E SegFormer-B0+KD 的 6-fold best_model.pth
# 3. 使用 FP16 推論
# 4. 重新跑 C1~C6 test set
# 5. 輸出：
#    - per_image_metrics_all.csv
#    - center_level_metrics.csv
#    - fold_val_data_metrics.csv
#    - all_fold_results.csv
#    - loco_summary.txt
# ============================================================


# 匯入 os，用來處理路徑、資料夾、檔案是否存在。
import os

# 匯入 csv，用來讀取 LOCO split CSV。
import csv

# 匯入 time，用來記錄整體評估花費時間。
import time

# 匯入 warnings，用來關閉非必要 warning。
import warnings

# 從 pathlib 匯入 Path，讓 Windows 路徑處理更穩定。
from pathlib import Path

# 從 typing 匯入 Dict 和 Tuple，讓函式回傳型態更清楚。
from typing import Dict, Tuple

# 關閉 warning，避免終端機被 Hugging Face / PIL warning 洗版。
warnings.filterwarnings("ignore")

# 匯入 numpy，用來計算 metric、平均值、array 操作。
import numpy as np

# 匯入 pandas，用來輸出與彙整 CSV 表格。
import pandas as pd

# 從 PIL 匯入 Image，用來讀取與儲存影像。
from PIL import Image

# 從 scipy 匯入 binary_erosion，用來計算 mask boundary。
from scipy.ndimage import binary_erosion

# 從 scipy 匯入 distance_transform_edt，用來計算 HD95 與 ASSD。
from scipy.ndimage import distance_transform_edt

# 匯入 torch，使用 PyTorch 載入模型與 GPU 推論。
import torch

# 匯入 torch.nn，建立模型 wrapper。
import torch.nn as nn

# 匯入 torch.nn.functional，使用 interpolate resize logits。
import torch.nn.functional as F

# 匯入 Dataset / DataLoader，用來建立 test loader。
from torch.utils.data import Dataset, DataLoader

# 匯入 SegFormer config 與模型。
from transformers import SegformerConfig, SegformerForSemanticSegmentation


# ============================================================
# 0. 路徑與主要設定
# ============================================================

# 設定 PolypGen 專案根目錄。
BASE_DIR = Path(os.environ.get("POLYPDG_DATA_ROOT", "./data"))

# 設定 LOCO split 根目錄，沿用 Stage3E 訓練時的 clean LOCO split。
LOCO_ROOT = BASE_DIR / "loco_1537_clean"

# 設定 Stage3E SegFormer-B0+KD 已訓練完成的權重根目錄。
STAGE3E_ROOT = BASE_DIR / "stage3E_segformer_b0_loco_kd_20260428_173054"

# 設定 Stage4 輸出根目錄。
STAGE4_ROOT = BASE_DIR / "stage4_deployment_benchmark"

# 設定 FP16 LOCO re-evaluation 的輸出資料夾。
OUT_ROOT = STAGE4_ROOT / "fp16_loco_results"

# 設定每個 fold 的輸出資料夾。
FOLD_OUT_ROOT = OUT_ROOT / "fold_results"

# 建立輸出資料夾。
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 建立每個 fold 的輸出根資料夾。
FOLD_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 設定要跑的六個 LOCO fold。
RUN_FOLDS = [
    "fold_test_C1",
    "fold_test_C2",
    "fold_test_C3",
    "fold_test_C4",
    "fold_test_C5",
    "fold_test_C6",
]

# 設定影像尺寸，必須與 Stage3E 保持一致。
IMAGE_SIZE = (352, 352)

# 設定 batch size；FP16 evaluation 不需要 teacher，所以 8 通常可行，若 OOM 改成 4。
BATCH_SIZE = 8

# 設定 DataLoader workers；Windows 下用 0 最穩。
NUM_WORKERS = 0

# 設定 pin memory；GPU 推論時可開啟。
PIN_MEMORY = True

# 設定 segmentation threshold，與 Stage3E 一致。
THRESHOLD = 0.5

# 設定小常數，避免除以 0。
EPS = 1e-7

# 設定 device。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 設定是否使用 FP16。
USE_FP16 = DEVICE == "cuda"

# 設定是否儲存 prediction mask。
SAVE_TEST_PRED = False


# ============================================================
# 1. 基本工具：找 CSV
# ============================================================

def find_split_csv(fold_dir: Path, split_name: str) -> str:
    # 先列出常見 CSV 位置。
    candidates = [
        fold_dir / f"{split_name}.csv",
        fold_dir / f"{split_name}_data.csv",
        fold_dir / f"{split_name}_split.csv",
        fold_dir / "csv" / f"{split_name}.csv",
        fold_dir / "splits" / f"{split_name}.csv",
    ]

    # 逐一檢查常見位置是否存在。
    for path in candidates:
        if path.is_file():
            return str(path)

    # 如果常見位置找不到，就遞迴搜尋 fold_dir。
    for root, _, files in os.walk(fold_dir):
        for file_name in files:
            lower = file_name.lower()

            # 只找副檔名是 .csv 且檔名包含 split_name 的檔案。
            if lower.endswith(".csv") and split_name in lower:
                return os.path.join(root, file_name)

    # 如果還是找不到，就丟錯誤。
    raise FileNotFoundError(f"找不到 {split_name}.csv，fold_dir={fold_dir}")


# ============================================================
# 2. 前處理：對齊 Stage3E
# ============================================================

def resize_pair(img, mask, image_size):
    # image 使用 bilinear resize。
    img = img.resize(image_size, Image.BILINEAR)

    # mask 使用 nearest resize，避免 label 被插值污染。
    mask = mask.resize(image_size, Image.NEAREST)

    # 回傳 resize 後的 image 與 mask。
    return img, mask


def image_to_chw_float(img):
    # PIL RGB 轉成 numpy float32，並正規化到 [0,1]。
    img_np = np.array(img, dtype=np.float32) / 255.0

    # 將 HWC 轉成 CHW，符合 PyTorch input 格式。
    img_np = np.transpose(img_np, (2, 0, 1))

    # 回傳 CHW image。
    return img_np


def mask_to_1hw_float(mask):
    # PIL mask 轉成 numpy uint8。
    mask_np = np.array(mask, dtype=np.uint8)

    # 以 127 作為 threshold 轉成 binary mask。
    mask_np = (mask_np > 127).astype(np.float32)

    # 將 HxW 轉成 1xHxW。
    mask_np = np.expand_dims(mask_np, axis=0)

    # 回傳 1HW mask。
    return mask_np


def preprocess_single_view(img, mask, image_size=(352, 352)):
    # test / eval 不做 augmentation，只做 resize。
    img, mask = resize_pair(img, mask, image_size)

    # image 轉成 CHW float。
    img_np = image_to_chw_float(img)

    # mask 轉成 1HW binary float。
    mask_np = mask_to_1hw_float(mask)

    # 回傳 image 與 mask。
    return img_np, mask_np


# ============================================================
# 3. Dataset
# ============================================================

class PolypDataset(Dataset):
    # 初始化 dataset。
    def __init__(self, csv_file, image_size=(352, 352)):
        # 建立 rows list。
        self.rows = []

        # 儲存 image size。
        self.image_size = image_size

        # 讀取 CSV。
        with open(csv_file, "r", encoding="utf-8-sig") as f:
            # 建立 DictReader。
            reader = csv.DictReader(f)

            # 將每一列加入 rows。
            for row in reader:
                self.rows.append(row)

    # 回傳資料數量。
    def __len__(self):
        return len(self.rows)

    # 取得單筆資料。
    def __getitem__(self, idx):
        # 取出第 idx 筆 row。
        row = self.rows[idx]

        # 讀取 RGB image。
        img = Image.open(row["image_path"]).convert("RGB")

        # 讀取 gray-scale mask。
        mask = Image.open(row["mask_path"]).convert("L")

        # 執行前處理。
        img_np, mask_np = preprocess_single_view(
            img=img,
            mask=mask,
            image_size=self.image_size,
        )

        # 若 CSV 有 stem 欄位就使用，沒有就從 image_path 檔名取 stem。
        stem = row.get("stem", os.path.splitext(os.path.basename(row["image_path"]))[0])

        # 若 CSV 有 center 欄位就使用，沒有就 UNKNOWN。
        center = row["center"] if "center" in row else "UNKNOWN"

        # 回傳 PyTorch tensor 與 metadata。
        return {
            "image": torch.tensor(img_np, dtype=torch.float32),
            "mask": torch.tensor(mask_np, dtype=torch.float32),
            "stem": stem,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "center": center,
        }


# ============================================================
# 4. SegFormer-B0 Student 模型
# ============================================================

class SegFormerB0BinaryStudent(nn.Module):
    # 初始化 SegFormer-B0 binary segmentation model。
    def __init__(self):
        # 初始化 nn.Module。
        super().__init__()

        # 建立 SegFormer-B0 config，避免重新下載 pretrained。
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

        # 由 config 建立 SegFormerForSemanticSegmentation。
        self.model = SegformerForSemanticSegmentation(config)

    # forward。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 取得輸入高度與寬度。
        input_h, input_w = x.shape[-2], x.shape[-1]

        # Hugging Face SegFormer forward。
        outputs = self.model(pixel_values=x)

        # 取得 logits。
        logits = outputs.logits

        # 如果 logits 尺寸不是輸入尺寸，就 resize 回輸入尺寸。
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits,
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
            )

        # 回傳 logits。
        return logits


# ============================================================
# 5. Checkpoint loading
# ============================================================

def extract_state_dict_from_checkpoint(checkpoint):
    # 如果 checkpoint 是 dict 且有 model_state_dict，就取 model_state_dict。
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    # 如果 checkpoint 是 dict 且有 state_dict，就取 state_dict。
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    # 如果 checkpoint 是 dict 且有 student_state_dict，就取 student_state_dict。
    if isinstance(checkpoint, dict) and "student_state_dict" in checkpoint:
        return checkpoint["student_state_dict"]

    # 其他情況假設 checkpoint 本身就是 state_dict。
    return checkpoint


def load_student_checkpoint(model, ckpt_path: Path):
    # 檢查 checkpoint 是否存在。
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    # 讀取 checkpoint。
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # 取出真正的 state_dict。
    state_dict = extract_state_dict_from_checkpoint(ckpt)

    # 清理 DataParallel 的 module. prefix。
    clean_state = {}

    # 逐一處理 key。
    for key, value in state_dict.items():
        # 複製原始 key。
        new_key = key

        # 若 key 以 module. 開頭，移除 module.。
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # 存入 clean_state。
        clean_state[new_key] = value

    # 嘗試 strict=True 載入。
    try:
        model.load_state_dict(clean_state, strict=True)

        # 回傳載入狀態。
        return 0, 0

    # 如果 strict=True 失敗，就改用 strict=False 並回報 mismatch。
    except RuntimeError:
        load_result = model.load_state_dict(clean_state, strict=False)

        # 取得 missing keys 數量。
        missing_count = len(load_result.missing_keys)

        # 取得 unexpected keys 數量。
        unexpected_count = len(load_result.unexpected_keys)

        # 回傳 mismatch 數量。
        return missing_count, unexpected_count


def get_best_val_metrics_from_checkpoint_or_result(fold_name: str, ckpt_path: Path):
    # 預設 best val 為 nan。
    best_val_dice = np.nan

    # 預設 best val iou 為 nan。
    best_val_iou = np.nan

    # 先嘗試從 checkpoint 讀 best_val_dice / best_val_iou。
    try:
        # 讀取 checkpoint。
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

        # 如果 checkpoint 有 best_val_dice，就讀取。
        if isinstance(ckpt, dict) and "best_val_dice" in ckpt:
            best_val_dice = float(ckpt["best_val_dice"])

        # 如果 checkpoint 有 best_val_iou，就讀取。
        if isinstance(ckpt, dict) and "best_val_iou" in ckpt:
            best_val_iou = float(ckpt["best_val_iou"])

    # 若讀取失敗就略過。
    except Exception:
        pass

    # 如果 checkpoint 沒有 val metrics，就嘗試讀 Stage3E result.txt。
    if np.isnan(best_val_dice) or np.isnan(best_val_iou):
        # 建立 result.txt 路徑。
        result_txt = STAGE3E_ROOT / fold_name / "stage3E_results" / "result.txt"

        # 如果 result.txt 存在，就解析。
        if result_txt.is_file():
            # 開啟 result.txt。
            with open(result_txt, "r", encoding="utf-8") as f:
                # 逐行讀取。
                for line in f:
                    # 如果該行是 best_val_dice。
                    if line.startswith("best_val_dice:"):
                        best_val_dice = float(line.strip().split(":")[-1])

                    # 如果該行是 best_val_iou。
                    if line.startswith("best_val_iou:"):
                        best_val_iou = float(line.strip().split(":")[-1])

    # 回傳 best val dice / iou。
    return best_val_dice, best_val_iou


# ============================================================
# 6. Metrics
# ============================================================

def safe_div(n: float, d: float) -> float:
    # 如果分母大於 0，正常除法；否則回傳 0。
    return float(n) / float(d) if d > 0 else 0.0


def binary_surface(mask: np.ndarray) -> np.ndarray:
    # 如果 mask 沒有前景，回傳全 False。
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    # 對 mask 做 erosion。
    eroded = binary_erosion(mask)

    # surface = mask XOR eroded。
    return mask.astype(bool) ^ eroded.astype(bool)


def hd95_and_assd(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float, bool]:
    # 將 pred 轉成 bool。
    pred = pred.astype(bool)

    # 將 gt 轉成 bool。
    gt = gt.astype(bool)

    # 若 pred 和 gt 都空，視為 perfect。
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0, 0.0, True

    # 若其中一個為空，boundary distance 不可用。
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan, np.nan, False

    # 計算 pred surface。
    pred_surface = binary_surface(pred)

    # 計算 gt surface。
    gt_surface = binary_surface(gt)

    # 若兩個 surface 都空，視為 perfect。
    if pred_surface.sum() == 0 and gt_surface.sum() == 0:
        return 0.0, 0.0, True

    # 若其中一個 surface 為空，boundary distance 不可用。
    if pred_surface.sum() == 0 or gt_surface.sum() == 0:
        return np.nan, np.nan, False

    # 計算 gt surface 的 distance transform。
    dt_gt = distance_transform_edt(~gt_surface)

    # 計算 pred surface 的 distance transform。
    dt_pred = distance_transform_edt(~pred_surface)

    # pred surface 到 gt surface 的距離。
    dist_pred_to_gt = dt_gt[pred_surface]

    # gt surface 到 pred surface 的距離。
    dist_gt_to_pred = dt_pred[gt_surface]

    # 如果距離陣列為空，回傳 nan。
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan, np.nan, False

    # 合併雙向距離。
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred], axis=0)

    # HD95 = 95 percentile distance。
    hd95 = np.percentile(all_dists, 95)

    # ASSD = average symmetric surface distance。
    assd = np.mean(all_dists)

    # 回傳 HD95、ASSD、valid flag。
    return float(hd95), float(assd), True


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    # 將 pred 轉成 uint8。
    pred = pred.astype(np.uint8)

    # 將 gt 轉成 uint8。
    gt = gt.astype(np.uint8)

    # 計算 pred 前景像素數。
    pred_sum = int(pred.sum())

    # 計算 gt 前景像素數。
    gt_sum = int(gt.sum())

    # 如果 pred 和 gt 都空，所有 metric 視為 perfect。
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

    # True positive。
    tp = int(((pred == 1) & (gt == 1)).sum())

    # True negative。
    tn = int(((pred == 0) & (gt == 0)).sum())

    # False positive。
    fp = int(((pred == 1) & (gt == 0)).sum())

    # False negative。
    fn = int(((pred == 0) & (gt == 1)).sum())

    # 計算 Dice。
    dice = safe_div(2 * tp, 2 * tp + fp + fn)

    # 計算 IoU。
    iou = safe_div(tp, tp + fp + fn)

    # 計算 Accuracy。
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    # 計算 Precision。
    precision = safe_div(tp, tp + fp)

    # 計算 Recall。
    recall = safe_div(tp, tp + fn)

    # 計算 HD95 / ASSD。
    hd95, assd, valid_boundary = hd95_and_assd(pred, gt)

    # 回傳所有 metric。
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


def aggregate_per_image_metrics(per_image_df: pd.DataFrame) -> Dict[str, float]:
    # 回傳 per-image 平均結果。
    return {
        "dice": float(per_image_df["Dice"].mean()),
        "iou": float(per_image_df["IoU"].mean()),
        "acc": float(per_image_df["Acc"].mean()),
        "precision": float(per_image_df["Precision"].mean()),
        "recall": float(per_image_df["Recall"].mean()),
        "hd95": float(per_image_df["HD95"].mean(skipna=True)),
        "assd": float(per_image_df["ASSD"].mean(skipna=True)),
    }


def summarize_center_level(per_image_df: pd.DataFrame) -> pd.DataFrame:
    # 建立 center-level 統計 rows。
    rows = []

    # 依 Center 分組。
    for center, group in per_image_df.groupby("Center"):
        # 建立單一 center row。
        row = {
            "Center": center,
            "Num_images": int(len(group)),
            "Dice": float(group["Dice"].mean()),
            "IoU": float(group["IoU"].mean()),
            "Acc": float(group["Acc"].mean()),
            "Precision": float(group["Precision"].mean()),
            "Recall": float(group["Recall"].mean()),
            "HD95": float(group["HD95"].mean(skipna=True)),
            "ASSD": float(group["ASSD"].mean(skipna=True)),
            "Valid_Boundary_Cases": int(group["Valid_Boundary_Case"].sum()),
        }

        # 加入 rows。
        rows.append(row)

    # 轉成 DataFrame。
    df = pd.DataFrame(rows)

    # 若有資料，依 Center 排序。
    if len(df) > 0:
        df = df.sort_values("Center").reset_index(drop=True)

    # 回傳 center-level DataFrame。
    return df


# ============================================================
# 7. 儲存 prediction
# ============================================================

def save_prediction_images(batch, preds_np, pred_dir: Path):
    # 建立 image / mask_gt / mask_pred 資料夾。
    image_dir = pred_dir / "image"
    gt_dir = pred_dir / "mask_gt"
    pred_mask_dir = pred_dir / "mask_pred"

    # 建立資料夾。
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    # 取出 batch image tensor。
    images = batch["image"].cpu().numpy()

    # 取出 batch mask tensor。
    masks = batch["mask"].cpu().numpy()

    # 逐張儲存。
    for i, stem in enumerate(batch["stem"]):
        # image CHW -> HWC。
        img_np = np.transpose(images[i], (1, 2, 0))

        # image [0,1] -> [0,255]。
        img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        # gt mask 1HW -> HW。
        gt_uint8 = (masks[i, 0] > 0.5).astype(np.uint8) * 255

        # pred mask HW -> uint8。
        pred_uint8 = preds_np[i].astype(np.uint8) * 255

        # 儲存 image。
        Image.fromarray(img_uint8).save(image_dir / f"{stem}.png")

        # 儲存 gt。
        Image.fromarray(gt_uint8).save(gt_dir / f"{stem}.png")

        # 儲存 pred。
        Image.fromarray(pred_uint8).save(pred_mask_dir / f"{stem}.png")


# ============================================================
# 8. 單一 fold FP16 evaluation
# ============================================================

def evaluate_single_fold_fp16(fold_name: str):
    # 印出目前 fold。
    print("=" * 100)
    print(f"[FP16 LOCO EVAL] {fold_name}")
    print("=" * 100)

    # 建立 fold split 目錄。
    fold_dir = LOCO_ROOT / fold_name

    # 建立 checkpoint 路徑。
    ckpt_path = STAGE3E_ROOT / fold_name / "best_model.pth"

    # 建立 fold 輸出資料夾。
    fold_out_dir = FOLD_OUT_ROOT / fold_name

    # 建立 prediction 輸出資料夾。
    pred_dir = fold_out_dir / "test_predictions"

    # 建立 fold output。
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    # 找 test csv。
    test_csv = find_split_csv(fold_dir, "test")

    # 建立 test dataset。
    test_dataset = PolypDataset(test_csv, IMAGE_SIZE)

    # 建立 test loader。
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY if DEVICE == "cuda" else False,
        drop_last=False,
    )

    # 印出測試資料數。
    print(f"[INFO] test_csv = {test_csv}")
    print(f"[INFO] num_test_images = {len(test_dataset)}")
    print(f"[INFO] checkpoint = {ckpt_path}")

    # 建立 student model。
    model = SegFormerB0BinaryStudent()

    # 載入 checkpoint。
    missing_count, unexpected_count = load_student_checkpoint(model, ckpt_path)

    # 印出載入狀態。
    print(f"[INFO] missing_keys_count = {missing_count}")
    print(f"[INFO] unexpected_keys_count = {unexpected_count}")

    # 將模型移到 device。
    model = model.to(DEVICE)

    # 切換 eval mode。
    model.eval()

    # 如果 CUDA 可用，轉成 FP16。
    if USE_FP16:
        model.half()
        print("[INFO] model converted to FP16")

    # 建立 per-image rows。
    per_image_rows = []

    # 評估開始時間。
    start_time = time.time()

    # 不計算 gradient。
    with torch.no_grad():
        # 逐 batch 評估。
        for batch_idx, batch in enumerate(test_loader):
            # 將 image 放到 device。
            images = batch["image"].to(DEVICE, non_blocking=True)

            # 如果使用 FP16，把 input 轉成 half。
            if USE_FP16:
                images = images.half()

            # 模型 forward。
            logits = model(images)

            # sigmoid 轉成 probability。
            probs = torch.sigmoid(logits)

            # probability > threshold 轉成 binary prediction。
            preds = (probs > THRESHOLD).float()

            # prediction 轉 numpy。
            preds_np = preds[:, 0].detach().cpu().numpy().astype(np.uint8)

            # gt mask 轉 numpy。
            masks_np = batch["mask"][:, 0].cpu().numpy().astype(np.uint8)

            # 如果要儲存 prediction。
            if SAVE_TEST_PRED:
                save_prediction_images(batch, preds_np, pred_dir)

            # 逐張計算 metric。
            for i in range(preds_np.shape[0]):
                # 取得 pred。
                pred_i = preds_np[i]

                # 取得 gt。
                gt_i = masks_np[i]

                # 計算 metric。
                metrics = compute_binary_metrics(pred_i, gt_i)

                # 建立 row。
                row = {
                    "Fold": fold_name.replace("fold_test_", ""),
                    "Center": batch["center"][i],
                    "Stem": batch["stem"][i],
                    "Image_Path": batch["image_path"][i],
                    "Mask_Path": batch["mask_path"][i],
                    "Pred_Foreground_Pixels": int(pred_i.sum()),
                    "GT_Foreground_Pixels": int(gt_i.sum()),
                    "Dice": metrics["Dice"],
                    "IoU": metrics["IoU"],
                    "Acc": metrics["Acc"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "HD95": metrics["HD95"],
                    "ASSD": metrics["ASSD"],
                    "Valid_Boundary_Case": metrics["Valid_Boundary_Case"],
                }

                # 加入 per-image rows。
                per_image_rows.append(row)

            # 每 20 個 batch 印一次進度。
            if (batch_idx + 1) == 1 or (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(test_loader):
             print(f"[INFO] {fold_name}: batch {batch_idx + 1}/{len(test_loader)}")

    # 計算 elapsed time。
    elapsed = time.time() - start_time

    # 建立 per-image DataFrame。
    per_image_df = pd.DataFrame(per_image_rows)

    # 儲存 fold per-image metrics。
    per_image_csv = fold_out_dir / "per_image_metrics.csv"
    per_image_df.to_csv(per_image_csv, index=False, encoding="utf-8-sig")

    # 建立 fold center-level metrics。
    center_df = summarize_center_level(per_image_df)

    # 儲存 fold center-level metrics。
    center_csv = fold_out_dir / "center_level_metrics.csv"
    center_df.to_csv(center_csv, index=False, encoding="utf-8-sig")

    # 聚合 fold test metrics。
    test_metrics = aggregate_per_image_metrics(per_image_df)

    # 取得 best validation metrics。
    best_val_dice, best_val_iou = get_best_val_metrics_from_checkpoint_or_result(fold_name, ckpt_path)

    # 計算 delta shift。
    delta_dice = best_val_dice - test_metrics["dice"]
    delta_iou = best_val_iou - test_metrics["iou"]

    # 建立 fold summary row。
    fold_row = {
        "Fold": fold_name.replace("fold_test_", ""),
        "Val_Dice": best_val_dice,
        "Test_Dice": test_metrics["dice"],
        "Delta_Dice": delta_dice,
        "Val_IoU": best_val_iou,
        "Test_IoU": test_metrics["iou"],
        "Delta_IoU": delta_iou,
        "Test_Acc": test_metrics["acc"],
        "Test_Precision": test_metrics["precision"],
        "Test_Recall": test_metrics["recall"],
        "Test_HD95": test_metrics["hd95"],
        "Test_ASSD": test_metrics["assd"],
        "Missing_Keys": missing_count,
        "Unexpected_Keys": unexpected_count,
        "Time_sec": elapsed,
        "Checkpoint": str(ckpt_path),
    }

    # 儲存 fold result txt。
    result_txt = fold_out_dir / "result.txt"
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("Stage 4 FP16 LOCO Re-evaluation\n")
        f.write("Model: SegFormer-B0 + LOCO + KD\n")
        f.write(f"Fold: {fold_name}\n")
        f.write(f"Val_Dice: {best_val_dice:.6f}\n")
        f.write(f"Test_Dice: {test_metrics['dice']:.6f}\n")
        f.write(f"Delta_Dice: {delta_dice:.6f}\n")
        f.write(f"Val_IoU: {best_val_iou:.6f}\n")
        f.write(f"Test_IoU: {test_metrics['iou']:.6f}\n")
        f.write(f"Delta_IoU: {delta_iou:.6f}\n")
        f.write(f"Test_Acc: {test_metrics['acc']:.6f}\n")
        f.write(f"Test_Precision: {test_metrics['precision']:.6f}\n")
        f.write(f"Test_Recall: {test_metrics['recall']:.6f}\n")
        f.write(f"Test_HD95: {test_metrics['hd95']:.6f}\n")
        f.write(f"Test_ASSD: {test_metrics['assd']:.6f}\n")
        f.write(f"Missing_Keys: {missing_count}\n")
        f.write(f"Unexpected_Keys: {unexpected_count}\n")
        f.write(f"Time_sec: {elapsed:.2f}\n")

    # 刪除模型以釋放 GPU 記憶體。
    del model

    # 清空 CUDA cache。
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 回傳 fold row 與 per-image df。
    return fold_row, per_image_df


# ============================================================
# 9. 全域報表
# ============================================================

def build_global_reports(fold_rows, per_image_dfs):
    # 合併所有 fold per-image metrics。
    per_image_all = pd.concat(per_image_dfs, axis=0, ignore_index=True)

    # 輸出 per_image_metrics_all.csv。
    per_image_all_csv = OUT_ROOT / "per_image_metrics_all.csv"
    per_image_all.to_csv(per_image_all_csv, index=False, encoding="utf-8-sig")

    # 建立全域 center-level metrics。
    center_level_df = summarize_center_level(per_image_all)

    # 輸出 center_level_metrics.csv。
    center_level_csv = OUT_ROOT / "center_level_metrics.csv"
    center_level_df.to_csv(center_level_csv, index=False, encoding="utf-8-sig")

    # 建立 all_fold_results.csv。
    all_fold_df = pd.DataFrame(fold_rows)

    # 輸出 all_fold_results.csv。
    all_fold_csv = OUT_ROOT / "all_fold_results.csv"
    all_fold_df.to_csv(all_fold_csv, index=False, encoding="utf-8-sig")

    # 建立 fold_val_data_metrics.csv。
    fold_val_df = all_fold_df[
        [
            "Fold",
            "Val_Dice",
            "Delta_Dice",
            "Val_IoU",
            "Delta_IoU",
        ]
    ].copy()

    # 輸出 fold_val_data_metrics.csv。
    fold_val_csv = OUT_ROOT / "fold_val_data_metrics.csv"
    fold_val_df.to_csv(fold_val_csv, index=False, encoding="utf-8-sig")

    # 計算 summary。
    mean_dice = float(center_level_df["Dice"].mean())
    mean_iou = float(center_level_df["IoU"].mean())
    mean_acc = float(center_level_df["Acc"].mean())
    mean_precision = float(center_level_df["Precision"].mean())
    mean_recall = float(center_level_df["Recall"].mean())
    mean_hd95 = float(center_level_df["HD95"].mean(skipna=True))
    mean_assd = float(center_level_df["ASSD"].mean(skipna=True))

    # 找 worst center。
    worst_row = center_level_df.loc[center_level_df["Dice"].idxmin()]

    # 找 best center。
    best_row = center_level_df.loc[center_level_df["Dice"].idxmax()]

    # 計算 mean absolute delta。
    mean_abs_delta_dice = float(np.mean(np.abs(all_fold_df["Delta_Dice"].values)))
    mean_abs_delta_iou = float(np.mean(np.abs(all_fold_df["Delta_IoU"].values)))

    # 輸出 loco_summary.txt。
    summary_txt = OUT_ROOT / "loco_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("Stage 4 FP16 LOCO Re-evaluation\n")
        f.write("Model: SegFormer-B0 + LOCO + KD\n")
        f.write("========================================\n")
        f.write(f"mean_dice: {mean_dice:.6f}\n")
        f.write(f"mean_iou: {mean_iou:.6f}\n")
        f.write(f"mean_acc: {mean_acc:.6f}\n")
        f.write(f"mean_precision: {mean_precision:.6f}\n")
        f.write(f"mean_recall: {mean_recall:.6f}\n")
        f.write(f"mean_hd95: {mean_hd95:.6f}\n")
        f.write(f"mean_assd: {mean_assd:.6f}\n")
        f.write(f"worst_center: {worst_row['Center']}\n")
        f.write(f"worst_center_dice: {worst_row['Dice']:.6f}\n")
        f.write(f"best_center: {best_row['Center']}\n")
        f.write(f"best_center_dice: {best_row['Dice']:.6f}\n")
        f.write(f"mean_abs_delta_dice: {mean_abs_delta_dice:.6f}\n")
        f.write(f"mean_abs_delta_iou: {mean_abs_delta_iou:.6f}\n")
        f.write("========================================\n")

    # 印出輸出位置。
    print("=" * 100)
    print("[SAVED GLOBAL REPORTS]")
    print(per_image_all_csv)
    print(center_level_csv)
    print(all_fold_csv)
    print(fold_val_csv)
    print(summary_txt)
    print("=" * 100)

    # 回傳主要報表。
    return center_level_df, all_fold_df, fold_val_df


# ============================================================
# 10. Main
# ============================================================

def main():
    # 印出執行資訊。
    print("=" * 100)
    print("Stage 4 FP16 LOCO Re-evaluation")
    print("=" * 100)
    print("DEVICE =", DEVICE)
    print("USE_FP16 =", USE_FP16)
    print("LOCO_ROOT =", LOCO_ROOT)
    print("STAGE3E_ROOT =", STAGE3E_ROOT)
    print("OUT_ROOT =", OUT_ROOT)
    print("IMAGE_SIZE =", IMAGE_SIZE)
    print("BATCH_SIZE =", BATCH_SIZE)
    print("=" * 100)

    # 建立 fold rows。
    fold_rows = []

    # 建立 per-image dfs。
    per_image_dfs = []

    # 逐一跑六個 fold。
    for fold_name in RUN_FOLDS:
        # 評估單一 fold。
        fold_row, per_image_df = evaluate_single_fold_fp16(fold_name)

        # 加入 fold rows。
        fold_rows.append(fold_row)

        # 加入 per-image dfs。
        per_image_dfs.append(per_image_df)

    # 建立全域報表。
    center_level_df, all_fold_df, fold_val_df = build_global_reports(
        fold_rows,
        per_image_dfs,
    )

    # 印出 center-level 結果。
    print("\n[CENTER LEVEL METRICS]")
    print(center_level_df)

    # 印出 fold delta 結果。
    print("\n[FOLD VAL / DELTA METRICS]")
    print(fold_val_df)

    # 印出完成訊息。
    print("\n[DONE] Stage 4 FP16 LOCO re-evaluation finished.")


# 如果直接執行此檔案，就執行 main。
if __name__ == "__main__":
    main()
