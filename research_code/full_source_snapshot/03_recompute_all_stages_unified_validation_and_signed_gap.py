
from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Paths and global policy
# ============================================================

BASE_DIR = Path(r"./data")
LOCO_ROOT = BASE_DIR / "loco_1537_clean"
JOURNAL_DIR = BASE_DIR / "journal_reevaluation_2026"
SCRIPT_DIR = Path(__file__).resolve().parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = DEVICE.type == "cuda"
BATCH_SIZE = 8
EPS = 1e-7

FOLDS = [f"C{i}" for i in range(1, 7)]

# Unified overlap policy:
# GT empty + pred empty => Dice=1, IoU=1
# only one empty         => Dice=0, IoU=0
# otherwise              => standard Dice/IoU
EMPTY_EMPTY_DICE = 1.0
EMPTY_EMPTY_IOU = 1.0


MODEL_REGISTRY = {
    "Stage1_UNet_LOCO": {
        "root": BASE_DIR / "U-Net+LOCO",
        "source": None,
        "class_name": "AdaptiveUNet",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage2A_SegFormerB2": {
        "root": BASE_DIR / "stage2A_segformer_B2",
        "source": SCRIPT_DIR / "segformer_B2+loco+consistency learning.py",
        "class_name": "SegFormerBinary",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage2B_SegFormerB2_PBLite": {
        "root": BASE_DIR / "stage2B_segformer_b2_pblite_v2_official",
        "source": SCRIPT_DIR / "segformer_B2+loco+consistency learning.py",
        "class_name": "SegFormerBinary",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage2C_SegFormerB2_Consistency": {
        "root": BASE_DIR / "stage2C_segformer_b2_consistency_improved_selfcons_official",
        "source": SCRIPT_DIR / "segformer_B2+loco+consistency learning.py",
        "class_name": "SegFormerBinary",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage2D_SegFormerB2_PB_Consistency": {
        "root": BASE_DIR / "stage2D_segformer_b2_pb_consistency_official",
        "source": SCRIPT_DIR / "SegFormer_B2+LOCO+Perturbation Bank+Consistency Learning.py",
        "class_name": "SegFormerBinary",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage3A_DDRNet23Slim": {
        "root": BASE_DIR / "stage3A_ddrnet23slim_loco_baseline_clean",
        "source": SCRIPT_DIR / "DDRNet23slim+LOCO(1).py",
        "class_name": "DDRNet23SlimBinarySeg",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage3B_SegFormerB0": {
        "root": BASE_DIR / "stage3B_segformer_b0_loco_baseline",
        "source": SCRIPT_DIR / "SegFormer-B0+LOCO(2).py",
        "class_name": "SegFormerB0BinarySeg",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage3C_DDRNet23Slim_KD": {
        "root": BASE_DIR / "stage3C_ddrnet23slim_loco_kd_v4_20260420_203109",
        "source": SCRIPT_DIR / "DDRNET-23-SLIM+LOCO+KD(1).py",
        "class_name": "CustomDDRNet23SlimBinarySeg",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage3D_BiSeNetV2": {
        "root": BASE_DIR / "stage3D_bisenetv2_loco",
        "source": SCRIPT_DIR / "BiSeNetV2+LOCO(3).py",
        "class_name": "BiSeNetV2BinarySeg",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage3D_BiSeNetV2_KD": {
        "root": BASE_DIR / "stage3D_bisenetv2_loco_kd",
        "source": SCRIPT_DIR / "BisSeNetV2+LOCO+KD(3).py",
        "class_name": "BiSeNetV2BinarySeg",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage3E_SegFormerB0_KD": {
        "root": BASE_DIR / "stage3E_segformer_b0_loco_kd_20260428_173054",
        "source": SCRIPT_DIR / "SegForer-B0+LOCO+KD(1).py",
        "class_name": "SegFormerB0BinaryStudent",
        "image_size": (352, 352),
        "precision": "fp32",
    },
    "Stage4_SegFormerB0_KD_FP16": {
        "root": BASE_DIR / "stage3E_segformer_b0_loco_kd_20260428_173054",
        "source": SCRIPT_DIR / "SegForer-B0+LOCO+KD(1).py",
        "class_name": "SegFormerB0BinaryStudent",
        "image_size": (352, 352),
        "precision": "fp16",
    },
}


# ============================================================
# 1. Generic validation dataset
# ============================================================

class ValidationDataset(Dataset):
    def __init__(self, csv_path: Path, image_size: tuple[int, int]):
        self.rows = pd.read_csv(csv_path).to_dict("records")
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = str(row["image_path"])
        mask_path = str(row["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))

        mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        mask_np = np.expand_dims(mask_np, axis=0)

        stem = str(row.get("stem", Path(image_path).stem))
        center = str(row.get("center", "UNKNOWN"))

        return {
            "image": torch.tensor(image_np, dtype=torch.float32),
            "mask": torch.tensor(mask_np, dtype=torch.float32),
            "stem": stem,
            "center": center,
            "image_path": image_path,
            "mask_path": mask_path,
        }


def find_split_csv(fold: str, split_name: str) -> Path:
    fold_dir = LOCO_ROOT / f"fold_test_{fold}"
    candidates = [
        fold_dir / f"{split_name}.csv",
        fold_dir / f"{split_name}_data.csv",
        fold_dir / f"{split_name}_split.csv",
        fold_dir / "csv" / f"{split_name}.csv",
        fold_dir / "splits" / f"{split_name}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path

    matches = sorted(
        path for path in fold_dir.rglob("*.csv")
        if split_name.lower() in path.name.lower()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"{fold}: expected exactly one {split_name} CSV, found {len(matches)}: {matches}"
        )
    return matches[0]


# ============================================================
# 2. Exact Stage 1 U-Net architecture
#    Source: U-Net+LOCO(C1~C6)(1).ipynb
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ExactStage1UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(
            base_ch * 16, base_ch * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(
            base_ch * 8, base_ch * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(
            base_ch * 4, base_ch * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(
            base_ch * 2, base_ch, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))

        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


# ============================================================
# 3. Dynamic source loading and checkpoint loading
# ============================================================

_MODULE_CACHE = {}


def load_source_module(source_path: Path):
    key = str(source_path.resolve())
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    module_name = "polyp_source_" + str(abs(hash(key)))
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source: {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[key] = module
    return module


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict(), checkpoint

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    for key in (
        "model_state_dict",
        "state_dict",
        "student_state_dict",
        "model",
        "net",
    ):
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return value.state_dict(), value
        if isinstance(value, dict):
            return value, None

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint, None

    raise ValueError(f"Cannot locate state_dict. Top-level keys: {list(checkpoint)[:30]}")


def state_dict_variants(state_dict):
    prefixes = ["module.", "model.", "net.", "student.", "student_model."]

    def strip_once(sd, prefix):
        if all(k.startswith(prefix) for k in sd):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    variants = [state_dict]
    current = state_dict
    for _ in range(5):
        changed = False
        for prefix in prefixes:
            candidate = strip_once(current, prefix)
            if candidate is not current:
                variants.append(candidate)
                current = candidate
                changed = True
                break
        if not changed:
            break

    # SegFormer historical wrapper variations.
    extra = []
    for sd in variants:
        extra.append({
            (k[len("segformer."):] if k.startswith("segformer.segformer.") or
             k.startswith("segformer.decode_head.") else k): v
            for k, v in sd.items()
        })
    variants.extend(extra)

    unique = []
    signatures = set()
    for sd in variants:
        signature = tuple(sd.keys())
        if signature not in signatures:
            signatures.add(signature)
            unique.append(sd)
    return unique


def find_checkpoint(model_root: Path, fold: str, model_name: str) -> Path:
    fold_root = model_root / f"fold_test_{fold}"
    if not fold_root.is_dir():
        raise FileNotFoundError(f"Missing fold root: {fold_root}")

    if model_name == "Stage1_UNet_LOCO":
        best_path = fold_root / "best_model.pth"
        if not best_path.is_file():
            raise FileNotFoundError(
                f"{model_root.name}/{fold}: missing required Stage 1 best model: "
                f"{best_path}"
            )
        return best_path

    matches = sorted(fold_root.rglob("best_model.pth"))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: (len(p.parts), str(p).lower()))
        shallowest_depth = len(matches[0].parts)
        shallowest = [p for p in matches if len(p.parts) == shallowest_depth]
        if len(shallowest) == 1:
            return shallowest[0]

    raise RuntimeError(
        f"{model_root.name}/{fold}: expected one unambiguous best_model.pth, found {matches}"
    )

def instantiate_model(model_name: str, config: dict, state_dict):
    if model_name == "Stage1_UNet_LOCO":
        model = ExactStage1UNet(
            in_channels=3,
            out_channels=1,
            base_ch=32,
        )
        load_errors = []
        for variant in state_dict_variants(state_dict):
            try:
                model.load_state_dict(variant, strict=True)
                return model, "ExactStage1UNet(base_ch=32, BatchNorm)"
            except Exception as exc:
                load_errors.append(str(exc).splitlines()[0])

        raise RuntimeError(
            "Strict checkpoint loading failed for the verified Stage 1 U-Net "
            f"architecture. Errors: {load_errors[:6]}"
        )

    source_path = Path(config["source"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing packaged source file: {source_path}")

    module = load_source_module(source_path)
    cls = getattr(module, config["class_name"])

    # Avoid network downloads in historical constructors.
    constructor_attempts = [
        {},
        {"load_pretrained": False},
        {"use_pretrained": False},
    ]

    constructor_errors = []
    model = None
    for kwargs in constructor_attempts:
        try:
            model = cls(**kwargs)
            break
        except TypeError as exc:
            constructor_errors.append(f"{kwargs}: {exc}")

    if model is None:
        raise RuntimeError(
            f"Cannot instantiate {config['class_name']}: {constructor_errors}"
        )

    load_errors = []
    for variant in state_dict_variants(state_dict):
        try:
            model.load_state_dict(variant, strict=True)
            return model, config["class_name"]
        except Exception as exc:
            load_errors.append(str(exc).splitlines()[0])

    raise RuntimeError(
        f"Strict checkpoint loading failed for {model_name}. "
        f"Errors: {load_errors[:6]}"
    )


def load_model(model_name: str, config: dict, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict, serialized_model = extract_state_dict(checkpoint)

    if serialized_model is not None:
        model = serialized_model
        builder = type(model).__name__ + " (serialized module)"
    else:
        model, builder = instantiate_model(model_name, config, state_dict)

    model = model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, builder


# ============================================================
# 4. Threshold discovery
# ============================================================

THRESHOLD_COLUMNS = [
    "Best_Threshold",
    "best_threshold",
    "Threshold",
    "threshold",
    "Val_Best_Threshold",
]


def normalize_fold(value) -> str | None:
    match = re.search(r"C([1-6])", str(value), flags=re.IGNORECASE)
    return f"C{match.group(1)}" if match else None


def find_fold_threshold(model_root: Path, fold: str) -> tuple[float, str]:
    candidate_csvs = sorted({
        *model_root.rglob("all_fold_results.csv"),
        *model_root.rglob("fold_val_data_metrics.csv"),
        *model_root.rglob("center_level_metrics.csv"),
    })

    findings = []
    for csv_path in candidate_csvs:
        try:
            dataframe = pd.read_csv(csv_path)
        except Exception:
            continue

        threshold_col = next(
            (col for col in THRESHOLD_COLUMNS if col in dataframe.columns),
            None,
        )
        if threshold_col is None:
            continue

        fold_col = next(
            (col for col in ("Fold", "fold", "Center", "center") if col in dataframe.columns),
            None,
        )

        if fold_col is None and len(dataframe) == 1:
            value = float(dataframe.iloc[0][threshold_col])
            findings.append((value, str(csv_path)))
            continue

        if fold_col is not None:
            matched = dataframe[
                dataframe[fold_col].map(normalize_fold) == fold
            ]
            for _, row in matched.iterrows():
                findings.append((float(row[threshold_col]), str(csv_path)))

    if not findings:
        return 0.5, "default_0.5"

    rounded_values = {round(value, 8) for value, _ in findings}
    if len(rounded_values) > 1:
        raise RuntimeError(
            f"{model_root.name}/{fold}: conflicting thresholds found: {findings}"
        )

    value = findings[0][0]
    if not 0.0 < value < 1.0:
        raise ValueError(f"Invalid threshold {value} for {model_root.name}/{fold}")
    return value, findings[0][1]


# ============================================================
# 5. Unified Dice and IoU
# ============================================================

def compute_dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return EMPTY_EMPTY_DICE, EMPTY_EMPTY_IOU, "both_empty"

    intersection = int(np.logical_and(pred, gt).sum())
    dice = (2.0 * intersection) / (pred_sum + gt_sum + EPS)
    union = int(np.logical_or(pred, gt).sum())
    iou = intersection / (union + EPS)

    if gt_sum > 0 and pred_sum == 0:
        status = "gt_positive_pred_empty"
    elif gt_sum == 0 and pred_sum > 0:
        status = "gt_empty_pred_positive"
    else:
        status = "both_positive"

    return float(dice), float(iou), status


def extract_logits(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 4:
                return item
    if isinstance(output, dict):
        for key in ("logits", "out", "main", "prediction"):
            item = output.get(key)
            if torch.is_tensor(item):
                return item
    if hasattr(output, "logits") and torch.is_tensor(output.logits):
        return output.logits
    raise TypeError(f"Cannot extract logits from output type {type(output)}")


@torch.inference_mode()
def evaluate_validation_fold(
    model_name: str,
    config: dict,
    fold: str,
    model: nn.Module,
    threshold: float,
):
    val_csv = find_split_csv(fold, "val")
    dataset = ValidationDataset(val_csv, config["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    rows = []
    use_fp16 = config["precision"] == "fp16" and DEVICE.type == "cuda"

    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        gt_batch = batch["mask"].cpu().numpy()[:, 0]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            output = model(images)
            logits = extract_logits(output)

        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probabilities = torch.sigmoid(logits.float())
        pred_batch = (probabilities >= threshold).cpu().numpy()[:, 0]

        for index in range(len(pred_batch)):
            dice, iou, case_status = compute_dice_iou(
                pred_batch[index], gt_batch[index]
            )
            rows.append({
                "Model": model_name,
                "Fold": fold,
                "Stem": batch["stem"][index],
                "Center": batch["center"][index],
                "Threshold": threshold,
                "Dice": dice,
                "IoU": iou,
                "CaseStatus": case_status,
                "GTForegroundPixels": int(gt_batch[index].sum()),
                "PredForegroundPixels": int(pred_batch[index].sum()),
                "ImagePath": batch["image_path"][index],
                "MaskPath": batch["mask_path"][index],
            })

    dataframe = pd.DataFrame(rows)
    expected_count = len(pd.read_csv(val_csv))
    if len(dataframe) != expected_count:
        raise RuntimeError(
            f"{model_name}/{fold}: validation count mismatch "
            f"{len(dataframe)} != {expected_count}"
        )
    return dataframe, val_csv


# ============================================================
# 6. Main
# ============================================================

def print_stage1_checkpoint_inventory():
    stage1_root = Path(MODEL_REGISTRY["Stage1_UNet_LOCO"]["root"])
    print("=" * 110)
    print("[STAGE 1 BEST MODEL INVENTORY]")
    for fold in FOLDS:
        path = stage1_root / f"fold_test_{fold}" / "best_model.pth"
        print(f"{fold}: {'FOUND' if path.is_file() else 'MISSING'} -> {path}")
    print("=" * 110)


def main():
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    print_stage1_checkpoint_inventory()

    test_center_path = JOURNAL_DIR / "center_level_metrics_all.csv"
    if not test_center_path.is_file():
        raise FileNotFoundError(
            f"Missing unified test result file: {test_center_path}"
        )
    test_center = pd.read_csv(test_center_path)

    all_per_image = []
    all_fold_rows = []
    audit_rows = []

    for model_name, config in MODEL_REGISTRY.items():
        model_root = Path(config["root"])

        for fold in FOLDS:
            checkpoint_path = None
            try:
                checkpoint_path = find_checkpoint(model_root, fold, model_name)
                threshold, threshold_source = find_fold_threshold(model_root, fold)
                model, builder = load_model(model_name, config, checkpoint_path)

                val_per_image, val_csv = evaluate_validation_fold(
                    model_name=model_name,
                    config=config,
                    fold=fold,
                    model=model,
                    threshold=threshold,
                )

                val_dice = float(val_per_image["Dice"].mean())
                val_iou = float(val_per_image["IoU"].mean())

                test_row = test_center[
                    (test_center["Model"] == model_name)
                    & (test_center["Center"].map(normalize_fold) == fold)
                ]
                if len(test_row) != 1:
                    raise RuntimeError(
                        f"{model_name}/{fold}: expected one unified test row, "
                        f"found {len(test_row)}"
                    )

                test_dice = float(test_row.iloc[0]["Dice"])
                test_iou = float(test_row.iloc[0]["IoU"])

                all_per_image.append(val_per_image)
                all_fold_rows.append({
                    "Model": model_name,
                    "Fold": fold,
                    "ValidationN": len(val_per_image),
                    "Threshold": threshold,
                    "Val_Dice": val_dice,
                    "Test_Dice": test_dice,
                    "Signed_Delta_Dice": val_dice - test_dice,
                    "Absolute_Delta_Dice": abs(val_dice - test_dice),
                    "Val_IoU": val_iou,
                    "Test_IoU": test_iou,
                    "Signed_Delta_IoU": val_iou - test_iou,
                    "Absolute_Delta_IoU": abs(val_iou - test_iou),
                    "DeltaDefinition": "Unified validation minus unified unseen-center test",
                    "ValidationPolicy": "both-empty=1; one-empty=0; per-image mean",
                    "TestPolicy": "both-empty=1; one-empty=0; per-image mean",
                    "CheckpointPath": str(checkpoint_path),
                    "ValidationCSV": str(val_csv),
                    "ThresholdSource": threshold_source,
                    "ModelBuilder": builder,
                    "Precision": config["precision"],
                })
                audit_rows.append({
                    "Model": model_name,
                    "Fold": fold,
                    "Status": "PASS",
                    "Message": "",
                    "CheckpointPath": str(checkpoint_path),
                    "Threshold": threshold,
                    "ModelBuilder": builder,
                })

                print(
                    f"[PASS] {model_name}/{fold}: "
                    f"Val Dice={val_dice:.4f}, Test Dice={test_dice:.4f}, "
                    f"Delta={val_dice-test_dice:+.4f}"
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                audit_rows.append({
                    "Model": model_name,
                    "Fold": fold,
                    "Status": "FAIL",
                    "Message": f"{type(exc).__name__}: {exc}",
                    "CheckpointPath": str(checkpoint_path or ""),
                    "Threshold": np.nan,
                    "ModelBuilder": "",
                })
                print(f"[FAIL] {model_name}/{fold}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=2)

    audit_df = pd.DataFrame(audit_rows)
    audit_path = JOURNAL_DIR / "unified_validation_inference_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    if all_per_image:
        per_image_df = pd.concat(all_per_image, ignore_index=True)
        per_image_path = JOURNAL_DIR / "unified_validation_per_image_metrics_all.csv"
        per_image_df.to_csv(per_image_path, index=False, encoding="utf-8-sig")

    fold_df = pd.DataFrame(all_fold_rows)
    fold_path = JOURNAL_DIR / "all_models_unified_signed_gap_by_fold.csv"
    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    if len(fold_df) > 0:
        for model_name, group in fold_df.groupby("Model", sort=True):
            summary_rows.append({
                "Model": model_name,
                "CompletedFolds": len(group),
                "Mean_Val_Dice": group["Val_Dice"].mean(),
                "Mean_Test_Dice": group["Test_Dice"].mean(),
                "Mean_Signed_Delta_Dice": group["Signed_Delta_Dice"].mean(),
                "Mean_Absolute_Delta_Dice": group["Absolute_Delta_Dice"].mean(),
                "Worst_Delta_Dice": group["Signed_Delta_Dice"].max(),
                "Worst_Delta_Dice_Fold": group.loc[
                    group["Signed_Delta_Dice"].idxmax(), "Fold"
                ],
                "Mean_Val_IoU": group["Val_IoU"].mean(),
                "Mean_Test_IoU": group["Test_IoU"].mean(),
                "Mean_Signed_Delta_IoU": group["Signed_Delta_IoU"].mean(),
                "Mean_Absolute_Delta_IoU": group["Absolute_Delta_IoU"].mean(),
                "Worst_Delta_IoU": group["Signed_Delta_IoU"].max(),
                "Worst_Delta_IoU_Fold": group.loc[
                    group["Signed_Delta_IoU"].idxmax(), "Fold"
                ],
                "AllFoldsComplete": len(group) == 6,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = JOURNAL_DIR / "all_models_unified_signed_gap_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    failed = audit_df[audit_df["Status"] != "PASS"]
    print("=" * 110)
    print("[OUTPUT]")
    print(audit_path)
    print(fold_path)
    print(summary_path)
    if all_per_image:
        print(per_image_path)
    print("=" * 110)

    if len(failed) > 0:
        print(f"[COMPLETED WITH FAILURES] {len(failed)} fold(s) failed.")
        print(
            failed[["Model", "Fold", "Message"]]
            .to_string(index=False)
        )
        raise SystemExit(2)

    expected = len(MODEL_REGISTRY) * len(FOLDS)
    if len(audit_df) != expected:
        raise RuntimeError(f"Audit row count mismatch: {len(audit_df)} != {expected}")

    print("[PASS] All 72 validation folds were recomputed with the unified policy.")


if __name__ == "__main__":
    main()
