# 匯入 os，用來取得檔案大小與處理系統層級資訊。
import os

# 匯入 csv，用來把 benchmark 結果存成 CSV 檔。
import csv

# 匯入 time，用於 CPU 模式下的時間量測。
import time

# 從 pathlib 匯入 Path，讓 Windows 路徑處理更穩定。
from pathlib import Path

# 匯入 numpy，用來計算平均值與標準差。
import numpy as np

# 匯入 torch，這是 PyTorch 主要套件。
import torch

# 匯入 torch.nn.functional，後面會用 interpolate 把 logits resize 回 352x352。
import torch.nn.functional as F

# 從 transformers 匯入 SegFormer 語義分割模型。
from transformers import SegformerForSemanticSegmentation


# ============================================================
# 1. 基本路徑設定
# ============================================================

# 設定你的 PolypGen 專案主資料夾。
BASE_DIR = Path(os.environ.get("POLYPDG_DATA_ROOT", "./data"))

# 設定 Stage 4 主資料夾。
STAGE4_DIR = BASE_DIR / "stage4_deployment_benchmark"

# 設定 benchmark 詳細結果輸出資料夾。
BENCHMARK_DIR = STAGE4_DIR / "benchmark_results"

# 設定 benchmark 統整結果輸出資料夾。
SUMMARY_DIR = STAGE4_DIR / "stage4_summary"

# 如果 benchmark_results 資料夾不存在，就自動建立。
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# 如果 stage4_summary 資料夾不存在，就自動建立。
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Benchmark 參數設定
# ============================================================

# 設定輸入高度，必須和你的實驗前處理 352x352 對齊。
INPUT_H = 352

# 設定輸入寬度，必須和你的實驗前處理 352x352 對齊。
INPUT_W = 352

# 設定 batch size 為 1，模擬臨床即時推論情境。
BATCH_SIZE = 1

# 設定正式測速前的 warm-up 次數，避免第一次 GPU 初始化造成時間偏差。
WARMUP_ITERS = 50

# 設定正式測速次數，200 次可以得到較穩定的平均 latency。
MEASURE_ITERS = 200

# 自動偵測是否有 CUDA GPU；有就用 GPU，沒有就用 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定 LOCO 的六個 fold。
FOLDS = ["C1", "C2", "C3", "C4", "C5", "C6"]


# ============================================================
# 3. 要測的模型設定
# ============================================================

# 建立模型設定清單。
MODEL_CONFIGS = [
    {
        # 模型名稱：Stage 2 最強 teacher。
        "model_name": "SegFormer-B2-Teacher-Consistency",

        # 指定 SegFormer 版本為 B2。
        "segformer_variant": "b2",

        # 指定 teacher 權重根目錄。
        "root": BASE_DIR / "stage2C_segformer_b2_consistency_improved_selfcons_official",

        # 指定 teacher 每個 fold 的權重相對路徑。
        "ckpt_pattern": "fold_test_{fold}/best_model.pth",

        # Teacher 只測 FP32。
        "precisions": ["fp32"],
    },
    {
        # 模型名稱：Stage 3 SegFormer-B0 no-KD baseline。
        "model_name": "SegFormer-B0-Baseline",

        # 指定 SegFormer 版本為 B0。
        "segformer_variant": "b0",

        # 指定 B0 baseline 權重根目錄。
        "root": BASE_DIR / "stage3B_segformer_b0_loco_baseline",

        # 注意：B0 baseline 的權重在 stage3B_results 裡面。
        "ckpt_pattern": "fold_test_{fold}/stage3B_results/best_model.pth",

        # B0 baseline 只測 FP32。
        "precisions": ["fp32"],
    },
    {
        # 模型名稱：Stage 3 最終學生模型 SegFormer-B0+KD。
        "model_name": "SegFormer-B0-KD",

        # 指定 SegFormer 版本為 B0。
        "segformer_variant": "b0",

        # 指定 B0+KD 權重根目錄。
        "root": BASE_DIR / "stage3E_segformer_b0_loco_kd_20260428_173054",

        # 指定 B0+KD 每個 fold 的權重相對路徑。
        "ckpt_pattern": "fold_test_{fold}/best_model.pth",

        # B0+KD 要測 FP32 和 FP16。
        "precisions": ["fp32", "fp16"],
    },
]


# ============================================================
# 4. 讀取 checkpoint 的工具函式
# ============================================================

# 定義函式：從 checkpoint 取出真正的 state_dict，並清理不同訓練程式造成的 key prefix。
def extract_state_dict(checkpoint_path):
    # 用 CPU 讀取 checkpoint，避免一開始就佔用 GPU 記憶體。
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 確認 checkpoint 是否為 dict。
    if isinstance(checkpoint, dict):

        # 常見格式：checkpoint 裡面有 model_state_dict。
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        # 常見格式：checkpoint 裡面有 state_dict。
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        # 常見格式：checkpoint 裡面有 model。
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        # 如果沒有上述 key，就假設 checkpoint 本身就是 state_dict。
        else:
            state_dict = checkpoint

    # 如果 checkpoint 不是 dict，就代表格式不支援。
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # 建立新的乾淨 state_dict。
    clean_state_dict = {}

    # 逐一處理 state_dict 裡面的 key 和 value。
    for key, value in state_dict.items():

        # 複製原始 key。
        new_key = key

        # 如果 key 以 module. 開頭，代表可能來自 DataParallel，移除 module.。
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # 如果 key 以 model. 開頭，代表可能來自自訂 wrapper，移除 model.。
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        # 如果 key 以 net. 開頭，代表可能來自自訂 wrapper，移除 net.。
        if new_key.startswith("net."):
            new_key = new_key[len("net."):]

        # 修正 Stage3B SegFormer-B0 baseline 的權重命名：
        # 原 key: segformer.segformer.encoder.xxx
        # 目標:   segformer.encoder.xxx
        # 原 key: segformer.decode_head.xxx
        # 目標:   decode_head.xxx
        if new_key.startswith("segformer.segformer.") or new_key.startswith("segformer.decode_head."):
            new_key = new_key[len("segformer."):]

        # 把清理後的 key 和 tensor 存起來。
        clean_state_dict[new_key] = value

    # 回傳清理後的 state_dict。
    return clean_state_dict


# 定義函式：從 state_dict 判斷輸出類別數。
def infer_num_labels_from_state_dict(state_dict):
    # 預設為 1，因為你的任務是 binary polyp segmentation。
    num_labels = 1

    # 逐一檢查 state_dict 的 key 和 tensor。
    for key, value in state_dict.items():

        # Hugging Face SegFormer segmentation head 常見權重名稱。
        if key.endswith("decode_head.classifier.weight"):

            # classifier.weight 的第 0 維就是輸出 channel 數。
            num_labels = int(value.shape[0])

            # 找到後就停止搜尋。
            break

    # 回傳推論出的類別數。
    return num_labels


# ============================================================
# 5. 建立 SegFormer 模型的工具函式
# ============================================================

# 定義函式：根據 b0 或 b2 建立 SegFormer 模型。
def build_segformer_model(segformer_variant, num_labels):
    # 如果指定 b0，就使用 SegFormer-B0。
    if segformer_variant == "b0":
        pretrained_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    # 如果指定 b2，就使用 SegFormer-B2。
    elif segformer_variant == "b2":
        pretrained_name = "nvidia/segformer-b2-finetuned-ade-512-512"

    # 如果不是 b0 或 b2，就直接報錯。
    else:
        raise ValueError(f"Unsupported SegFormer variant: {segformer_variant}")

    # 建立 Hugging Face SegFormer 語義分割模型。
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # 回傳模型。
    return model


# 定義函式：從 checkpoint 建立並載入模型。
def load_model_from_checkpoint(segformer_variant, checkpoint_path):
    # 從 checkpoint 取出 state_dict。
    state_dict = extract_state_dict(checkpoint_path)

    # 從 state_dict 判斷 num_labels。
    num_labels = infer_num_labels_from_state_dict(state_dict)

    # 根據 SegFormer 版本與 num_labels 建立模型。
    model = build_segformer_model(segformer_variant, num_labels)

    # 將 checkpoint 權重載入模型。
    incompatible = model.load_state_dict(state_dict, strict=False)

    # 取得 missing keys 數量。
    missing_count = len(incompatible.missing_keys)

    # 取得 unexpected keys 數量。
    unexpected_count = len(incompatible.unexpected_keys)

    # 如果仍有 key 對不上，就印出前幾個例子方便除錯。
    if missing_count > 0 or unexpected_count > 0:
        print(f"[WARNING] Weight loading mismatch for: {checkpoint_path}")
        print(f"[WARNING] missing_keys_count={missing_count}, unexpected_keys_count={unexpected_count}")

        # 印出前 10 個 missing keys。
        if missing_count > 0:
            print("[WARNING] First missing keys:")
            for key in incompatible.missing_keys[:10]:
                print(f"  missing: {key}")

        # 印出前 10 個 unexpected keys。
        if unexpected_count > 0:
            print("[WARNING] First unexpected keys:")
            for key in incompatible.unexpected_keys[:10]:
                print(f"  unexpected: {key}")

    # 回傳模型與相關資訊。
    return model, num_labels, missing_count, unexpected_count


# ============================================================
# 6. 參數量與模型大小工具函式
# ============================================================

# 定義函式：計算模型參數總數。
def count_parameters(model):
    # 將模型中所有參數的元素數量加總。
    return sum(param.numel() for param in model.parameters())


# 定義函式：取得 checkpoint 檔案大小，單位 MB。
def get_file_size_mb(file_path):
    # 取得檔案大小，單位 bytes。
    size_bytes = os.path.getsize(file_path)

    # 將 bytes 轉換成 MB。
    size_mb = size_bytes / (1024 ** 2)

    # 回傳 MB。
    return size_mb


# 定義函式：根據參數量與 precision 估計部署參數大小。
def estimate_param_size_mb(num_params, precision):
    # FP32 每個參數佔 4 bytes。
    if precision == "fp32":
        bytes_per_param = 4

    # FP16 每個參數佔 2 bytes。
    elif precision == "fp16":
        bytes_per_param = 2

    # 其他未知情況預設使用 FP32。
    else:
        bytes_per_param = 4

    # 估計模型參數大小，單位 MB。
    size_mb = num_params * bytes_per_param / (1024 ** 2)

    # 回傳估計大小。
    return size_mb


# ============================================================
# 7. 單次推論函式
# ============================================================

# 關閉梯度計算，因為 benchmark 只做 inference。
@torch.no_grad()
def inference_once(model, input_tensor):
    # 使用 Hugging Face SegFormer 的 forward 介面進行推論。
    outputs = model(pixel_values=input_tensor)

    # 取得模型輸出的 logits。
    logits = outputs.logits

    # 將 logits resize 回原本設定的 352x352。
    logits = F.interpolate(
        logits,
        size=(INPUT_H, INPUT_W),
        mode="bilinear",
        align_corners=False,
    )

    # 如果輸出 channel 是 1，代表 binary segmentation，用 sigmoid。
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)

    # 如果輸出 channel 大於 1，代表多類別 segmentation，用 softmax。
    else:
        probs = torch.softmax(logits, dim=1)

    # 回傳機率圖，避免推論結果被忽略。
    return probs


# ============================================================
# 8. Benchmark 單一模型
# ============================================================

# 定義函式：測量某個模型在指定 precision 下的推論速度與 VRAM。
def benchmark_model(model, precision):
    # 將模型切換為 eval 模式。
    model.eval()

    # 將模型放到 GPU 或 CPU。
    model.to(DEVICE)

    # 如果使用 FP16 且目前是 CUDA，就把模型轉成 half precision。
    if precision == "fp16" and DEVICE.type == "cuda":
        model.half()

    # 其他情況都使用 FP32。
    else:
        model.float()

    # 建立一個隨機輸入 tensor，形狀為 [B, C, H, W]。
    input_tensor = torch.randn(
        BATCH_SIZE,
        3,
        INPUT_H,
        INPUT_W,
        device=DEVICE,
    )

    # 如果是 FP16，就把輸入也轉成 half precision。
    if precision == "fp16" and DEVICE.type == "cuda":
        input_tensor = input_tensor.half()

    # 如果使用 CUDA，先清空 GPU cache。
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # 如果使用 CUDA，重設 peak memory 統計。
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 先做 warm-up，不列入正式時間。
    for _ in range(WARMUP_ITERS):
        _ = inference_once(model, input_tensor)

    # 如果使用 CUDA，等待 GPU 完成 warm-up。
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # 建立 list 儲存每一次 latency。
    latencies_ms = []

    # 如果使用 CUDA，就用 CUDA Event 量測 GPU 推論時間。
    if DEVICE.type == "cuda":

        # 正式量測 MEASURE_ITERS 次。
        for _ in range(MEASURE_ITERS):

            # 建立開始事件。
            start_event = torch.cuda.Event(enable_timing=True)

            # 建立結束事件。
            end_event = torch.cuda.Event(enable_timing=True)

            # 記錄開始時間。
            start_event.record()

            # 執行一次推論。
            _ = inference_once(model, input_tensor)

            # 記錄結束時間。
            end_event.record()

            # 等待 GPU 完成推論。
            torch.cuda.synchronize()

            # 計算 latency，單位 ms。
            elapsed_ms = start_event.elapsed_time(end_event)

            # 將這次 latency 加入 list。
            latencies_ms.append(elapsed_ms)

    # 如果沒有 CUDA，就用 CPU 的 perf_counter 量測。
    else:

        # 正式量測 MEASURE_ITERS 次。
        for _ in range(MEASURE_ITERS):

            # 記錄開始時間。
            start_time = time.perf_counter()

            # 執行一次推論。
            _ = inference_once(model, input_tensor)

            # 記錄結束時間。
            end_time = time.perf_counter()

            # 換算成 ms。
            elapsed_ms = (end_time - start_time) * 1000.0

            # 將這次 latency 加入 list。
            latencies_ms.append(elapsed_ms)

    # 計算平均 latency。
    mean_latency_ms = float(np.mean(latencies_ms))

    # 計算 latency 標準差。
    std_latency_ms = float(np.std(latencies_ms))

    # 由 latency 換算 FPS。
    fps = 1000.0 / mean_latency_ms

    # 如果使用 CUDA，就取得 peak VRAM。
    if DEVICE.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 如果沒有 CUDA，VRAM 記為 0。
    else:
        peak_vram_mb = 0.0

    # 回傳 benchmark 結果。
    return {
        "mean_latency_ms": mean_latency_ms,
        "std_latency_ms": std_latency_ms,
        "fps": fps,
        "peak_vram_mb": peak_vram_mb,
    }


# ============================================================
# 9. 主程式
# ============================================================

# 定義主函式。
def main():
    # 印出目前使用的 device。
    print(f"[INFO] Device: {DEVICE}")

    # 印出 PyTorch 版本。
    print(f"[INFO] Torch version: {torch.__version__}")

    # 如果使用 CUDA，印出 GPU 名稱。
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # 建立 list 儲存所有 fold 的 benchmark 結果。
    all_rows = []

    # 逐一處理 MODEL_CONFIGS 裡面每個模型設定。
    for config in MODEL_CONFIGS:

        # 取得模型名稱。
        model_name = config["model_name"]

        # 取得 SegFormer 版本。
        segformer_variant = config["segformer_variant"]

        # 取得 checkpoint 根目錄。
        root = config["root"]

        # 取得 checkpoint 路徑樣板。
        ckpt_pattern = config["ckpt_pattern"]

        # 取得要測的 precision 清單。
        precisions = config["precisions"]

        # 逐一處理 C1 到 C6。
        for fold in FOLDS:

            # 組出該 fold 的 checkpoint 完整路徑。
            checkpoint_path = root / ckpt_pattern.format(fold=fold)

            # 如果 checkpoint 不存在，就印出警告並跳過。
            if not checkpoint_path.exists():
                print(f"[WARNING] Missing checkpoint: {checkpoint_path}")
                continue

            # 逐一處理 FP32 或 FP16。
            for precision in precisions:

                # 如果是 FP16 但沒有 CUDA，就跳過，因為 CPU FP16 沒有部署意義。
                if precision == "fp16" and DEVICE.type != "cuda":
                    print(f"[WARNING] Skip FP16 on CPU: {model_name}, {fold}")
                    continue

                # 印出目前正在測的模型、fold、precision。
                print(f"\n[RUN] model={model_name}, fold={fold}, precision={precision}")

                # 載入模型。
                model, num_labels, missing_count, unexpected_count = load_model_from_checkpoint(
                    segformer_variant,
                    checkpoint_path,
                )

                # 計算模型參數量。
                num_params = count_parameters(model)

                # 計算 checkpoint 檔案大小。
                ckpt_size_mb = get_file_size_mb(checkpoint_path)

                # 估計此 precision 下的參數大小。
                estimated_param_size_mb = estimate_param_size_mb(num_params, precision)

                # 執行 benchmark。
                bench = benchmark_model(model, precision)

                # 建立一列結果。
                row = {
                    "model_name": model_name,
                    "fold": fold,
                    "precision": precision,
                    "segformer_variant": segformer_variant,
                    "num_labels": num_labels,
                    "num_params": num_params,
                    "ckpt_size_mb": ckpt_size_mb,
                    "estimated_param_size_mb": estimated_param_size_mb,
                    "fps": bench["fps"],
                    "mean_latency_ms": bench["mean_latency_ms"],
                    "std_latency_ms": bench["std_latency_ms"],
                    "peak_vram_mb": bench["peak_vram_mb"],
                    "missing_keys_count": missing_count,
                    "unexpected_keys_count": unexpected_count,
                    "checkpoint_path": str(checkpoint_path),
                }

                # 把這列結果存入 all_rows。
                all_rows.append(row)

                # 印出簡短 benchmark 結果。
                print(
                    f"[RESULT] FPS={row['fps']:.2f}, "
                    f"Latency={row['mean_latency_ms']:.2f} ms, "
                    f"VRAM={row['peak_vram_mb']:.1f} MB, "
                    f"Params={row['num_params'] / 1e6:.2f}M, "
                    f"Missing={row['missing_keys_count']}, "
                    f"Unexpected={row['unexpected_keys_count']}"
                )

                # 刪除模型，釋放 RAM / VRAM。
                del model

                # 如果使用 CUDA，清空 GPU cache。
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    # 設定 per-fold CSV 輸出路徑。
    per_fold_csv = BENCHMARK_DIR / "deployment_benchmark_by_fold.csv"

    # 如果 all_rows 有資料，就輸出 per-fold CSV。
    if all_rows:

        # 取得 CSV 欄位名稱。
        fieldnames = list(all_rows[0].keys())

        # 開啟 CSV 檔案。
        with open(per_fold_csv, "w", newline="", encoding="utf-8-sig") as f:

            # 建立 CSV writer。
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 寫入表頭。
            writer.writeheader()

            # 寫入所有 rows。
            writer.writerows(all_rows)

        # 印出儲存位置。
        print(f"\n[SAVED] {per_fold_csv}")

    # 建立 summary rows。
    summary_rows = []

    # 取得所有 model_name 和 precision 組合。
    groups = sorted(set((row["model_name"], row["precision"]) for row in all_rows))

    # 逐一統整每個 group。
    for model_name, precision in groups:

        # 篩選同一個 model_name 和 precision 的 rows。
        group_rows = [
            row for row in all_rows
            if row["model_name"] == model_name and row["precision"] == precision
        ]

        # 如果沒有資料，就跳過。
        if not group_rows:
            continue

        # 建立 summary row。
        summary_row = {
            "model_name": model_name,
            "precision": precision,
            "num_folds": len(group_rows),
            "mean_fps": float(np.mean([r["fps"] for r in group_rows])),
            "std_fps": float(np.std([r["fps"] for r in group_rows])),
            "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in group_rows])),
            "std_latency_ms": float(np.std([r["mean_latency_ms"] for r in group_rows])),
            "mean_peak_vram_mb": float(np.mean([r["peak_vram_mb"] for r in group_rows])),
            "mean_num_params": float(np.mean([r["num_params"] for r in group_rows])),
            "mean_ckpt_size_mb": float(np.mean([r["ckpt_size_mb"] for r in group_rows])),
            "mean_estimated_param_size_mb": float(np.mean([r["estimated_param_size_mb"] for r in group_rows])),
            "mean_missing_keys_count": float(np.mean([r["missing_keys_count"] for r in group_rows])),
            "mean_unexpected_keys_count": float(np.mean([r["unexpected_keys_count"] for r in group_rows])),
        }

        # 把 summary row 加入 summary_rows。
        summary_rows.append(summary_row)

    # 設定 summary CSV 輸出路徑。
    summary_csv = SUMMARY_DIR / "deployment_summary_runtime.csv"

    # 如果 summary_rows 有資料，就輸出 summary CSV。
    if summary_rows:

        # 取得 summary CSV 欄位名稱。
        summary_fieldnames = list(summary_rows[0].keys())

        # 開啟 summary CSV 檔案。
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:

            # 建立 CSV writer。
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)

            # 寫入表頭。
            writer.writeheader()

            # 寫入所有 summary rows。
            writer.writerows(summary_rows)

        # 印出 summary 儲存位置。
        print(f"[SAVED] {summary_csv}")

    # 印出完成訊息。
    print("\n[DONE] Stage 4 runtime benchmark finished.")


# 如果此檔案是被直接執行，就執行 main。
if __name__ == "__main__":

    # 執行主函式。
    main()
