# 匯入 os，用來取得 checkpoint 檔案大小。
import os

# 匯入 csv，用來輸出 benchmark 結果。
import csv

# 匯入 time，用來控制測試秒數與記錄時間。
import time

# 匯入 subprocess，用來呼叫 nvidia-smi 讀取 GPU 功耗。
import subprocess

# 匯入 threading，用來背景持續取樣 GPU power。
import threading

# 從 pathlib 匯入 Path，讓 Windows 路徑處理更穩。
from pathlib import Path

# 匯入 numpy，用來計算平均值、最大值、標準差。
import numpy as np

# 匯入 torch，用來載入與推論模型。
import torch

# 匯入 torch.nn.functional，用 interpolate 把 logits resize 回 352x352。
import torch.nn.functional as F

# 從 transformers 匯入 SegFormer segmentation model。
from transformers import SegformerForSemanticSegmentation


# ============================================================
# 1. 基本路徑設定
# ============================================================

# 設定 PolypGen 專案根目錄。
BASE_DIR = Path(os.environ.get("POLYPDG_DATA_ROOT", "./data"))

# 設定 Stage 4 power analysis 輸出資料夾。
OUT_DIR = BASE_DIR / "stage4_deployment_benchmark" / "power_jframe_results"

# 如果輸出資料夾不存在，就建立。
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 設定 per-fold power 結果 CSV。
PER_FOLD_CSV = OUT_DIR / "power_jframe_by_fold.csv"

# 設定 summary power 結果 CSV。
SUMMARY_CSV = OUT_DIR / "power_jframe_summary.csv"


# ============================================================
# 2. Benchmark 設定
# ============================================================

# 設定輸入高度，對齊你實驗的 352x352。
INPUT_H = 352

# 設定輸入寬度，對齊你實驗的 352x352。
INPUT_W = 352

# 設定 batch size = 1，模擬單張內視鏡影像即時推論。
BATCH_SIZE = 1

# 設定 warm-up 次數，避免 GPU 第一次初始化影響測量。
WARMUP_ITERS = 50

# 每個模型每個 fold 實際測量秒數。
MEASURE_SECONDS = 30

# idle power 取樣秒數，用來估計空載功耗。
IDLE_SAMPLE_SECONDS = 5

# power 取樣間隔，單位秒。
POWER_SAMPLE_INTERVAL = 1.0

# 自動判斷是否使用 CUDA。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 設定 LOCO folds。
FOLDS = ["C1", "C2", "C3", "C4", "C5", "C6"]


# ============================================================
# 3. 要測的模型設定
# ============================================================

# 建立模型設定清單。
MODEL_CONFIGS = [
    {
        "model_name": "SegFormer-B2-Teacher-Consistency",
        "segformer_variant": "b2",
        "root": BASE_DIR / "stage2C_segformer_b2_consistency_improved_selfcons_official",
        "ckpt_pattern": "fold_test_{fold}/best_model.pth",
        "precisions": ["fp32"],
    },
    {
        "model_name": "SegFormer-B0-Baseline",
        "segformer_variant": "b0",
        "root": BASE_DIR / "stage3B_segformer_b0_loco_baseline",
        "ckpt_pattern": "fold_test_{fold}/stage3B_results/best_model.pth",
        "precisions": ["fp32"],
    },
    {
        "model_name": "SegFormer-B0-KD",
        "segformer_variant": "b0",
        "root": BASE_DIR / "stage3E_segformer_b0_loco_kd_20260428_173054",
        "ckpt_pattern": "fold_test_{fold}/best_model.pth",
        "precisions": ["fp32", "fp16"],
    },
]


# ============================================================
# 4. nvidia-smi power 讀取工具
# ============================================================

def query_gpu_power_once():
    # 呼叫 nvidia-smi，讀取 power.draw、GPU 使用率、memory 使用量。
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]

    # 執行 nvidia-smi 並取得文字輸出。
    output = subprocess.check_output(cmd, encoding="utf-8")

    # 取第一行結果。
    line = output.strip().splitlines()[0]

    # 依照逗號切開 power、util、memory。
    parts = [p.strip() for p in line.split(",")]

    # 轉成 float。
    power_w = float(parts[0])

    # 轉成 float。
    gpu_util = float(parts[1])

    # 轉成 float。
    mem_used_mib = float(parts[2])

    # 回傳這次取樣。
    return power_w, gpu_util, mem_used_mib


def sample_idle_power(seconds=5):
    # 建立 power list。
    powers = []

    # 建立 GPU utilization list。
    utils = []

    # 建立 memory list。
    mems = []

    # 記錄開始時間。
    start = time.time()

    # 在指定秒數內持續取樣。
    while time.time() - start < seconds:
        # 讀取一次 GPU power。
        power_w, gpu_util, mem_used = query_gpu_power_once()

        # 加入 power list。
        powers.append(power_w)

        # 加入 utilization list。
        utils.append(gpu_util)

        # 加入 memory list。
        mems.append(mem_used)

        # 等待 1 秒。
        time.sleep(1.0)

    # 回傳 idle 統計。
    return {
        "idle_mean_power_w": float(np.mean(powers)) if powers else np.nan,
        "idle_peak_power_w": float(np.max(powers)) if powers else np.nan,
        "idle_mean_gpu_util": float(np.mean(utils)) if utils else np.nan,
        "idle_mean_mem_used_mib": float(np.mean(mems)) if mems else np.nan,
        "idle_num_samples": len(powers),
    }


class PowerSampler:
    # 初始化 power sampler。
    def __init__(self, interval_sec=1.0):
        # 設定取樣間隔。
        self.interval_sec = interval_sec

        # 建立停止事件。
        self.stop_event = threading.Event()

        # 建立 samples list。
        self.samples = []

        # 建立 thread。
        self.thread = None

    # 背景取樣函式。
    def _run(self):
        # 如果沒有收到停止訊號，就持續取樣。
        while not self.stop_event.is_set():
            try:
                # 讀取一次 GPU power。
                power_w, gpu_util, mem_used = query_gpu_power_once()

                # 記錄 timestamp 與 power 資料。
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "power_w": power_w,
                        "gpu_util": gpu_util,
                        "mem_used_mib": mem_used,
                    }
                )

            except Exception as e:
                # 如果 nvidia-smi 偶發失敗，就記錄錯誤但不中斷主程式。
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "power_w": np.nan,
                        "gpu_util": np.nan,
                        "mem_used_mib": np.nan,
                    }
                )

            # 等待指定取樣間隔。
            time.sleep(self.interval_sec)

    # 開始背景取樣。
    def start(self):
        # 清空停止事件。
        self.stop_event.clear()

        # 建立 thread。
        self.thread = threading.Thread(target=self._run)

        # 設成 daemon，避免主程式退出時卡住。
        self.thread.daemon = True

        # 啟動 thread。
        self.thread.start()

    # 停止背景取樣。
    def stop(self):
        # 設定停止事件。
        self.stop_event.set()

        # 如果 thread 存在，就等待結束。
        if self.thread is not None:
            self.thread.join()

    # 統計取樣結果。
    def summary(self):
        # 如果沒有 samples，回傳 nan。
        if len(self.samples) == 0:
            return {
                "mean_power_w": np.nan,
                "peak_power_w": np.nan,
                "std_power_w": np.nan,
                "mean_gpu_util": np.nan,
                "mean_mem_used_mib": np.nan,
                "num_power_samples": 0,
            }

        # 取出 power array。
        powers = np.array([s["power_w"] for s in self.samples], dtype=np.float32)

        # 取出 gpu util array。
        utils = np.array([s["gpu_util"] for s in self.samples], dtype=np.float32)

        # 取出 memory array。
        mems = np.array([s["mem_used_mib"] for s in self.samples], dtype=np.float32)

        # 移除 nan。
        powers = powers[~np.isnan(powers)]

        # 移除 nan。
        utils = utils[~np.isnan(utils)]

        # 移除 nan。
        mems = mems[~np.isnan(mems)]

        # 回傳統計結果。
        return {
            "mean_power_w": float(np.mean(powers)) if len(powers) > 0 else np.nan,
            "peak_power_w": float(np.max(powers)) if len(powers) > 0 else np.nan,
            "std_power_w": float(np.std(powers)) if len(powers) > 0 else np.nan,
            "mean_gpu_util": float(np.mean(utils)) if len(utils) > 0 else np.nan,
            "mean_mem_used_mib": float(np.mean(mems)) if len(mems) > 0 else np.nan,
            "num_power_samples": int(len(powers)),
        }


# ============================================================
# 5. Checkpoint loading
# ============================================================

def extract_state_dict(checkpoint_path):
    # 使用 CPU 讀 checkpoint，避免一開始佔 GPU。
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 如果 checkpoint 是 dict。
    if isinstance(checkpoint, dict):

        # 優先取 model_state_dict。
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        # 其次取 state_dict。
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        # 其次取 model。
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        # 否則假設 checkpoint 本身就是 state_dict。
        else:
            state_dict = checkpoint

    # 如果不是 dict，直接報錯。
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    # 建立乾淨 state_dict。
    clean_state_dict = {}

    # 逐一清理 key。
    for key, value in state_dict.items():
        # 複製原 key。
        new_key = key

        # 移除 module. prefix。
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # 移除 model. prefix。
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        # 移除 net. prefix。
        if new_key.startswith("net."):
            new_key = new_key[len("net."):]

        # 修正 Stage3B B0 baseline 多包的 segformer. prefix。
        if new_key.startswith("segformer.segformer.") or new_key.startswith("segformer.decode_head."):
            new_key = new_key[len("segformer."):]

        # 儲存清理後 key。
        clean_state_dict[new_key] = value

    # 回傳清理後 state_dict。
    return clean_state_dict


def infer_num_labels_from_state_dict(state_dict):
    # 預設 binary segmentation = 1。
    num_labels = 1

    # 搜尋 classifier weight。
    for key, value in state_dict.items():
        # 如果 key 是 segmentation classifier。
        if key.endswith("decode_head.classifier.weight"):
            # 第 0 維是類別數。
            num_labels = int(value.shape[0])

            # 找到就停止。
            break

    # 回傳類別數。
    return num_labels


def build_segformer_model(segformer_variant, num_labels):
    # 如果是 B0。
    if segformer_variant == "b0":
        pretrained_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    # 如果是 B2。
    elif segformer_variant == "b2":
        pretrained_name = "nvidia/segformer-b2-finetuned-ade-512-512"

    # 其他不支援。
    else:
        raise ValueError(f"Unsupported SegFormer variant: {segformer_variant}")

    # 建立 SegFormer model。
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # 回傳模型。
    return model


def load_model_from_checkpoint(segformer_variant, checkpoint_path):
    # 取出 state_dict。
    state_dict = extract_state_dict(checkpoint_path)

    # 判斷 num_labels。
    num_labels = infer_num_labels_from_state_dict(state_dict)

    # 建立模型。
    model = build_segformer_model(segformer_variant, num_labels)

    # 載入權重。
    incompatible = model.load_state_dict(state_dict, strict=False)

    # missing keys 數量。
    missing_count = len(incompatible.missing_keys)

    # unexpected keys 數量。
    unexpected_count = len(incompatible.unexpected_keys)

    # 回傳模型與載入狀態。
    return model, num_labels, missing_count, unexpected_count


# ============================================================
# 6. 模型大小與推論
# ============================================================

def count_parameters(model):
    # 回傳模型所有參數數量。
    return sum(param.numel() for param in model.parameters())


def get_file_size_mb(file_path):
    # 取得檔案大小 bytes。
    size_bytes = os.path.getsize(file_path)

    # 換算成 MB。
    return size_bytes / (1024 ** 2)


def estimate_param_size_mb(num_params, precision):
    # FP32 每個參數 4 bytes。
    if precision == "fp32":
        bytes_per_param = 4

    # FP16 每個參數 2 bytes。
    elif precision == "fp16":
        bytes_per_param = 2

    # 其他預設 4 bytes。
    else:
        bytes_per_param = 4

    # 回傳估計大小 MB。
    return num_params * bytes_per_param / (1024 ** 2)


@torch.no_grad()
def inference_once(model, input_tensor):
    # SegFormer forward。
    outputs = model(pixel_values=input_tensor)

    # 取得 logits。
    logits = outputs.logits

    # resize 回 352x352。
    logits = F.interpolate(
        logits,
        size=(INPUT_H, INPUT_W),
        mode="bilinear",
        align_corners=False,
    )

    # binary segmentation 用 sigmoid。
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)

    # multi-class 用 softmax。
    else:
        probs = torch.softmax(logits, dim=1)

    # 回傳 probability map。
    return probs


# ============================================================
# 7. Power + J/frame benchmark
# ============================================================

def benchmark_model_power(model, precision):
    # 切 eval mode。
    model.eval()

    # 移動到 GPU 或 CPU。
    model.to(DEVICE)

    # 如果是 FP16 且 GPU 可用，模型轉 half。
    if precision == "fp16" and DEVICE.type == "cuda":
        model.half()

    # 其他情況用 FP32。
    else:
        model.float()

    # 建立 dummy input。
    input_tensor = torch.randn(
        BATCH_SIZE,
        3,
        INPUT_H,
        INPUT_W,
        device=DEVICE,
    )

    # 如果 FP16，把 input 也轉 half。
    if precision == "fp16" and DEVICE.type == "cuda":
        input_tensor = input_tensor.half()

    # 清 GPU cache。
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # warm-up。
    for _ in range(WARMUP_ITERS):
        _ = inference_once(model, input_tensor)

    # 等待 GPU 完成 warm-up。
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # 量 idle power。
    idle_stats = sample_idle_power(IDLE_SAMPLE_SECONDS)

    # 建立 power sampler。
    sampler = PowerSampler(interval_sec=POWER_SAMPLE_INTERVAL)

    # 啟動 power sampler。
    sampler.start()

    # 記錄正式測量開始時間。
    start_time = time.time()

    # 建立 inference 計數。
    num_frames = 0

    # 在指定秒數內持續推論。
    while time.time() - start_time < MEASURE_SECONDS:
        # 執行一次推論。
        _ = inference_once(model, input_tensor)

        # 等待 GPU 完成，確保 timing 與 power 對齊。
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # 推論張數 +1。
        num_frames += BATCH_SIZE

    # 記錄正式測量結束時間。
    end_time = time.time()

    # 停止 power sampler。
    sampler.stop()

    # 計算實際測量秒數。
    elapsed_sec = end_time - start_time

    # 計算 FPS。
    fps = num_frames / elapsed_sec

    # 計算 latency。
    mean_latency_ms = 1000.0 / fps

    # 取得 active power summary。
    active_stats = sampler.summary()

    # 取得平均功耗。
    mean_power_w = active_stats["mean_power_w"]

    # 取得 idle power。
    idle_power_w = idle_stats["idle_mean_power_w"]

    # 計算 total J/frame。
    j_per_frame_total = mean_power_w / fps if fps > 0 else np.nan

    # 計算扣掉 idle 後的 dynamic power。
    dynamic_power_w = max(mean_power_w - idle_power_w, 0.0)

    # 計算 dynamic J/frame。
    j_per_frame_dynamic = dynamic_power_w / fps if fps > 0 else np.nan

    # 回傳測量結果。
    return {
        "fps_power_measured": fps,
        "mean_latency_ms_power_measured": mean_latency_ms,
        "idle_mean_power_w": idle_stats["idle_mean_power_w"],
        "idle_peak_power_w": idle_stats["idle_peak_power_w"],
        "mean_power_w": active_stats["mean_power_w"],
        "peak_power_w": active_stats["peak_power_w"],
        "std_power_w": active_stats["std_power_w"],
        "dynamic_power_w": dynamic_power_w,
        "j_per_frame_total": j_per_frame_total,
        "j_per_frame_dynamic": j_per_frame_dynamic,
        "mean_gpu_util": active_stats["mean_gpu_util"],
        "mean_mem_used_mib": active_stats["mean_mem_used_mib"],
        "num_power_samples": active_stats["num_power_samples"],
        "measure_seconds": elapsed_sec,
        "num_frames": num_frames,
    }


# ============================================================
# 8. Main
# ============================================================

def main():
    # 如果沒有 CUDA，直接停止。
    if DEVICE.type != "cuda":
        raise RuntimeError("Power/J-frame benchmark requires CUDA GPU.")

    # 印出基本資訊。
    print("=" * 100)
    print("Stage 4 Power / J-frame Benchmark")
    print("=" * 100)
    print("DEVICE =", DEVICE)
    print("GPU =", torch.cuda.get_device_name(0))
    print("MEASURE_SECONDS =", MEASURE_SECONDS)
    print("IDLE_SAMPLE_SECONDS =", IDLE_SAMPLE_SECONDS)
    print("OUT_DIR =", OUT_DIR)
    print("=" * 100)

    # 儲存所有結果。
    all_rows = []

    # 逐一處理模型設定。
    for config in MODEL_CONFIGS:
        # 取得模型名稱。
        model_name = config["model_name"]

        # 取得 SegFormer 版本。
        segformer_variant = config["segformer_variant"]

        # 取得權重根目錄。
        root = config["root"]

        # 取得 checkpoint pattern。
        ckpt_pattern = config["ckpt_pattern"]

        # 取得 precision list。
        precisions = config["precisions"]

        # 逐一跑 C1-C6。
        for fold in FOLDS:
            # 組 checkpoint path。
            checkpoint_path = root / ckpt_pattern.format(fold=fold)

            # 如果 checkpoint 不存在就跳過。
            if not checkpoint_path.exists():
                print(f"[WARNING] Missing checkpoint: {checkpoint_path}")
                continue

            # 逐一跑 precision。
            for precision in precisions:
                # 如果 FP16 但不是 CUDA，就跳過。
                if precision == "fp16" and DEVICE.type != "cuda":
                    continue

                # 印出目前執行項目。
                print("\n" + "-" * 100)
                print(f"[RUN] model={model_name}, fold={fold}, precision={precision}")
                print(f"[CKPT] {checkpoint_path}")

                # 載入模型。
                model, num_labels, missing_count, unexpected_count = load_model_from_checkpoint(
                    segformer_variant,
                    checkpoint_path,
                )

                # 計算參數量。
                num_params = count_parameters(model)

                # checkpoint size。
                ckpt_size_mb = get_file_size_mb(checkpoint_path)

                # estimated model size。
                estimated_param_size_mb = estimate_param_size_mb(num_params, precision)

                # 執行 power benchmark。
                bench = benchmark_model_power(model, precision)

                # 建立一列結果。
                row = {
                    "model_name": model_name,
                    "fold": fold,
                    "precision": precision,
                    "fps_power_measured": bench["fps_power_measured"],
                    "mean_latency_ms_power_measured": bench["mean_latency_ms_power_measured"],
                    "idle_mean_power_w": bench["idle_mean_power_w"],
                    "idle_peak_power_w": bench["idle_peak_power_w"],
                    "mean_power_w": bench["mean_power_w"],
                    "peak_power_w": bench["peak_power_w"],
                    "std_power_w": bench["std_power_w"],
                    "dynamic_power_w": bench["dynamic_power_w"],
                    "j_per_frame_total": bench["j_per_frame_total"],
                    "j_per_frame_dynamic": bench["j_per_frame_dynamic"],
                    "mean_gpu_util": bench["mean_gpu_util"],
                    "mean_mem_used_mib": bench["mean_mem_used_mib"],
                    "num_power_samples": bench["num_power_samples"],
                    "measure_seconds": bench["measure_seconds"],
                    "num_frames": bench["num_frames"],
                    "num_labels": num_labels,
                    "num_params": num_params,
                    "params_m": num_params / 1e6,
                    "ckpt_size_mb": ckpt_size_mb,
                    "estimated_param_size_mb": estimated_param_size_mb,
                    "missing_keys_count": missing_count,
                    "unexpected_keys_count": unexpected_count,
                    "checkpoint_path": str(checkpoint_path),
                }

                # 加入 all_rows。
                all_rows.append(row)

                # 印出簡短結果。
                print(
                    f"[RESULT] FPS={row['fps_power_measured']:.2f}, "
                    f"Latency={row['mean_latency_ms_power_measured']:.2f} ms, "
                    f"MeanPower={row['mean_power_w']:.2f} W, "
                    f"PeakPower={row['peak_power_w']:.2f} W, "
                    f"J/frame={row['j_per_frame_total']:.4f}, "
                    f"DynamicJ/frame={row['j_per_frame_dynamic']:.4f}, "
                    f"GPUUtil={row['mean_gpu_util']:.1f}%"
                )

                # 刪除模型。
                del model

                # 清 GPU cache。
                torch.cuda.empty_cache()

    # 如果沒有任何結果，報錯。
    if len(all_rows) == 0:
        raise RuntimeError("No benchmark results generated.")

    # 寫出 per-fold CSV。
    with open(PER_FOLD_CSV, "w", newline="", encoding="utf-8-sig") as f:
        # 建立 writer。
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))

        # 寫 header。
        writer.writeheader()

        # 寫 rows。
        writer.writerows(all_rows)

    # 建立 summary。
    summary_rows = []

    # 找出所有 model_name + precision 組合。
    groups = sorted(set((r["model_name"], r["precision"]) for r in all_rows))

    # 逐一統整每一組。
    for model_name, precision in groups:
        # 篩選同組 rows。
        group_rows = [
            r for r in all_rows
            if r["model_name"] == model_name and r["precision"] == precision
        ]

        # 建立 summary row。
        summary_row = {
            "model_name": model_name,
            "precision": precision,
            "num_folds": len(group_rows),
            "mean_fps_power_measured": float(np.mean([r["fps_power_measured"] for r in group_rows])),
            "std_fps_power_measured": float(np.std([r["fps_power_measured"] for r in group_rows])),
            "mean_latency_ms_power_measured": float(np.mean([r["mean_latency_ms_power_measured"] for r in group_rows])),
            "mean_idle_power_w": float(np.mean([r["idle_mean_power_w"] for r in group_rows])),
            "mean_power_w": float(np.mean([r["mean_power_w"] for r in group_rows])),
            "peak_power_w": float(np.max([r["peak_power_w"] for r in group_rows])),
            "mean_dynamic_power_w": float(np.mean([r["dynamic_power_w"] for r in group_rows])),
            "mean_j_per_frame_total": float(np.mean([r["j_per_frame_total"] for r in group_rows])),
            "mean_j_per_frame_dynamic": float(np.mean([r["j_per_frame_dynamic"] for r in group_rows])),
            "mean_gpu_util": float(np.mean([r["mean_gpu_util"] for r in group_rows])),
            "mean_mem_used_mib": float(np.mean([r["mean_mem_used_mib"] for r in group_rows])),
            "mean_params_m": float(np.mean([r["params_m"] for r in group_rows])),
            "mean_estimated_param_size_mb": float(np.mean([r["estimated_param_size_mb"] for r in group_rows])),
            "mean_missing_keys_count": float(np.mean([r["missing_keys_count"] for r in group_rows])),
            "mean_unexpected_keys_count": float(np.mean([r["unexpected_keys_count"] for r in group_rows])),
        }

        # 加入 summary。
        summary_rows.append(summary_row)

    # 寫出 summary CSV。
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as f:
        # 建立 writer。
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))

        # 寫 header。
        writer.writeheader()

        # 寫 rows。
        writer.writerows(summary_rows)

    # 印出完成訊息。
    print("\n" + "=" * 100)
    print("[DONE] Power / J-frame benchmark finished.")
    print(f"[PER-FOLD] {PER_FOLD_CSV}")
    print(f"[SUMMARY] {SUMMARY_CSV}")
    print("=" * 100)


# 直接執行時進入 main。
if __name__ == "__main__":
    main()
