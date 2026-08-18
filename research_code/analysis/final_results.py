# 匯入 Path，用來處理 Windows 路徑。
import os
from pathlib import Path

# 匯入 pandas，用來建立與輸出 CSV 表格。
import pandas as pd

# 設定 PolypGen 專案主資料夾。
BASE_DIR = Path(os.environ.get("POLYPDG_DATA_ROOT", "./data"))

# 設定 full paper 最終整理資料夾。
OUT_DIR = BASE_DIR / "fullpaper_final_results"

# 設定表格輸出資料夾。
TABLE_DIR = OUT_DIR / "01_tables"

# 建立 fullpaper_final_results 資料夾。
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 建立 01_tables 資料夾。
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Table 1：Dataset / LOCO split
# ============================================================

# 建立資料集中心分布資料。
table1_data = [
    {"Center": "C1", "Num_images": 256},
    {"Center": "C2", "Num_images": 301},
    {"Center": "C3", "Num_images": 457},
    {"Center": "C4", "Num_images": 227},
    {"Center": "C5", "Num_images": 208},
    {"Center": "C6", "Num_images": 88},
    {"Center": "Total", "Num_images": 1537},
]

# 將資料轉成 DataFrame。
table1_df = pd.DataFrame(table1_data)

# 輸出 Table 1。
table1_df.to_csv(
    TABLE_DIR / "table1_dataset_loco_split.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Table 2：Stage-wise summary
# ============================================================

# 建立 Stage 1 到 Stage 4 的總結表。
table2_data = [
    {
        "Stage": "Stage 1",
        "Model": "U-Net + LOCO",
        "Purpose": "Cross-center diagnostic baseline",
        "Mean_Dice": 63.40,
        "Worst_Center": "C4",
        "C4_Dice": 43.45,
        "Mean_HD95": 71.13,
        "Mean_ASSD": 25.63,
        "Mean_Abs_Delta_Dice": 12.65,
    },
    {
        "Stage": "Stage 2",
        "Model": "SegFormer-B2 + Consistency",
        "Purpose": "Robust teacher",
        "Mean_Dice": 79.61,
        "Worst_Center": "C4",
        "C4_Dice": 66.13,
        "Mean_HD95": 29.15,
        "Mean_ASSD": 8.32,
        "Mean_Abs_Delta_Dice": 7.82,
    },
    {
        "Stage": "Stage 3",
        "Model": "SegFormer-B0 + KD",
        "Purpose": "Final lightweight student",
        "Mean_Dice": 77.05,
        "Worst_Center": "C4",
        "C4_Dice": 60.44,
        "Mean_HD95": 32.81,
        "Mean_ASSD": 10.63,
        "Mean_Abs_Delta_Dice": 8.26,
    },
    {
        "Stage": "Stage 4",
        "Model": "SegFormer-B0 + KD FP16",
        "Purpose": "Final deployment candidate",
        "Mean_Dice": 77.05,
        "Worst_Center": "C4",
        "C4_Dice": 60.44,
        "Mean_HD95": 32.81,
        "Mean_ASSD": 10.63,
        "Mean_Abs_Delta_Dice": 8.26,
    },
]

# 將 Stage-wise summary 轉成 DataFrame。
table2_df = pd.DataFrame(table2_data)

# 輸出 Table 2。
table2_df.to_csv(
    TABLE_DIR / "table2_stagewise_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Table 5：Stage 4 deployment benchmark
# ============================================================

# 建立 Stage 4 deployment benchmark 主表。
table5_data = [
    {
        "Model": "SegFormer-B2 Teacher",
        "Precision": "FP32",
        "Mean_Dice": 79.61,
        "C4_Dice": 66.13,
        "FPS": 31.44,
        "Latency_ms": 32.08,
        "VRAM_MB": 422.35,
        "Params_M": 27.35,
        "Estimated_Size_MB": 104.32,
        "Decision": "Accurate but heavy",
    },
    {
        "Model": "SegFormer-B0",
        "Precision": "FP32",
        "Mean_Dice": 73.15,
        "C4_Dice": 47.02,
        "FPS": 88.90,
        "Latency_ms": 11.26,
        "VRAM_MB": 125.64,
        "Params_M": 3.71,
        "Estimated_Size_MB": 14.17,
        "Decision": "Fast but weak C4",
    },
    {
        "Model": "SegFormer-B0 + KD",
        "Precision": "FP32",
        "Mean_Dice": 77.05,
        "C4_Dice": 60.44,
        "FPS": 87.05,
        "Latency_ms": 11.57,
        "VRAM_MB": 125.64,
        "Params_M": 3.71,
        "Estimated_Size_MB": 14.17,
        "Decision": "Balanced",
    },
    {
        "Model": "SegFormer-B0 + KD",
        "Precision": "FP16",
        "Mean_Dice": 77.05,
        "C4_Dice": 60.44,
        "FPS": 95.26,
        "Latency_ms": 10.53,
        "VRAM_MB": 55.43,
        "Params_M": 3.71,
        "Estimated_Size_MB": 7.08,
        "Decision": "Final deployment candidate",
    },
]

# 將 Stage 4 deployment benchmark 轉成 DataFrame。
table5_df = pd.DataFrame(table5_data)

# 輸出 Table 5。
table5_df.to_csv(
    TABLE_DIR / "table5_stage4_deployment_benchmark.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 完成訊息
# ============================================================

# 印出整理完成位置。
print("=" * 100)
print("[DONE] Full paper result tables generated.")
print(f"[OUTPUT FOLDER] {TABLE_DIR}")
print("=" * 100)

# 印出目前產生的表格。
for file_path in TABLE_DIR.glob("*.csv"):
    print(file_path)
