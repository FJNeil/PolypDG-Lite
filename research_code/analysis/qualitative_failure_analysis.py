# 匯入 os，用來做資料夾搜尋與路徑處理。
import os

# 從 pathlib 匯入 Path，讓 Windows 路徑處理更穩定。
from pathlib import Path

# 匯入 numpy，用來處理影像與 mask array。
import numpy as np

# 匯入 pandas，用來輸出案例選擇表格。
import pandas as pd

# 從 PIL 匯入 Image，用來讀取 PNG/JPG 影像。
from PIL import Image

# 匯入 matplotlib.pyplot，用來產生 qualitative figure。
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本路徑設定
# ============================================================

# 設定 PolypGen 專案主資料夾。
BASE_DIR = Path(os.environ.get("POLYPDG_DATA_ROOT", "./data"))

# 設定輸出資料夾。
OUT_DIR = BASE_DIR / "fullpaper_final_results" / "02_figures" / "qualitative_failure_analysis"

# 建立輸出資料夾。
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 指定要分析的 worst-center fold。
TARGET_FOLD = "fold_test_C4"

# 最多挑幾個案例放進圖中。
MAX_CASES = 5


# ============================================================
# 2. 要比較的模型設定
# ============================================================

# 設定每個模型對應的資料夾關鍵字。
# 程式會自動在 BASE_DIR 底下找符合關鍵字、且含有 fold_test_C4/test_predictions 的資料夾。
MODEL_SEARCH_RULES = [
    {
        "display_name": "U-Net + LOCO",
        "keywords": ["unet"],
    },
    {
        "display_name": "SegFormer-B2 + Consistency",
        "keywords": ["stage2c", "segformer", "consistency"],
    },
    {
        "display_name": "SegFormer-B0 + LOCO",
        "keywords": ["stage3b", "segformer", "b0"],
    },
    {
        "display_name": "SegFormer-B0 + KD",
        "keywords": ["stage3e", "segformer", "b0", "kd"],
    },
]


# ============================================================
# 3. 工具函式：找 test_predictions 資料夾
# ============================================================

def contains_all_keywords(folder_name, keywords):
    # 將資料夾名稱轉成小寫，方便不分大小寫比對。
    lower_name = folder_name.lower()

    # 檢查所有 keywords 是否都出現在資料夾名稱中。
    return all(keyword.lower() in lower_name for keyword in keywords)


def find_test_predictions_dir(experiment_root):
    # 建立目標 fold 路徑。
    fold_dir = experiment_root / TARGET_FOLD

    # 如果 fold_test_C4 不存在，就回傳 None。
    if not fold_dir.exists():
        return None

    # 遞迴搜尋 fold_test_C4 底下所有資料夾。
    for root, dirs, files in os.walk(fold_dir):
        # 將目前 root 轉成 Path。
        root_path = Path(root)

        # 如果目前資料夾名稱是 test_predictions。
        if root_path.name == "test_predictions":
            # 檢查是否有 mask_pred 資料夾。
            has_pred = (root_path / "mask_pred").is_dir()

            # 檢查是否有 mask_gt 資料夾。
            has_gt = (root_path / "mask_gt").is_dir()

            # 檢查是否有 image 資料夾。
            has_image = (root_path / "image").is_dir()

            # 如果三個資料夾都有，就回傳這個 test_predictions 路徑。
            if has_pred and has_gt and has_image:
                return root_path

    # 如果找不到合格的 test_predictions，就回傳 None。
    return None


def discover_model_prediction_dirs():
    # 建立字典，儲存每個模型找到的 test_predictions 路徑。
    model_dirs = {}

    # 取得 BASE_DIR 底下所有第一層資料夾。
    candidate_roots = [p for p in BASE_DIR.iterdir() if p.is_dir()]

    # 逐一處理每個模型規則。
    for rule in MODEL_SEARCH_RULES:
        # 取得顯示名稱。
        display_name = rule["display_name"]

        # 取得關鍵字。
        keywords = rule["keywords"]

        # 建立候選清單。
        matched_roots = []

        # 逐一檢查每個資料夾是否符合關鍵字。
        for root in candidate_roots:
            # 如果資料夾名稱包含所有指定關鍵字，就加入候選。
            if contains_all_keywords(root.name, keywords):
                matched_roots.append(root)

        # 依照路徑長度排序，通常較短的是主實驗資料夾。
        matched_roots = sorted(matched_roots, key=lambda p: len(str(p)))

        # 預設找不到。
        found_dir = None

        # 逐一檢查候選資料夾是否有 test_predictions。
        for root in matched_roots:
            # 搜尋 test_predictions。
            pred_dir = find_test_predictions_dir(root)

            # 如果找到就停止。
            if pred_dir is not None:
                found_dir = pred_dir
                break

        # 如果找到就記錄。
        if found_dir is not None:
            model_dirs[display_name] = found_dir
            print(f"[FOUND] {display_name}: {found_dir}")

        # 如果找不到就警告。
        else:
            print(f"[WARNING] Cannot find test_predictions for: {display_name}")

    # 回傳找到的模型預測資料夾。
    return model_dirs


# ============================================================
# 4. 工具函式：讀取 mask 與 image
# ============================================================

def list_png_by_stem(folder):
    # 建立 stem -> path 字典。
    mapping = {}

    # 如果資料夾不存在，回傳空字典。
    if not folder.is_dir():
        return mapping

    # 讀取資料夾內所有 png。
    for file_path in folder.glob("*.png"):
        # 使用不含副檔名的檔名作為 stem。
        mapping[file_path.stem] = file_path

    # 回傳 mapping。
    return mapping


def read_binary_mask(mask_path):
    # 使用 PIL 讀取 mask 並轉成灰階。
    mask = Image.open(mask_path).convert("L")

    # 轉成 numpy array。
    mask_np = np.array(mask)

    # 二值化，>127 視為前景。
    mask_bin = (mask_np > 127).astype(np.uint8)

    # 回傳 binary mask。
    return mask_bin


def read_rgb_image(image_path):
    # 使用 PIL 讀取 image 並轉成 RGB。
    image = Image.open(image_path).convert("RGB")

    # 轉成 numpy array。
    image_np = np.array(image)

    # 回傳 RGB image。
    return image_np


# ============================================================
# 5. 工具函式：計算 Dice / IoU
# ============================================================

def compute_dice_iou(pred, gt):
    # 將 pred 轉成 uint8。
    pred = pred.astype(np.uint8)

    # 將 gt 轉成 uint8。
    gt = gt.astype(np.uint8)

    # 計算 TP。
    tp = int(((pred == 1) & (gt == 1)).sum())

    # 計算 FP。
    fp = int(((pred == 1) & (gt == 0)).sum())

    # 計算 FN。
    fn = int(((pred == 0) & (gt == 1)).sum())

    # 如果 pred 和 gt 都沒有前景，Dice / IoU 視為 1。
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 1.0

    # 計算 Dice。
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    # 計算 IoU。
    iou = tp / (tp + fp + fn + 1e-7)

    # 回傳 Dice 和 IoU。
    return float(dice), float(iou)


# ============================================================
# 6. 工具函式：overlay mask
# ============================================================

def overlay_mask(image_np, mask_np, color=(255, 0, 0), alpha=0.45):
    # 複製原圖，避免改到原始 array。
    overlay = image_np.copy().astype(np.float32)

    # 建立布林 mask。
    mask_bool = mask_np.astype(bool)

    # 將顏色轉成 numpy array。
    color_np = np.array(color, dtype=np.float32)

    # 對 mask 區域做半透明 overlay。
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_np

    # 轉回 uint8。
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # 回傳 overlay image。
    return overlay


# ============================================================
# 7. 建立所有共同案例與指標
# ============================================================

def build_case_metric_table(model_dirs):
    # 如果沒有找到任何模型，直接報錯。
    if len(model_dirs) == 0:
        raise RuntimeError("No model prediction folders were found.")

    # 建立每個模型的 mask_pred mapping。
    pred_maps = {}

    # 建立 gt 與 image mapping，使用第一個模型的 test_predictions 當來源。
    gt_map = None
    image_map = None

    # 逐一讀取每個模型的 prediction mapping。
    for model_name, pred_root in model_dirs.items():
        # 建立 mask_pred mapping。
        pred_maps[model_name] = list_png_by_stem(pred_root / "mask_pred")

        # 如果 gt_map 尚未建立，就使用這個模型的 mask_gt。
        if gt_map is None:
            gt_map = list_png_by_stem(pred_root / "mask_gt")

        # 如果 image_map 尚未建立，就使用這個模型的 image。
        if image_map is None:
            image_map = list_png_by_stem(pred_root / "image")

    # 從 gt_map 開始取得所有 stem。
    common_stems = set(gt_map.keys())

    # 只保留有 image 的 stem。
    common_stems = common_stems.intersection(set(image_map.keys()))

    # 只保留每個模型都有 prediction 的 stem。
    for model_name, pred_map in pred_maps.items():
        common_stems = common_stems.intersection(set(pred_map.keys()))

    # 排序 stem。
    common_stems = sorted(list(common_stems))

    # 如果沒有共同案例，報錯。
    if len(common_stems) == 0:
        raise RuntimeError("No common stems found across selected models.")

    # 建立 rows。
    rows = []

    # 逐一計算每張影像的每個模型 Dice。
    for stem in common_stems:
        # 讀取 gt mask。
        gt = read_binary_mask(gt_map[stem])

        # 建立 row。
        row = {
            "Stem": stem,
            "Image_Path": str(image_map[stem]),
            "GT_Path": str(gt_map[stem]),
        }

        # 逐一處理每個模型。
        for model_name, pred_map in pred_maps.items():
            # 讀取 prediction mask。
            pred = read_binary_mask(pred_map[stem])

            # 計算 Dice / IoU。
            dice, iou = compute_dice_iou(pred, gt)

            # 存入 row。
            row[f"{model_name}_Dice"] = dice
            row[f"{model_name}_IoU"] = iou
            row[f"{model_name}_Pred_Path"] = str(pred_map[stem])

        # 加入 rows。
        rows.append(row)

    # 轉成 DataFrame。
    df = pd.DataFrame(rows)

    # 回傳 metric table。
    return df


# ============================================================
# 8. 自動挑選代表性 failure / improvement cases
# ============================================================

def select_cases(metric_df):
    # 複製 DataFrame。
    df = metric_df.copy()

    # 建立選擇結果 list。
    selected = []

    # 設定常用欄位名稱。
    kd_col = "SegFormer-B0 + KD_Dice"
    unet_col = "U-Net + LOCO_Dice"
    b0_col = "SegFormer-B0 + LOCO_Dice"

    # 如果有 U-Net 與 B0+KD，就挑 B0+KD 相對 U-Net 改善最多的案例。
    if kd_col in df.columns and unet_col in df.columns:
        # 計算 improvement。
        df["KD_vs_UNet_Improvement"] = df[kd_col] - df[unet_col]

        # 挑改善最多的 2 張。
        top_improved = df.sort_values("KD_vs_UNet_Improvement", ascending=False).head(2)

        # 記錄案例類型。
        for _, row in top_improved.iterrows():
            selected.append((row["Stem"], "Improved over U-Net"))

    # 如果有 B0 與 B0+KD，就挑 KD 相對 B0 改善最多的案例。
    if kd_col in df.columns and b0_col in df.columns:
        # 計算 KD improvement。
        df["KD_vs_B0_Improvement"] = df[kd_col] - df[b0_col]

        # 挑改善最多的 2 張。
        top_kd = df.sort_values("KD_vs_B0_Improvement", ascending=False).head(2)

        # 記錄案例類型。
        for _, row in top_kd.iterrows():
            selected.append((row["Stem"], "KD improvement"))

    # 如果有 B0+KD，就挑 B0+KD 自己 Dice 最低的案例，作為 remaining failure。
    if kd_col in df.columns:
        # 挑 final model Dice 最低的 2 張。
        failure_cases = df.sort_values(kd_col, ascending=True).head(2)

        # 記錄案例類型。
        for _, row in failure_cases.iterrows():
            selected.append((row["Stem"], "Remaining failure"))

    # 去除重複 stem，保留最早出現的 case type。
    unique_selected = []

    # 建立 seen set。
    seen = set()

    # 逐一處理 selected。
    for stem, case_type in selected:
        # 如果 stem 還沒出現過，就加入。
        if stem not in seen:
            unique_selected.append((stem, case_type))
            seen.add(stem)

        # 如果已達 MAX_CASES，就停止。
        if len(unique_selected) >= MAX_CASES:
            break

    # 如果選到的案例不足，就用 B0+KD Dice 由低到高補滿。
    if len(unique_selected) < MAX_CASES and kd_col in df.columns:
        # 依 final Dice 由低到高排序。
        fallback = df.sort_values(kd_col, ascending=True)

        # 逐一補案例。
        for _, row in fallback.iterrows():
            stem = row["Stem"]

            # 避免重複。
            if stem not in seen:
                unique_selected.append((stem, "Additional hard case"))
                seen.add(stem)

            # 補滿就停止。
            if len(unique_selected) >= MAX_CASES:
                break

    # 建立 selected DataFrame。
    selected_rows = []

    # 逐一取出完整 row。
    for stem, case_type in unique_selected:
        # 找到該 stem 的 row。
        row = df[df["Stem"] == stem].iloc[0].to_dict()

        # 加入 Case_Type。
        row["Case_Type"] = case_type

        # 加入 selected rows。
        selected_rows.append(row)

    # 回傳選到的案例 DataFrame。
    return pd.DataFrame(selected_rows)


# ============================================================
# 9. 畫 qualitative figure
# ============================================================

def plot_qualitative_figure(selected_df, model_dirs):
    # 取得模型名稱列表。
    model_names = list(model_dirs.keys())

    # 設定欄位名稱：Original + GT + 每個模型 prediction。
    columns = ["Image", "Ground Truth"] + model_names

    # 計算列數。
    n_rows = len(selected_df)

    # 計算欄數。
    n_cols = len(columns)

    # 設定整張圖大小。
    plt.figure(figsize=(3.0 * n_cols, 3.2 * n_rows))

    # 逐一畫每個 selected case。
    for row_idx, (_, row) in enumerate(selected_df.iterrows()):
        # 讀取 original image。
        image_np = read_rgb_image(row["Image_Path"])

        # 讀取 gt mask。
        gt_np = read_binary_mask(row["GT_Path"])

        # 第一欄：原圖。
        ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + 1)
        ax.imshow(image_np)
        ax.set_title(f"{row['Stem']}\n{row['Case_Type']}", fontsize=9)
        ax.axis("off")

        # 第二欄：GT overlay。
        gt_overlay = overlay_mask(image_np, gt_np, color=(0, 255, 0), alpha=0.45)
        ax = plt.subplot(n_rows, n_cols, row_idx * n_cols + 2)
        ax.imshow(gt_overlay)
        ax.set_title("Ground Truth", fontsize=9)
        ax.axis("off")

        # 從第三欄開始畫每個模型 prediction。
        for model_idx, model_name in enumerate(model_names):
            # 取得 prediction path 欄位。
            pred_path_col = f"{model_name}_Pred_Path"

            # 取得 Dice 欄位。
            dice_col = f"{model_name}_Dice"

            # 讀取 prediction mask。
            pred_np = read_binary_mask(row[pred_path_col])

            # 建立紅色 overlay。
            pred_overlay = overlay_mask(image_np, pred_np, color=(255, 0, 0), alpha=0.45)

            # 計算 subplot index。
            subplot_index = row_idx * n_cols + 3 + model_idx

            # 畫 prediction overlay。
            ax = plt.subplot(n_rows, n_cols, subplot_index)
            ax.imshow(pred_overlay)

            # 顯示模型名稱與 Dice。
            ax.set_title(f"{model_name}\nDice={row[dice_col] * 100:.2f}%", fontsize=9)
            ax.axis("off")

    # 自動排版。
    plt.tight_layout()

    # 設定輸出圖檔路徑。
    fig_path = OUT_DIR / "qualitative_c4_failure_analysis.png"

    # 儲存圖片。
    plt.savefig(fig_path, dpi=300)

    # 關閉 figure。
    plt.close()

    # 回傳圖片路徑。
    return fig_path


# ============================================================
# 10. Main
# ============================================================

def main():
    # 印出開始訊息。
    print("=" * 100)
    print("Qualitative Failure Analysis on C4")
    print("=" * 100)

    # 自動尋找各模型 test_predictions 資料夾。
    model_dirs = discover_model_prediction_dirs()

    # 如果找到的模型少於 2 個，提醒使用者。
    if len(model_dirs) < 2:
        raise RuntimeError("Less than two model prediction folders found. Please check your result paths.")

    # 建立 per-case metric table。
    metric_df = build_case_metric_table(model_dirs)

    # 輸出所有共同案例的 metric。
    all_case_csv = OUT_DIR / "qualitative_c4_all_common_cases_metrics.csv"
    metric_df.to_csv(all_case_csv, index=False, encoding="utf-8-sig")

    # 自動挑選代表性案例。
    selected_df = select_cases(metric_df)

    # 輸出選中的案例清單。
    selected_csv = OUT_DIR / "qualitative_c4_case_selection.csv"
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    # 產生 qualitative figure。
    fig_path = plot_qualitative_figure(selected_df, model_dirs)

    # 印出完成訊息。
    print("=" * 100)
    print("[DONE] Qualitative failure analysis finished.")
    print(f"[ALL CASE METRICS] {all_case_csv}")
    print(f"[SELECTED CASES] {selected_csv}")
    print(f"[FIGURE] {fig_path}")
    print("=" * 100)


# 如果直接執行這支程式，就進入 main。
if __name__ == "__main__":
    main()
