import pandas as pd
import ast
from pathlib import Path
import logging

# ==================== 配置区 ====================
INPUT_CSV = Path("data/processed/preprocessed_complete_dataset.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILTERED = PROCESSED_DIR / "final_filtered_data_for_combid.csv"

# 8个核心疾病标签
CORE_DISEASES = [
    "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Lung Opacity",
    "Edema", "Enlarged Cardiomediastinum", "Fracture", "Atelectasis"
]

# 所有19个疾病标签（用于识别非疾病列）
ALL_DISEASE_COLS = [
    "No Finding", "Lung Opacity", "Cardiomegaly", "Atelectasis", "Pleural Effusion",
    "Support Devices", "Edema", "Pneumonia", "Pneumothorax", "Lung Lesion",
    "Fracture", "Enlarged Cardiomediastinum", "Consolidation", "Pleural Other",
    "Calcification of the Aorta", "Tortuous Aorta", "Pneumoperitoneum",
    "Subcutaneous Emphysema", "Pneumomediastinum"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_all_icd9(diagnoses_data):
    if pd.isna(diagnoses_data):
        return False
    try:
        if isinstance(diagnoses_data, str):
            diagnoses_list = ast.literal_eval(
                diagnoses_data.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            )
        else:
            diagnoses_list = diagnoses_data
        if not isinstance(diagnoses_list, list) or len(diagnoses_list) == 0:
            return False
        return all(item.get("icd_version") == 9 for item in diagnoses_list)
    except:
        return False

def simplify_icd_codes(diagnoses_data):
    if pd.isna(diagnoses_data):
        return ""
    try:
        if isinstance(diagnoses_data, str):
            diagnoses_list = ast.literal_eval(
                diagnoses_data.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            )
        else:
            diagnoses_list = diagnoses_data
        icd_codes = [item.get("icd_code", "").strip() for item in diagnoses_list if item.get("icd_code", "").strip()]
        return ",".join(icd_codes)
    except:
        return ""

def main():
    if not INPUT_CSV.exists():
        logging.error(f"输入文件不存在: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    total_rows = len(df)
    print(f"原始数据总行数：{total_rows}")

    # 筛选
    mask_test = df["split"] == "test"
    mask_icd9 = df["standardized_diagnoses"].apply(is_all_icd9)
    df_filtered = df[mask_test & mask_icd9].copy()
    print(f"筛选后数据行数：{len(df_filtered)} ({len(df_filtered) / total_rows:.2%})")

    # 简化ICD列
    df_filtered["standardized_diagnoses"] = df_filtered["standardized_diagnoses"].apply(simplify_icd_codes)

    # 保留列
    non_disease_cols = [col for col in df_filtered.columns if col not in ALL_DISEASE_COLS]
    final_keep_cols = non_disease_cols + CORE_DISEASES
    df_final = df_filtered[final_keep_cols].copy()

    df_final.to_csv(OUTPUT_FILTERED, index=False, encoding="utf-8")
    print(f"最终保留列数：{len(df_final.columns)}")
    print(f"结果已保存至: {OUTPUT_FILTERED}")
    print("\n保留的疾病标签：", ", ".join(CORE_DISEASES))

if __name__ == "__main__":
    main()