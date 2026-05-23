import csv
import ast
import pandas as pd
import json
import numpy as np
from pathlib import Path
import logging

# ==================== 配置区 ====================
INPUT_CSV = Path("data/processed_data/chest_disease_patients_with_miccai_and_chexpert_labels.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMPLETE = PROCESSED_DIR / "preprocessed_complete_dataset.csv"
OUTPUT_PARTIAL = PROCESSED_DIR / "preprocessed_partial_dataset.csv"

# 19个MICCAI疾病标签
DISEASE_LABELS = [
    'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis',
    'Pleural Effusion', 'Support Devices', 'Edema', 'Pneumonia',
    'Pneumothorax', 'Lung Lesion', 'Fracture', 'Enlarged Cardiomediastinum',
    'Consolidation', 'Pleural Other', 'Calcification of the Aorta',
    'Tortuous Aorta', 'Pneumoperitoneum', 'Subcutaneous Emphysema',
    'Pneumomediastinum'
]

ESSENTIAL_FIELDS = [
    'subject_id', 'study_id', 'split',
    'report_text', 'discharge_summary_text',
    'mimic_iv_diagnoses', 'mimic_iv_hadm_id', 'study_date'
]

OPTIONAL_FIELDS = ['image_paths']
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# ===============================================

def read_csv_fields_and_row(csv_path: Path, row_index: int = 0):
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            fields = csv_reader.fieldnames
            logging.info(f"CSV字段总数: {len(fields)}")
            print("字段列表:", fields)
            for idx, row in enumerate(csv_reader):
                if idx == row_index:
                    print(f"\n第{row_index+1}行样本:")
                    for k, v in list(row.items())[:10]:
                        val_str = str(v)
                        print(f"  {k}: {val_str[:100]}..." if len(val_str)>100 else f"  {k}: {val_str}")
                    break
    except Exception as e:
        logging.error(f"读取CSV失败: {e}")

def load_and_inspect_data(file_path: Path):
    logging.info("正在加载数据...")
    df = pd.read_csv(file_path)
    logging.info(f"数据规模: {len(df):,} 行 × {len(df.columns)} 列")
    print(f"📊 数据类型分布:\n{df.dtypes.value_counts()}")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"❓ 缺失值字段: {len(missing_cols)}")
        for col, cnt in missing_cols.items():
            print(f"  {col}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    else:
        print("✅ 无缺失值")
    return df

def standardize_icd_codes(diagnoses_str):
    if pd.isna(diagnoses_str) or diagnoses_str == '':
        return []
    try:
        if isinstance(diagnoses_str, str):
            diagnoses_list = ast.literal_eval(diagnoses_str)
        else:
            diagnoses_list = diagnoses_str
        if not isinstance(diagnoses_list, list):
            return []
        standardized = []
        for d in diagnoses_list:
            if isinstance(d, dict) and 'icd_code' in d:
                code = str(d['icd_code'])
                version = d.get('icd_version', 9)
                if version == 9:
                    cleaned = code.upper() if code.startswith(('V', 'E')) else code.lstrip('0') or code
                else:
                    cleaned = code.upper()
                standardized.append({'icd_code': cleaned, 'icd_version': version, 'original_code': code})
        return standardized
    except Exception as e:
        logging.warning(f"ICD解析失败: {str(e)[:100]}")
        return []

def process_icd_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['original_diagnoses'] = df['mimic_iv_diagnoses'].copy()
    df['standardized_diagnoses'] = df['mimic_iv_diagnoses'].apply(standardize_icd_codes)
    logging.info(f"ICD标准化完成，总编码数: {df['standardized_diagnoses'].apply(len).sum():,}")
    return df

def standardize_disease_labels(df: pd.DataFrame, disease_labels):
    df = df.copy()
    for label in disease_labels:
        if label in df.columns:
            def to_binary(v):
                if pd.isna(v):
                    return None
                return 1 if v in [1, 1.0, '1', True] else 0 if v in [0, 0.0, '0', False] else None
            df[label] = df[label].apply(to_binary).astype('Int64')
    return df

def create_clean_datasets(df: pd.DataFrame, essential_fields, disease_labels, optional_fields):
    fields_to_keep = essential_fields.copy()
    existing_labels = [lbl for lbl in disease_labels if lbl in df.columns]
    fields_to_keep.extend(existing_labels)
    for f in optional_fields:
        if f in df.columns:
            fields_to_keep.append(f)
    if 'standardized_diagnoses' in df.columns:
        fields_to_keep.append('standardized_diagnoses')
    clean_df = df[fields_to_keep].copy()
    logging.info(f"最终清理数据集: {clean_df.shape[0]:,} 行 × {clean_df.shape[1]} 列")
    return clean_df, fields_to_keep

def save_preprocessing_results(clean_complete, clean_partial, label_stats, completeness_stats, final_fields, output_prefix='preprocessed'):
    print("💾 保存预处理结果...")
    complete_file = PROCESSED_DIR / f'{output_prefix}_complete_dataset.csv'
    clean_complete.to_csv(complete_file, index=False)
    print(f"✅ 完整数据集已保存: {complete_file}")

    partial_file = None
    if clean_partial is not None and len(clean_partial) > 0:
        partial_file = PROCESSED_DIR / f'{output_prefix}_partial_dataset.csv'
        clean_partial.to_csv(partial_file, index=False)
        print(f"✅ 部分数据集已保存: {partial_file}")

    # 简化报告
    report = {
        'preprocessing_summary': {
            'complete_shape': [int(clean_complete.shape[0]), int(clean_complete.shape[1])],
            'retention_rate': float(len(clean_complete) / len(clean_complete)) if 'df_original' in globals() else 1.0
        },
        'final_fields': final_fields
    }
    report_file = PROCESSED_DIR / f'{output_prefix}_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ 报告已保存: {report_file}")
    return report

def main():
    if not INPUT_CSV.exists():
        logging.error(f"输入文件不存在: {INPUT_CSV}")
        return
    # read_csv_fields_and_row(INPUT_CSV)  # 调试时取消注释
    df = load_and_inspect_data(INPUT_CSV)
    df = process_icd_column(df)
    df = standardize_disease_labels(df, DISEASE_LABELS)
    clean_complete, final_fields = create_clean_datasets(df, ESSENTIAL_FIELDS, DISEASE_LABELS, OPTIONAL_FIELDS)
    save_preprocessing_results(clean_complete, None, {}, [], final_fields)
    logging.info("✅ 预处理完成！")

if __name__ == "__main__":
    main()