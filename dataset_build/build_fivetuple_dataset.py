import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE = Path(__file__).parent.parent
INPUT = BASE / "data/processed/final_filtered_data_for_combid.csv"
OUT_DIR = BASE / "data/processed_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "fivetuple_dataset.csv"
CORE_DISEASES = [
    "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Edema",
    "Lung Opacity", "Fracture", "Atelectasis", "Enlarged Cardiomediastinum"
]

def build():
    if not INPUT.exists():
        logging.error(f"找不到输入文件: {INPUT}。请先运行过滤脚本。")
        return
    df = pd.read_csv(INPUT)
    rows = []
    for _, row in df.iterrows():
        k_def = row.get("knowledge_def", "")
        k_rag = row.get("rag_context", "")
        k_def = "" if pd.isna(k_def) else k_def
        k_rag = "" if pd.isna(k_rag) else k_rag
        text = row.get("discharge_summary_text", "")
        if pd.isna(text) or text == "":
            text = row.get("report_text", "")
        quintuple = {
            "hadm_id": row.get("hadm_id", "unknown"),
            "split": row.get("split", "train"),  # 极其关键！保留数据集划分标签
            "I": "Predict 8 chest diseases from clinical note.",
            "K_def": k_def,
            "K_rag": k_rag,
            "D": text,
            "Y": [int(row[d]) if pd.notna(row.get(d)) else 0 for d in CORE_DISEASES]
        }
        rows.append(quintuple)

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUT_FILE, index=False)
    logging.info(f"✅ 五元组数据集已生成，共 {len(final_df)} 条记录。")
    logging.info(f"💾 结果已保存至 {OUT_FILE}")
    logging.info(f"📊 数据分布统计：\n{final_df['split'].value_counts()}")


if __name__ == "__main__":
    build()