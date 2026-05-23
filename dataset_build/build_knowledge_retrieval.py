import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE = Path(__file__).parent.parent
INPUT = BASE / "data/processed/final_filtered_data_for_combid.csv"
OUT_FILE = BASE / "data/processed/knowledge_enriched_data.csv"
CORE_DISEASES = [
    "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Edema",
    "Lung Opacity", "Fracture", "Atelectasis", "Enlarged Cardiomediastinum"
]

DISEASE_DEFINITIONS = {
    "Cardiomegaly": "Cardiomegaly is a medical condition in which the heart is enlarged, often indicated by a cardiothoracic ratio > 0.5 on a PA chest X-ray.",
    "Pneumothorax": "Pneumothorax is an abnormal collection of air in the pleural space between the lung and the chest wall, potentially causing lung collapse.",
    "Pleural Effusion": "Pleural effusion is an unusual amount of fluid around the lung, gathering in the pleura which can impair breathing.",
    "Edema": "Pulmonary edema is fluid accumulation in the tissue and air spaces of the lungs, often secondary to congestive heart failure.",
    "Lung Opacity": "Lung opacity represents any area in the chest radiograph that is more opaque (whiter) than it should be, masking the underlying air.",
    "Fracture": "A bone fracture in the thoracic region, typically involving the ribs, clavicle, or spine, identifiable by cortical discontinuities.",
    "Atelectasis": "Atelectasis is the collapse or closure of a lung resulting in reduced or absent gas exchange.",
    "Enlarged Cardiomediastinum": "An enlargement of the cardiomediastinal silhouette, which may indicate abnormalities in the heart or mediastinal structures."
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_EXEMPLARS = 4

def get_clinical_text(row):
    """向下兼容获取文本"""
    text = row.get("discharge_summary_text", "")
    if pd.isna(text) or text == "":
        text = row.get("report_text", "")
    return str(text)

def build_disease_labels_string(row):
    present = [d for d in CORE_DISEASES if row.get(d) == 1]
    return ", ".join(present) if present else "No Finding"

def main():
    if not INPUT.exists():
        logging.error(f"找不到输入文件: {INPUT}")
        return

    df = pd.read_csv(INPUT)
    logging.info(f"加载数据成功，共 {len(df)} 条。")
    df['working_text'] = df.apply(get_clinical_text, axis=1)
    logging.info("正在注入医学知识定义 (K_def)...")
    k_def_list = []
    for text in df['working_text']:
        text_lower = text.lower()
        matched_defs = []
        # 简单关键字匹配：只要文本中出现了疾病词（或同义词），就把定义加进去供大模型参考
        for disease, definition in DISEASE_DEFINITIONS.items():
            if disease.lower() in text_lower:
                matched_defs.append(f"- {disease}: {definition}")
        # 如果什么都没匹配到，为了防止空白，提供所有核心疾病的列表
        if not matched_defs:
            k_def = "Target diseases to evaluate: " + ", ".join(CORE_DISEASES)
        else:
            k_def = "\n".join(matched_defs)
        k_def_list.append(k_def)

    df['knowledge_def'] = k_def_list
    logging.info(f"加载 Embedding 模型 ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    train_df = df[df['split'] == 'train'].reset_index()
    if len(train_df) == 0:
        logging.error("未找到 train 数据集，无法构建 RAG 向量库！")
        return

    logging.info(f"正在向量化 {len(train_df)} 条训练集病例库...")
    train_embeddings = model.encode(train_df['working_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    dimension = train_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(train_embeddings)
    index.add(train_embeddings)

    logging.info("正在检索相似病例...")
    all_embeddings = model.encode(df['working_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(all_embeddings)
    D, I = index.search(all_embeddings, TOP_K_EXEMPLARS + 1)

    k_rag_list = []
    for i, row in df.iterrows():
        is_train = (row['split'] == 'train')
        retrieved_indices = I[i]

        exemplars = []
        for idx in retrieved_indices:
            train_row = train_df.iloc[idx]
            # 必须排除自己：如果当前样本是训练集，且召回的样本和自己hadm_id相同，则跳过
            if is_train and train_row['hadm_id'] == row['hadm_id']:
                continue
            exemplar_text = train_row['working_text'][:600]  # 截断防止 Prompt 超长 (OOM保护)
            exemplar_labels = build_disease_labels_string(train_row)
            exemplars.append(f"Case Snippet: {exemplar_text}...\nDiagnoses: {exemplar_labels}\n")

            if len(exemplars) == TOP_K_EXEMPLARS:
                break
        k_rag_list.append("\n".join(exemplars))

    df['rag_context'] = k_rag_list
    df = df.drop(columns=['working_text'])
    df.to_csv(OUT_FILE, index=False)
    logging.info(f"✅ 知识融合完毕！输出文件已保存至: {OUT_FILE}")

if __name__ == "__main__":
    main()