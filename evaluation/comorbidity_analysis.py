import pandas as pd
import ast
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import logging
import os

# ==================== 配置区 ====================
PROCESSED_DATA = Path("data/processed/preprocessed_complete_dataset.csv")
ICD_CCS_MAP = Path("data/raw/allicd_standardized_ccs_mapped_with_desc.csv")
OUTPUT_DIR = Path("results/cooccurrence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_LABEL_COLUMNS = [
    'No Finding', 'Lung Opacity', 'Cardiomegaly', 'Atelectasis', 'Pleural Effusion',
    'Support Devices', 'Edema', 'Pneumonia', 'Pneumothorax', 'Lung Lesion',
    'Fracture', 'Enlarged Cardiomediastinum', 'Consolidation', 'Pleural Other',
    'Calcification of the Aorta', 'Tortuous Aorta', 'Pneumoperitoneum',
    'Subcutaneous Emphysema', 'Pneumomediastinum'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ===============================================

def save_cooccurrence_to_csv(cooccurrence_dict, filename_prefix, description):
    if not cooccurrence_dict:
        print(f"Warning: No data for {description}")
        return
    data_for_df = [{'Source': src, 'Target': tgt, 'Count': cnt} for (src, tgt), cnt in cooccurrence_dict.items()]
    df = pd.DataFrame(data_for_df)
    df_sorted = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    filepath = OUTPUT_DIR / f"{filename_prefix}_cooccurrence.csv"
    df_sorted.to_csv(filepath, index=False)
    print(f"✅ 已保存 {description} 到 {filepath}")
    print(df_sorted.head(10))


def plot_cooccurrence_network(cooccurrence_dict, title, filename_suffix, top_n_edges=50):
    if not cooccurrence_dict:
        return
    edges = [(src, tgt, cnt) for (src, tgt), cnt in cooccurrence_dict.items()]
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)[:top_n_edges]
    G = nx.Graph()
    for u, v, w in edges_sorted:
        G.add_edge(u, v, weight=w)
    plt.figure(figsize=(20, 14))
    pos = nx.spring_layout(G, k=0.8 / (G.number_of_nodes() ** 0.5), iterations=100)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue')
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    nx.draw_networkx_edges(G, pos, width=[w / max_w * 8 + 0.5 for w in edge_weights], alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(title)
    plt.axis('off')
    plt.savefig(OUTPUT_DIR / f"{filename_suffix}_network.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    patient_df = pd.read_csv(PROCESSED_DATA)
    icd_ccs_df = pd.read_csv(ICD_CCS_MAP)
    icd_to_ccs_map = dict(zip(icd_ccs_df['standardized_icd_code'].astype(str), icd_ccs_df['ccs_category_description']))

    patient_df['parsed_icd_codes'] = patient_df['standardized_diagnoses'].apply(
        lambda x: [d['icd_code'] for d in ast.literal_eval(x)] if pd.notna(x) and str(x).strip() else []
    )

    # Visit-Level
    visit_icd_co = defaultdict(int)
    visit_ccs_co = defaultdict(int)
    for _, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="Visit-Level"):
        icds = set(row['parsed_icd_codes'])
        ccs_set = {icd_to_ccs_map.get(str(i), "Unknown_CCS") for i in icds}
        active_labels = [lbl for lbl in DISEASE_LABEL_COLUMNS if row[lbl] == 1]
        for icd in icds:
            for lbl in active_labels:
                visit_icd_co[(icd, lbl)] += 1
        for ccs in ccs_set:
            for lbl in active_labels:
                visit_ccs_co[(ccs, lbl)] += 1

    # Patient-Level
    patient_agg = defaultdict(lambda: {'icds': set(), 'labels': set()})
    for _, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="Patient Aggregation"):
        sid = row['subject_id']
        for icd in row['parsed_icd_codes']:
            patient_agg[sid]['icds'].add(icd)
        for lbl in DISEASE_LABEL_COLUMNS:
            if row[lbl] == 1:
                patient_agg[sid]['labels'].add(lbl)

    patient_icd_co = defaultdict(int)
    patient_ccs_co = defaultdict(int)
    for data in tqdm(patient_agg.values(), desc="Patient-Level"):
        icds = data['icds']
        labels = data['labels']
        ccs_set = {icd_to_ccs_map.get(str(i), "Unknown_CCS") for i in icds}
        for icd in icds:
            for lbl in labels:
                patient_icd_co[(icd, lbl)] += 1
        for ccs in ccs_set:
            for lbl in labels:
                patient_ccs_co[(ccs, lbl)] += 1

    # 保存与可视化
    save_cooccurrence_to_csv(visit_icd_co, "visit_level_icd_disease", "Visit-Level ICD-Disease")
    save_cooccurrence_to_csv(visit_ccs_co, "visit_level_ccs_disease", "Visit-Level CCS-Disease")
    save_cooccurrence_to_csv(patient_icd_co, "patient_level_icd_disease", "Patient-Level ICD-Disease")
    save_cooccurrence_to_csv(patient_ccs_co, "patient_level_ccs_disease", "Patient-Level CCS-Disease")

    plot_cooccurrence_network(visit_icd_co, "Visit-Level ICD-Disease Network", "visit_level_icd_disease")
    plot_cooccurrence_network(patient_icd_co, "Patient-Level ICD-Disease Network", "patient_level_icd_disease")

    logging.info(f"✅ 基础共现分析完成，结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()