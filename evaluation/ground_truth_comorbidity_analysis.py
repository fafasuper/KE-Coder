import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置区 ====================
INPUT_FILE = Path("data/processed/final_filtered_data_for_combid.csv")
OUTPUT_DIR = Path("results/comorbidity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_DISEASES = [
    "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Lung Opacity",
    "Edema", "Enlarged Cardiomediastinum", "Fracture", "Atelectasis"
]

OUTPUT_CLEAN_DATA = OUTPUT_DIR / "clean_ground_truth_data.csv"
OUTPUT_CORR_MATRIX = OUTPUT_DIR / "heatmap_correlation_values.csv"
OUTPUT_FIG_PEARSON = OUTPUT_DIR / "fig_ground_truth_pearson.png"
OUTPUT_FIG_COUNTS = OUTPUT_DIR / "fig_ground_truth_counts.png"

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"成功加载 {len(df)} 行数据")

    # 保存清爽版
    available_cols = [col for col in ['subject_id', 'study_id', 'mimic_iv_hadm_id',
                                      'report_text', 'discharge_summary_text', 'standardized_diagnoses'] + CORE_DISEASES
                      if col in df.columns]
    clean_df = df[available_cols].copy()
    clean_df.to_csv(OUTPUT_CLEAN_DATA, index=False)
    print(f"清爽数据集已保存: {OUTPUT_CLEAN_DATA}")

    matrix_gold = clean_df[CORE_DISEASES].fillna(0).astype(int)
    corr_matrix = matrix_gold.corr(method='pearson')
    corr_matrix.to_csv(OUTPUT_CORR_MATRIX)
    print(f"相关系数矩阵已保存: {OUTPUT_CORR_MATRIX}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r',
                vmin=-0.4, vmax=0.8, center=0, square=True, linewidths=0.5)

    p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), columns=CORE_DISEASES, index=CORE_DISEASES)
    for i, col1 in enumerate(CORE_DISEASES):
        for j, col2 in enumerate(CORE_DISEASES):
            if col1 != col2:
                _, p = pearsonr(matrix_gold[col1], matrix_gold[col2])
                p_matrix.iloc[i, j] = p
                if p < 0.05 and abs(corr_matrix.iloc[i, j]) > 0.1:
                    plt.text(j + 0.5, i + 0.3, "*", ha='center', va='center',
                             color='black', fontsize=10, fontweight='bold')

    plt.title('Ground Truth: Disease Co-occurrence Patterns\n(Test Set N={})'.format(len(df)), pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_PEARSON, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pearson热力图已保存: {OUTPUT_FIG_PEARSON}")

    cooccurrence_counts = matrix_gold.T.dot(matrix_gold)
    np.fill_diagonal(cooccurrence_counts.values, 0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_counts, annot=True, fmt="d", cmap='YlOrRd',
                square=True, linewidths=0.5)
    plt.title('Ground Truth: Absolute Co-occurrence Frequency', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG_COUNTS, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"共现频次图已保存: {OUTPUT_FIG_COUNTS}")

if __name__ == "__main__":
    main()