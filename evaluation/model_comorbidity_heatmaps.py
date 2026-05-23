import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

# ==================== 配置区 ====================
OUTPUT_DIR = Path("results/comorbidity_analysis")
FILE_GT = OUTPUT_DIR / "heatmap_correlation_values.csv"
FILE_VANILLA = OUTPUT_DIR / "vallian_heatmap_v2.csv"
FILE_KECODER = OUTPUT_DIR / "ke_coder_heatmap_v2.csv"

CORE_DISEASES = [
    "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Lung Opacity",
    "Edema", "Enlarged Cardiomediastinum", "Fracture", "Atelectasis"
]

def get_upper_triangle_values(file_path):
    df = pd.read_csv(file_path, index_col=0)
    df = df.loc[CORE_DISEASES, CORE_DISEASES]
    matrix = df.values
    upper_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_indices]
def main():
    vec_gt = get_upper_triangle_values(FILE_GT)
    vec_vanilla = get_upper_triangle_values(FILE_VANILLA)
    vec_kecoder = get_upper_triangle_values(FILE_KECODER)

    corr_vanilla, _ = pearsonr(vec_gt, vec_vanilla)
    corr_kecoder, _ = pearsonr(vec_gt, vec_kecoder)
    mse_vanilla = np.mean((vec_gt - vec_vanilla) ** 2)
    mse_kecoder = np.mean((vec_gt - vec_kecoder) ** 2)

    print("=" * 50)
    print("共病矩阵量化对比结果")
    print(f"Matrix Similarity (PCC) - Vanilla: {corr_vanilla:.4f} | KE-Coder: {corr_kecoder:.4f}")
    print(f"MSE - Vanilla: {mse_vanilla:.4f} | KE-Coder: {mse_kecoder:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()