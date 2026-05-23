import pandas as pd
import numpy as np
from pathlib import Path
from evaluation.metric_calculator import auc_micro, auc_macro, f1_micro, f1_macro, precision_at_k
from evaluation.comorbidity_matrix import build_comorbidity_matrix
from evaluation.structural_consistency_analysis import pcc, mse

BASE = Path(__file__).parent.parent
INPUT = BASE / "results/model_predictions/all_model_predictions.csv"
OUT = BASE / "results/evaluation"
OUT.mkdir(exist_ok=True)

DISEASES = [
    "Cardiomegaly", "Pneumothorax", "Pleural Effusion", "Edema",
    "Lung Opacity", "Fracture", "Atelectasis", "Enlarged Cardiomediastinum"
]

def parse(s):
    r = np.zeros(len(DISEASES))
    for i, d in enumerate(DISEASES):
        if d.lower() in s.lower():
            r[i] = 1
    return r

def score(s):
    r = np.zeros(len(DISEASES))
    for i, d in enumerate(DISEASES):
        r[i] = 0.9 if d.lower() in s.lower() else 0.1
    return r

def run():
    df = pd.read_csv(INPUT)
    y_true = df[DISEASES].values
    gold_mat = build_comorbidity_matrix(df[DISEASES])
    res = {}

    for m in ["vanilla_pred", "ke_coder_pred", "few_shot_pred", "disease_classification_pred"]:
        if m not in df.columns:
            continue
        y_pred = np.array([parse(x) for x in df[m]])
        y_score = np.array([score(x) for x in df[m]])
        res[m] = {
            "AUC-Micro": round(auc_micro(y_true, y_score), 4),
            "AUC-Macro": round(auc_macro(y_true, y_score), 4),
            "F1-Micro": round(f1_micro(y_true, y_pred), 4),
            "F1-Macro": round(f1_macro(y_true, y_pred), 4),
            "P@1": round(precision_at_k(y_true, y_score, 1), 4),
            "P@3": round(precision_at_k(y_true, y_score, 3), 4),
            "P@5": round(precision_at_k(y_true, y_score, 5), 4),
            "PCC": round(pcc(build_comorbidity_matrix(pd.DataFrame(y_pred)), gold_mat), 4),
            "MSE": round(mse(build_comorbidity_matrix(pd.DataFrame(y_pred)), gold_mat), 6)
        }

    final = pd.DataFrame(res).T
    final.to_csv(OUT / "results.csv")

if __name__ == "__main__":
    run()