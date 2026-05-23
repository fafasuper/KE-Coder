import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def auc_micro(y_true, y_score):
    return roc_auc_score(y_true, y_score, average='micro')

def auc_macro(y_true, y_score):
    return roc_auc_score(y_true, y_score, average='macro')

def f1_micro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro', zero_division=0)

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def precision_at_k(y_true, y_score, k):
    total = 0.0
    n_sample = len(y_true)

    for i in range(n_sample):
        true_labels = y_true[i]
        scores = y_score[i]
        topk_indices = np.argsort(scores)[::-1][:k]
        hits = 0
        for idx in topk_indices:
            if true_labels[idx] == 1:
                hits += 1
        total += hits / k

    return total / n_sample