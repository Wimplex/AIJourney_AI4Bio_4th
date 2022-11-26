from typing import Iterable

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    roc_curve,
)


def classification_metrics(trues: Iterable, preds: Iterable, probs: Iterable = None, labels: Iterable = None):
    metrics = {
        "F1": f1_score(trues, preds, average="macro"),
        "Accuracy": accuracy_score(trues, preds),
        "Precision": precision_score(trues, preds, average="macro"),
        "Recall": recall_score(trues, preds, average="macro"),
    }
    if probs is not None:
        metrics.update({
            "ROC-AUC OneVsOne": roc_auc_score(
                trues.squeeze(), probs.squeeze(), average="macro", multi_class="ovo", labels=labels),
            "ROC-AUC OneVsRest": roc_auc_score(
                trues.squeeze(), probs.squeeze(), average="macro", multi_class="ovr", labels=labels)
        })
    return metrics


def eer_precise(trues: Iterable, scores: Iterable):
    fpr, tpr, thr = roc_curve(trues, scores)
    fnr = 1 - tpr
    fpr_interp = interp1d(thr, fpr)
    fnr_interp = interp1d(thr, fnr)
    new_thr = np.linspace(np.min(thr), np.max(thr), 1000)
    new_fpr = fpr_interp(new_thr)
    new_fnr = fnr_interp(new_thr)
    min_idx = np.argmin(np.abs(new_fpr - new_fnr))
    eer_thr = new_thr[min_idx]
    eer = (new_fpr[min_idx] + new_fnr[min_idx]) / 2
    return eer, eer_thr


def fr_point(trues: Iterable, scores: Iterable, fr: float):
    fpr, _, thr = roc_curve(trues, scores)
    interp = interp1d(thr, fpr)
    new_thr = np.linspace(np.min(thr), np.max(thr), 1000)
    new_fpr = interp(new_thr)

    idx = np.argmin(np.abs(new_fpr - fr))
    return new_thr[idx]


def fa_point(trues: Iterable, scores: Iterable, fa: float):
    _, tpr, thr = roc_curve(trues, scores)
    fnr = 1 - tpr
    fnr_interp = interp1d(thr, fnr)
    new_thr = np.linspace(np.min(thr), np.max(thr), 1000)
    new_fnr = fnr_interp(new_thr)

    idx = np.argmin(np.abs(new_fnr - fa))
    return new_thr[idx]
