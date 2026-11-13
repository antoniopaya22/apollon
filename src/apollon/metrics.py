"""Detection metrics (attack class = 1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score

__all__ = ["evaluate", "format_metrics"]

ATTACK = 1


def evaluate(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Metrics of a single prediction.

    Returns accuracy, detection rate (recall on attacks), F1 on attacks, false positive
    rate on benign traffic and ROC AUC (from ``y_score`` if given, else from the labels).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "detection_rate": float(recall_score(y_true, y_pred, pos_label=ATTACK, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=ATTACK, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
    }
    if len(np.unique(y_true)) == 2:
        score = y_pred if y_score is None else np.asarray(y_score, dtype=float).ravel()
        out["roc_auc"] = float(roc_auc_score(y_true, score))
    return out


def format_metrics(m: dict[str, float]) -> str:
    """One metric per line, aligned."""
    w = max(map(len, m))
    return "\n".join(f"{k:<{w}}  {v:.4f}" for k, v in m.items())
