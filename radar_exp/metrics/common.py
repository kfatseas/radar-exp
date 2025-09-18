"""Common classification metrics.

Functions in this module wrap scikit‑learn metrics to compute
accuracy, F1 score and AUROC.  They handle absent probability
predictions gracefully by skipping AUROC when probabilities are
unavailable.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted class probabilities of shape `(n_samples, n_classes)`.
        If provided and the problem is binary classification, the ROC
        AUC score is computed using the positive class probabilities.

    Returns
    -------
    dict
        Dictionary containing `accuracy`, `f1` and optionally `auroc`.
    """
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro"))
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        # Use positive class probabilities
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            metrics["auroc"] = float(auc)
        except Exception:
            metrics["auroc"] = float("nan")
    return metrics