"""Wrappers around sklearn classifiers.

This module exposes a simple interface for fitting a classifier on a
feature matrix and predicting labels or probabilities.  Supported
models include logistic regression, support vector machines and random
forests.  The intention is not to be exhaustive but to provide
baseline models for quick experimentation.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logistic",
    params: Dict[str, Any] | None = None,
) -> Any:
    """Train a classifier on the provided data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape `(n_samples, n_features)`.
    y : np.ndarray
        Label vector of shape `(n_samples,)`.
    model_type : str, optional
        Type of classifier to train.  One of `'logistic'`, `'svm'`
        (support vector machine with RBF kernel) or `'random_forest'`.
    params : dict, optional
        Additional hyper‑parameters passed to the underlying sklearn
        constructor.

    Returns
    -------
    model
        A fitted sklearn classifier.
    """
    if params is None:
        params = {}
    model_type = model_type.lower()
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, **params)
    elif model_type == "svm":
        # Use probability=True to enable probability predictions
        model = SVC(probability=True, **params)
    elif model_type in {"rf", "random_forest", "randomforest"}:
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.fit(X, y)
    return model


def predict(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    """Predict labels and optionally probabilities using a fitted model.

    Parameters
    ----------
    model : object
        Fitted sklearn classifier.
    X : np.ndarray
        Feature matrix for which to predict labels.

    Returns
    -------
    tuple
        `(y_pred, y_prob)` where `y_pred` is an array of predicted
        labels.  If the model supports probability estimates then
        `y_prob` is an array of shape `(n_samples, n_classes)` with
        probabilities; otherwise it is `None`.
    """
    y_pred = model.predict(X)
    y_prob = None
    # Check if the model exposes predict_proba
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
        except Exception:
            y_prob = None
    return y_pred, y_prob