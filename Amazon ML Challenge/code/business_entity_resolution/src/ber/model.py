"""LightGBM pairwise matcher (MIT-licensed, well under 8B parameters).

Trains a binary classifier that scores each candidate pair with P(match). Class
imbalance (matches are rare) is handled by the decision threshold in resolve.py,
tuned for the precision-heavy F0.5 objective, rather than by oversampling.
"""
from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation


def train_matcher(cfg, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: Optional[np.ndarray] = None,
                  y_val: Optional[np.ndarray] = None) -> LGBMClassifier:
    model = LGBMClassifier(**cfg.lgbm_params)
    fit_kwargs = {}
    if X_val is not None and len(X_val) and len(np.unique(y_val)) > 1:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        # log-loss keeps improving as probabilities sharpen, so early stopping does
        # not halt the moment classes separate (unlike a ranking metric such as AP).
        fit_kwargs["eval_metric"] = "binary_logloss"
        fit_kwargs["callbacks"] = [
            early_stopping(cfg.early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def predict_scores(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def feature_importance(model: LGBMClassifier, names: List[str]) -> List[tuple]:
    imp = model.feature_importances_
    return sorted(zip(names, imp), key=lambda t: -t[1])


def save_model(path: str, model, threshold: float, feature_names: List[str],
               extra: Optional[dict] = None) -> None:
    payload = {"model": model, "threshold": threshold, "feature_names": feature_names}
    if extra:
        payload["extra"] = extra
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def load_model(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
