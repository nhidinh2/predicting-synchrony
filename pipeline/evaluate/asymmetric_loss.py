"""Asymmetric loss: underprediction penalized more than overprediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import LOSS


def asymmetric_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_weight: float = LOSS.under_weight,
    over_weight: float = LOSS.over_weight,
) -> float:
    err = y_true - y_pred
    weighted = np.where(err > 0, under_weight * err, -over_weight * err)
    return float(np.mean(weighted))


def asymmetric_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_weight: float = LOSS.under_weight,
    over_weight: float = LOSS.over_weight,
    eps: float = 1.0,
) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    err = (y_true - y_pred) / denom
    weighted = np.where(err > 0, under_weight * err, -over_weight * err)
    return float(np.mean(weighted))


def score_submission(actual: pd.DataFrame, predicted: pd.DataFrame) -> dict:
    """Compute per-metric asymmetric MAPE on merged frame."""
    m = actual.merge(
        predicted, on=["portfolio", "date", "interval_idx"], suffixes=("_true", "_pred")
    )
    return {
        "cv": asymmetric_mape(m["cv_true"].values, m["cv_pred"].values),
        "cct": asymmetric_mape(m["cct_true"].values, m["cct_pred"].values),
        "abd_rate": asymmetric_mape(m["abd_rate_true"].values, m["abd_rate_pred"].values),
    }
