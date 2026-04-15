"""Isotonic calibration of daily forecasts against holdout residuals."""
from __future__ import annotations

import pandas as pd
from sklearn.isotonic import IsotonicRegression


def isotonic_calibrate(
    pred: pd.Series, actual: pd.Series, new_pred: pd.Series
) -> pd.Series:
    """Fit y ~ isotonic(pred) on holdout, return calibrated predictions for new_pred."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred.values, actual.values)
    return pd.Series(iso.predict(new_pred.values), index=new_pred.index)
