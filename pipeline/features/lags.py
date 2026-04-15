"""Lag and rolling features per (portfolio, metric)."""
from __future__ import annotations

import pandas as pd

LAG_DAYS = (1, 7, 14, 28, 365)
ROLL_WINDOWS = (7, 28)


def add_lag_features(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values(["portfolio", "date"]).copy()
    for col in target_cols:
        g = out.groupby("portfolio")[col]
        for lag in LAG_DAYS:
            out[f"{col}_lag_{lag}"] = g.shift(lag)
        for w in ROLL_WINDOWS:
            out[f"{col}_roll_mean_{w}"] = g.shift(1).rolling(w).mean().reset_index(0, drop=True)
            out[f"{col}_roll_std_{w}"] = g.shift(1).rolling(w).std().reset_index(0, drop=True)
    return out
