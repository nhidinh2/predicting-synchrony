"""DOW × half-hour shape profiles for Stage-2 disaggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import INTERVALS_PER_DAY


def build_dow_interval_profiles(intervals: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return normalized profile: rows = (portfolio, dow, interval_idx), sums to 1 per (portfolio, dow)."""
    df = intervals.copy()
    df["dow"] = df["date"].dt.dayofweek

    agg = df.groupby(["portfolio", "dow", "interval_idx"])[metric].mean().reset_index()
    totals = agg.groupby(["portfolio", "dow"])[metric].transform("sum")
    agg["weight"] = np.where(totals > 0, agg[metric] / totals, 1.0 / INTERVALS_PER_DAY)
    return agg[["portfolio", "dow", "interval_idx", "weight"]]


def apply_profile(daily_totals: pd.DataFrame, profile: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Expand a daily forecast to interval level by multiplying with the profile."""
    df = daily_totals.copy()
    df["dow"] = df["date"].dt.dayofweek
    merged = df.merge(profile, on=["portfolio", "dow"], how="left")
    merged[metric] = merged[f"{metric}_daily"] * merged["weight"]
    return merged[["portfolio", "date", "interval_idx", metric]]
