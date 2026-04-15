"""Calendar features: DOW, month, holiday flags, holiday-adjacency."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    d = out["date"]
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["day_of_month"] = d.dt.day
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)

    hol = set(pd.to_datetime(holidays["date"]).dt.normalize())
    out["is_holiday"] = d.dt.normalize().isin(hol).astype(int)
    out["days_to_holiday"] = _days_to_nearest(d, hol, direction="forward")
    out["days_from_holiday"] = _days_to_nearest(d, hol, direction="backward")
    return out


def _days_to_nearest(dates: pd.Series, holidays: set, direction: str) -> pd.Series:
    sorted_hol = np.array(sorted(pd.Timestamp(h).value for h in holidays))
    if len(sorted_hol) == 0:
        return pd.Series(np.full(len(dates), 365), index=dates.index)
    vals = dates.values.astype("datetime64[ns]").astype(np.int64)
    if direction == "forward":
        idx = np.searchsorted(sorted_hol, vals, side="left")
        idx = np.clip(idx, 0, len(sorted_hol) - 1)
        diff = (sorted_hol[idx] - vals) / (1e9 * 86400)
    else:
        idx = np.searchsorted(sorted_hol, vals, side="right") - 1
        idx = np.clip(idx, 0, len(sorted_hol) - 1)
        diff = (vals - sorted_hol[idx]) / (1e9 * 86400)
    return pd.Series(np.clip(diff, 0, 30).astype(int), index=dates.index)
