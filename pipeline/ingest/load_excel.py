"""Read the raw Excel workbook into tidy long-format frames."""
from __future__ import annotations

import pandas as pd

from pipeline.config import PORTFOLIOS, RAW_WORKBOOK


def load_daily(path=RAW_WORKBOOK) -> pd.DataFrame:
    """Return daily totals in long format: date, portfolio, cv, cct, abd_calls, abd_rate."""
    frames = []
    for p in PORTFOLIOS:
        df = pd.read_excel(path, sheet_name=f"Daily_{p}")
        df = df.rename(columns=str.lower)
        df["portfolio"] = p
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["abd_rate"] = out["abd_calls"] / out["cv"].clip(lower=1)
    return out.sort_values(["portfolio", "date"]).reset_index(drop=True)


def load_intervals(path=RAW_WORKBOOK) -> pd.DataFrame:
    """Return 30-minute interval observations for Apr–Jun 2025."""
    frames = []
    for p in PORTFOLIOS:
        df = pd.read_excel(path, sheet_name=f"Interval_{p}")
        df = df.rename(columns=str.lower)
        df["portfolio"] = p
        df["date"] = pd.to_datetime(df["date"])
        df["interval_idx"] = _interval_to_idx(df["interval"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _interval_to_idx(s: pd.Series) -> pd.Series:
    """Convert '0:00', '0:30', ..., '23:30' → 0..47."""
    parts = s.astype(str).str.split(":", expand=True).astype(int)
    return parts[0] * 2 + (parts[1] // 30)
