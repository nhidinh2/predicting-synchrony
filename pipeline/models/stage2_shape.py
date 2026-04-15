"""Stage-2: disaggregate daily forecasts to 30-minute intervals via DOW profiles."""
from __future__ import annotations

import pandas as pd

from pipeline.config import INTERVALS_PER_DAY
from pipeline.features.profiles import build_dow_interval_profiles


class Stage2Shape:
    def __init__(self) -> None:
        self.profiles: dict[str, pd.DataFrame] = {}

    def fit(self, intervals: pd.DataFrame) -> "Stage2Shape":
        self.profiles["cv"] = build_dow_interval_profiles(intervals, "cv")
        self.profiles["cct"] = build_dow_interval_profiles(intervals, "cct")
        self.profiles["abd_calls"] = build_dow_interval_profiles(intervals, "abd_calls")
        return self

    def expand(self, daily: pd.DataFrame) -> pd.DataFrame:
        """daily has cv_daily, cct_daily, abd_calls_daily per (portfolio, date)."""
        df = daily.copy()
        df["dow"] = df["date"].dt.dayofweek

        grid = df.merge(
            pd.DataFrame({"interval_idx": range(INTERVALS_PER_DAY)}), how="cross"
        )

        cv_prof = self.profiles["cv"].rename(columns={"weight": "w_cv"})
        cct_prof = self.profiles["cct"].rename(columns={"weight": "w_cct"})
        abd_prof = self.profiles["abd_calls"].rename(columns={"weight": "w_abd"})

        grid = grid.merge(cv_prof, on=["portfolio", "dow", "interval_idx"], how="left")
        grid = grid.merge(cct_prof, on=["portfolio", "dow", "interval_idx"], how="left")
        grid = grid.merge(abd_prof, on=["portfolio", "dow", "interval_idx"], how="left")

        grid["cv"] = grid["cv_daily"] * grid["w_cv"] * INTERVALS_PER_DAY * grid["w_cv"].pipe(_renorm)
        grid["cv"] = grid["cv_daily"] * grid["w_cv"]
        grid["cct"] = grid["cct_daily"]  # CCT is a per-call average; carry daily through by DOW
        grid["cct"] = grid["cct_daily"] * (grid["w_cct"] * INTERVALS_PER_DAY)
        grid["abd_calls"] = grid["abd_calls_daily"] * grid["w_abd"]
        grid["abd_rate"] = grid["abd_calls"] / grid["cv"].clip(lower=1)

        return grid[
            ["portfolio", "date", "interval_idx", "cv", "cct", "abd_calls", "abd_rate"]
        ]


def _renorm(w: pd.Series) -> pd.Series:
    return w.fillna(0).clip(lower=0)
