"""Leave-month-out backtesting harness."""
from __future__ import annotations

import pandas as pd


def leave_month_out(
    daily: pd.DataFrame, holdout_month: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily frame into (train, holdout) on a YYYY-MM boundary."""
    holdout_ts = pd.Timestamp(holdout_month)
    mask = (daily["date"].dt.year == holdout_ts.year) & (
        daily["date"].dt.month == holdout_ts.month
    )
    return daily.loc[~mask].copy(), daily.loc[mask].copy()
