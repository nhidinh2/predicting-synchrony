"""Write the wide forecast CSV matching the reporting template."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import EXPORTS_DIR, INTERVALS_PER_DAY, PORTFOLIOS


def write_submission_csv(intervals: pd.DataFrame, path: Path | None = None) -> Path:
    """`intervals` has long-format portfolio/date/interval_idx/cv/cct/abd_calls/abd_rate."""
    path = path or (EXPORTS_DIR / "submission.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df = intervals.copy()
    df["Month"] = df["date"].dt.strftime("%B")
    df["Day"] = df["date"].dt.day
    df["Interval"] = df["interval_idx"].map(_idx_to_label)

    wide = df.pivot_table(
        index=["Month", "Day", "Interval"],
        columns="portfolio",
        values=["cv", "abd_calls", "abd_rate", "cct"],
    )
    wide.columns = [f"{_col_rename(a)}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    ordered = ["Month", "Day", "Interval"]
    for p in PORTFOLIOS:
        ordered += [f"Calls_Offered_{p}", f"Abandoned_Calls_{p}", f"Abandoned_Rate_{p}", f"CCT_{p}"]
    wide = wide.reindex(columns=ordered)

    wide["_ord"] = wide["Interval"].map({_idx_to_label(i): i for i in range(INTERVALS_PER_DAY)})
    wide = wide.sort_values(["Day", "_ord"]).drop(columns="_ord")
    wide.to_csv(path, index=False)
    return path


def _idx_to_label(i: int) -> str:
    return f"{i // 2}:{'00' if i % 2 == 0 else '30'}"


def _col_rename(metric: str) -> str:
    return {
        "cv": "Calls_Offered",
        "abd_calls": "Abandoned_Calls",
        "abd_rate": "Abandoned_Rate",
        "cct": "CCT",
    }[metric]
