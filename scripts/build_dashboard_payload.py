"""Convert an existing submission CSV into dashboard.json for the React UI.

Useful when you want to render the dashboard without re-running the full pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from pipeline.config import EXPORTS_DIR, PORTFOLIOS

MONTH_LOOKUP = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"], start=1)}


@click.command()
@click.option("--submission", default=str(EXPORTS_DIR / "submission.csv"))
@click.option("--out", default=str(EXPORTS_DIR / "dashboard.json"))
@click.option("--year", default=2025, type=int)
def main(submission: str, out: str, year: int):
    df = pd.read_csv(submission)
    df["month_num"] = df["Month"].map(MONTH_LOOKUP)
    df["date"] = pd.to_datetime(dict(year=year, month=df["month_num"], day=df["Day"]))
    df["interval_idx"] = df["Interval"].map(_time_to_idx)

    intervals: list[dict] = []
    daily_rows: list[dict] = []

    for p in PORTFOLIOS:
        sub = df[[
            "date", "interval_idx",
            f"Calls_Offered_{p}", f"Abandoned_Calls_{p}",
            f"Abandoned_Rate_{p}", f"CCT_{p}",
        ]].rename(columns={
            f"Calls_Offered_{p}": "cv",
            f"Abandoned_Calls_{p}": "abd_calls",
            f"Abandoned_Rate_{p}": "abd_rate",
            f"CCT_{p}": "cct",
        }).copy()
        sub["portfolio"] = p
        sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
        intervals.extend(sub.to_dict(orient="records"))

        daily = sub.groupby("date").agg(
            cv_daily=("cv", "sum"),
            abd_calls_daily=("abd_calls", "sum"),
            cct_daily=("cct", "mean"),
        ).reset_index()
        daily["portfolio"] = p
        daily["abd_rate_daily"] = daily["abd_calls_daily"] / daily["cv_daily"].clip(lower=1)
        daily_rows.extend(daily.to_dict(orient="records"))

    payload = {
        "meta": {
            "portfolios": list(PORTFOLIOS),
            "forecast_month": "August 2025",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "scores": {"note": "derived from existing submission.csv — run `make backtest` for holdout scores"},
        "intervals": intervals,
        "daily": daily_rows,
        "residuals": [],
    }

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, default=str)
    click.echo(f"wrote {out}  ({len(intervals):,} interval rows, {len(daily_rows)} daily rows)")


def _time_to_idx(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 2 + (int(m) // 30)


if __name__ == "__main__":
    main()
