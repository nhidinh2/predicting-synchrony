"""End-to-end pipeline entrypoint: ingest → features → fit → forecast → export."""
from __future__ import annotations

import click
import pandas as pd

from pipeline.config import (
    EXPORTS_DIR,
    FORECAST_DAYS,
    FORECAST_MONTH,
    INTERVAL_OBS_END,
    INTERVAL_OBS_START,
    PORTFOLIOS,
)
from pipeline.evaluate import score_submission
from pipeline.export import write_dashboard_payload, write_submission_csv
from pipeline.features import add_calendar_features, add_lag_features
from pipeline.ingest import fetch_holidays, load_daily, load_intervals
from pipeline.models import Stage1Daily, Stage2Shape


@click.command()
@click.option("--skip-api", is_flag=True, help="Use cached holidays only.")
@click.option("--output-dir", type=click.Path(), default=str(EXPORTS_DIR))
def main(skip_api: bool, output_dir: str):
    click.echo("[1/6] Loading raw data…")
    daily = load_daily()
    intervals = load_intervals()

    click.echo("[2/6] Fetching holidays…")
    years = sorted({daily["date"].dt.year.min(), daily["date"].dt.year.max(), 2025})
    holidays = fetch_holidays(list(years), force_refresh=False)

    click.echo("[3/6] Engineering features…")
    daily = add_calendar_features(daily, holidays)
    daily = add_lag_features(daily, target_cols=["cv", "cct", "abd_calls"])
    daily = daily.dropna(subset=[c for c in daily.columns if c.endswith("_lag_1")])

    feature_cols = [
        c for c in daily.columns
        if c not in {"date", "portfolio", "cv", "cct", "abd_calls", "abd_rate"}
    ]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(daily[c])]

    click.echo("[4/6] Fitting Stage-1 (daily) and Stage-2 (shape)…")
    stage1 = Stage1Daily(feature_cols=feature_cols).fit(daily)
    stage2 = Stage2Shape().fit(
        intervals[(intervals["date"] >= INTERVAL_OBS_START) & (intervals["date"] <= INTERVAL_OBS_END)]
    )

    click.echo("[5/6] Forecasting August 2025…")
    future = _august_frame(daily, feature_cols)
    daily_fc = stage1.predict(future)
    interval_fc = stage2.expand(daily_fc)

    click.echo("[6/6] Writing outputs…")
    sub_path = write_submission_csv(interval_fc)
    scores = {"asymmetric_mape_holdout": None, "note": "run `make backtest` for holdout scores"}
    dash_path = write_dashboard_payload(
        intervals=interval_fc,
        daily=daily_fc,
        scores=scores,
    )
    click.echo(f"  submission → {sub_path}")
    click.echo(f"  dashboard  → {dash_path}")


def _august_frame(daily: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build an August-2025 prediction grid using last-known lag values per portfolio."""
    start = pd.Timestamp(f"{FORECAST_MONTH}-01")
    dates = pd.date_range(start, periods=FORECAST_DAYS, freq="D")
    rows = []
    for p in PORTFOLIOS:
        last = daily[daily["portfolio"] == p].sort_values("date").iloc[-1]
        for d in dates:
            r = last[feature_cols].to_dict()
            r["portfolio"] = p
            r["date"] = d
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
