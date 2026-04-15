"""Backtest harness: leave-month-out evaluation with asymmetric loss."""
from __future__ import annotations

import click
import pandas as pd

from pipeline.evaluate import asymmetric_mape, leave_month_out
from pipeline.features import add_calendar_features, add_lag_features
from pipeline.ingest import fetch_holidays, load_daily
from pipeline.models import Stage1Daily


@click.command()
@click.option("--holdout", default="2024-08", help="YYYY-MM month to hold out")
def main(holdout: str):
    click.echo(f"Backtest holdout: {holdout}")
    daily = load_daily()
    years = sorted(daily["date"].dt.year.unique().tolist())
    holidays = fetch_holidays(years, force_refresh=False)
    daily = add_calendar_features(daily, holidays)
    daily = add_lag_features(daily, target_cols=["cv", "cct", "abd_calls"])
    daily = daily.dropna(subset=[c for c in daily.columns if c.endswith("_lag_1")])

    train, test = leave_month_out(daily, holdout)

    feature_cols = [
        c for c in daily.columns
        if c not in {"date", "portfolio", "cv", "cct", "abd_calls", "abd_rate"}
    ]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(daily[c])]

    model = Stage1Daily(feature_cols=feature_cols).fit(train)
    pred = model.predict(test)

    merged = test.merge(pred, on=["portfolio", "date"])
    for metric, pred_col, true_col in [
        ("CV", "cv_daily", "cv"),
        ("CCT", "cct_daily", "cct"),
        ("ABD_calls", "abd_calls_daily", "abd_calls"),
    ]:
        score = asymmetric_mape(merged[true_col].values, merged[pred_col].values)
        click.echo(f"  {metric:<10s} asymmetric MAPE = {score:.4f}")


if __name__ == "__main__":
    main()
