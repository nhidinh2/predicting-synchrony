"""Write JSON payload consumed by the React dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.config import EXPORTS_DIR, PORTFOLIOS


def write_dashboard_payload(
    intervals: pd.DataFrame,
    daily: pd.DataFrame,
    scores: dict,
    residuals: pd.DataFrame | None = None,
    path: Path | None = None,
) -> Path:
    path = path or (EXPORTS_DIR / "dashboard.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "portfolios": list(PORTFOLIOS),
            "forecast_month": "August 2025",
            "generated_at": pd.Timestamp.utcnow().isoformat(),
        },
        "scores": scores,
        "intervals": _interval_records(intervals),
        "daily": _daily_records(daily),
        "residuals": residuals.to_dict(orient="records") if residuals is not None else [],
    }

    with path.open("w") as f:
        json.dump(payload, f, default=str)
    return path


def _interval_records(df: pd.DataFrame) -> list[dict]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d.to_dict(orient="records")


def _daily_records(df: pd.DataFrame) -> list[dict]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d.to_dict(orient="records")
