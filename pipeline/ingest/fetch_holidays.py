"""Fetch US federal holidays from nager.at public API with local caching."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from pipeline.config import HOLIDAYS_API, HOLIDAYS_CACHE


def fetch_holidays(
    years: list[int],
    cache_path: Path = HOLIDAYS_CACHE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return DataFrame with columns: date, name, type. Cached locally."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_cache(cache_path) if cache_path.exists() and not force_refresh else {}

    records: list[dict] = []
    for year in years:
        key = str(year)
        if key in cached:
            records.extend(cached[key])
            continue
        resp = requests.get(HOLIDAYS_API.format(year=year), timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        cached[key] = payload
        records.extend(payload)

    _save_cache(cache_path, cached)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["date", "name", "type"])
    df["date"] = pd.to_datetime(df["date"])
    df["type"] = df.get("types", pd.Series([["Public"]] * len(df))).apply(
        lambda x: x[0] if isinstance(x, list) and x else "Public"
    )
    return df[["date", "name", "type"]].drop_duplicates("date").reset_index(drop=True)


def _load_cache(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _save_cache(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)
