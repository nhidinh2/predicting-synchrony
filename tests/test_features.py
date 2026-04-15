import pandas as pd

from pipeline.features.calendar import add_calendar_features
from pipeline.features.lags import add_lag_features


def test_calendar_flags_weekend_and_holiday():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2025-07-04", "2025-07-05", "2025-07-07"]),
        "portfolio": ["A", "A", "A"],
    })
    hol = pd.DataFrame({
        "date": pd.to_datetime(["2025-07-04"]),
        "name": ["Independence Day"],
        "type": ["Public"],
    })
    out = add_calendar_features(df, hol)
    assert out.loc[0, "is_holiday"] == 1
    assert out.loc[1, "is_weekend"] == 1
    assert out.loc[2, "is_weekend"] == 0


def test_lags_preserve_row_count():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=400),
        "portfolio": ["A"] * 400,
        "cv": range(400),
        "cct": range(400),
        "abd_calls": range(400),
    })
    out = add_lag_features(df, ["cv", "cct", "abd_calls"])
    assert len(out) == len(df)
    assert "cv_lag_365" in out.columns
