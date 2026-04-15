"""Global configuration: paths, portfolios, date ranges, model hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_DIR = DATA / "raw"
INTERIM_DIR = DATA / "interim"
PROCESSED_DIR = DATA / "processed"
EXPORTS_DIR = DATA / "exports"

RAW_WORKBOOK = RAW_DIR / "Data for Datathon (Revised).xlsx"

PORTFOLIOS = ("A", "B", "C", "D")
METRICS = ("CV", "CCT", "ABD")  # Call Volume, Customer Care Time, Abandoned

INTERVALS_PER_DAY = 48
FORECAST_MONTH = "2025-08"
FORECAST_DAYS = 31
INTERVAL_MINUTES = 30

TRAIN_START = "2024-01-01"
TRAIN_END = "2025-07-31"
INTERVAL_OBS_START = "2025-04-01"
INTERVAL_OBS_END = "2025-06-30"

SEED = 42


@dataclass(frozen=True)
class AsymmetricLoss:
    """Scoring penalty: underprediction hurts more than overprediction."""
    under_weight: float = 1.5
    over_weight: float = 1.0


@dataclass(frozen=True)
class Stage1Params:
    cv_quantile_alpha: dict = field(default_factory=lambda: {
        "A": 0.58, "B": 0.60, "C": 0.55, "D": 0.62,
    })
    cct_quantile_alpha: float = 0.57
    abd_poisson_shift: float = 0.05
    n_estimators: int = 800
    max_depth: int = 6
    learning_rate: float = 0.04
    subsample: float = 0.85
    colsample_bytree: float = 0.85


@dataclass(frozen=True)
class BlendWeights:
    """Blend weights for Abandoned Calls: w * xgb + (1-w) * seasonal_naive."""
    per_portfolio: dict = field(default_factory=lambda: {
        "A": 0.75, "B": 0.65, "C": 0.95, "D": 0.85,
    })


HOLIDAYS_API = "https://date.nager.at/api/v3/PublicHolidays/{year}/US"
HOLIDAYS_CACHE = INTERIM_DIR / "holidays.json"

LOSS = AsymmetricLoss()
STAGE1 = Stage1Params()
BLEND = BlendWeights()
