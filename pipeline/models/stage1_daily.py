"""Stage-1 daily model: one booster per metric, pooled across portfolios.

Call Volume → quantile loss (per-portfolio alpha).
CCT         → quantile loss (alpha ≈ 0.57).
Abandoned   → count:poisson on abd_calls (rate derived post-hoc).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.config import PORTFOLIOS, SEED, STAGE1


@dataclass
class Stage1Daily:
    feature_cols: list[str]
    models: dict = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "Stage1Daily":
        X = train[self.feature_cols]
        # CV: per-portfolio quantile models
        self.models["cv"] = {}
        for p in PORTFOLIOS:
            mask = train["portfolio"] == p
            alpha = STAGE1.cv_quantile_alpha[p]
            m = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                n_estimators=STAGE1.n_estimators,
                max_depth=STAGE1.max_depth,
                learning_rate=STAGE1.learning_rate,
                subsample=STAGE1.subsample,
                colsample_bytree=STAGE1.colsample_bytree,
                random_state=SEED,
            )
            m.fit(X[mask], train.loc[mask, "cv"])
            self.models["cv"][p] = m

        # CCT: single pooled quantile model with portfolio one-hot
        m_cct = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=STAGE1.cct_quantile_alpha,
            n_estimators=STAGE1.n_estimators,
            max_depth=STAGE1.max_depth,
            learning_rate=STAGE1.learning_rate,
            subsample=STAGE1.subsample,
            colsample_bytree=STAGE1.colsample_bytree,
            random_state=SEED,
        )
        m_cct.fit(X, train["cct"])
        self.models["cct"] = m_cct

        # Abandoned Calls: Poisson count model
        m_abd = xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=STAGE1.n_estimators,
            max_depth=STAGE1.max_depth,
            learning_rate=STAGE1.learning_rate,
            subsample=STAGE1.subsample,
            colsample_bytree=STAGE1.colsample_bytree,
            random_state=SEED,
        )
        m_abd.fit(X, train["abd_calls"])
        self.models["abd"] = m_abd

        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.feature_cols]
        out = frame[["portfolio", "date"]].copy()
        out["cv_daily"] = np.nan
        for p in PORTFOLIOS:
            mask = frame["portfolio"] == p
            if mask.any():
                out.loc[mask, "cv_daily"] = self.models["cv"][p].predict(X[mask])
        out["cct_daily"] = self.models["cct"].predict(X)
        out["abd_calls_daily"] = self.models["abd"].predict(X) * (1 + STAGE1.abd_poisson_shift)
        out["abd_rate_daily"] = out["abd_calls_daily"] / out["cv_daily"].clip(lower=1)
        return out
