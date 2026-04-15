# Research Notebooks

Original exploratory notebooks, frozen in place for reproducibility. The
`pipeline/` Python package is the source of truth for the productionized
pipeline; these notebooks document the analytical journey.

| Notebook | Purpose |
| --- | --- |
| `1_eda.ipynb` | Initial EDA on daily + interval data |
| `2_baseline.ipynb` | Ridge baseline and residual structure |
| `3_validation.ipynb` | Residual diagnostics, loss / target choice |
| `4_final_model.ipynb` | Final two-stage XGBoost model |
| `4b_final_model_rate.ipynb` | Rate-target variant for Abandoned |
| `5_experiments.ipynb` | Adversarial validation, Erlang-A, seasonal-naive |
