"""
Synchrony Datathon — Two Baseline Models with Proper Validation
================================================================
Baseline 1: Seasonal Naive (same DOW from recent weeks)
Baseline 2: DOW + Holiday Group Mean (historical average by DOW bucket)

Validation: Train on Jan 2024 - Sep 2025, validate on Oct-Dec 2025
Metrics: WAPE, MAE, Bias (mean error), underprediction rate
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

os.makedirs('baseline_plots', exist_ok=True)

DATA_FILE = "Data for Datathon (Revised).xlsx"
PORTFOLIOS = ['A', 'B', 'C', 'D']

# US Holidays
US_HOLIDAYS = set([
    datetime(2024, 1, 1), datetime(2024, 1, 15), datetime(2024, 2, 19),
    datetime(2024, 5, 27), datetime(2024, 7, 4), datetime(2024, 9, 2),
    datetime(2024, 10, 14), datetime(2024, 11, 11), datetime(2024, 11, 28),
    datetime(2024, 12, 25),
    datetime(2025, 1, 1), datetime(2025, 1, 20), datetime(2025, 2, 17),
    datetime(2025, 5, 26), datetime(2025, 7, 4), datetime(2025, 9, 1),
    datetime(2025, 10, 13), datetime(2025, 11, 11), datetime(2025, 11, 27),
    datetime(2025, 12, 25),
    datetime(2026, 7, 3), datetime(2026, 7, 4),
])

METRICS = ['Call Volume', 'CCT', 'Abandon Rate']
VAL_CUTOFF = pd.Timestamp('2025-10-01')

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")

daily_data = {}
for p in PORTFOLIOS:
    df = pd.read_excel(DATA_FILE, sheet_name=f"{p} - Daily")
    df['Date'] = pd.to_datetime(df['Date'].str.strip().str[:8], format='%m/%d/%y')
    df = df.sort_values('Date').reset_index(drop=True)
    df['dow'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['is_holiday'] = df['Date'].isin(US_HOLIDAYS).astype(int)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    daily_data[p] = df

print("Data loaded.\n")

# =============================================================================
# SPLIT: TRAIN vs VALIDATION
# =============================================================================
print("=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

train_data = {}
val_data = {}
for p in PORTFOLIOS:
    df = daily_data[p]
    train_data[p] = df[df['Date'] < VAL_CUTOFF].copy()
    val_data[p] = df[df['Date'] >= VAL_CUTOFF].copy()
    print(f"  Portfolio {p}: train={len(train_data[p])} days "
          f"({train_data[p]['Date'].min().date()} to {train_data[p]['Date'].max().date()}), "
          f"val={len(val_data[p])} days "
          f"({val_data[p]['Date'].min().date()} to {val_data[p]['Date'].max().date()})")

print()

# =============================================================================
# BASELINE 1: SEASONAL NAIVE
# =============================================================================
print("=" * 70)
print("BASELINE 1: SEASONAL NAIVE")
print("=" * 70)
print("  Rule: For each val day, use the most recent same-DOW day from training.\n")

def seasonal_naive_predict(train_df, val_df):
    """For each validation day, find the most recent same-DOW day in training."""
    predictions = []
    for _, val_row in val_df.iterrows():
        target_date = val_row['Date']
        target_dow = val_row['dow']
        target_is_holiday = val_row['is_holiday']

        # If holiday, find most recent holiday in training
        if target_is_holiday:
            candidates = train_df[train_df['is_holiday'] == 1]
        else:
            # Find same DOW, non-holiday
            candidates = train_df[(train_df['dow'] == target_dow) & (train_df['is_holiday'] == 0)]

        if len(candidates) == 0:
            candidates = train_df[train_df['dow'] == target_dow]

        # Most recent
        candidates = candidates.sort_values('Date', ascending=False)
        best = candidates.iloc[0]

        pred = {
            'Date': target_date,
            'dow': target_dow,
            'is_holiday': target_is_holiday,
        }
        for m in METRICS:
            pred[f'{m}_pred'] = best[m]
            pred[f'{m}_actual'] = val_row[m]
        predictions.append(pred)

    return pd.DataFrame(predictions)


baseline1_results = {}
for p in PORTFOLIOS:
    baseline1_results[p] = seasonal_naive_predict(train_data[p], val_data[p])

# =============================================================================
# BASELINE 2: DOW + HOLIDAY GROUP MEAN
# =============================================================================
print("=" * 70)
print("BASELINE 2: DOW + HOLIDAY GROUP MEAN")
print("=" * 70)
print("  Rule: Average all historical same-DOW days. Holidays get their own bucket.\n")

def dow_group_mean_predict(train_df, val_df):
    """For each validation day, use the average of all same-DOW days in training."""
    predictions = []

    # Compute group means
    # Group 1: holidays (all holidays together)
    holiday_means = {}
    hol_train = train_df[train_df['is_holiday'] == 1]
    for m in METRICS:
        holiday_means[m] = hol_train[m].mean() if len(hol_train) > 0 else train_df[m].mean()

    # Group 2: by DOW (non-holiday only)
    dow_means = {}
    non_hol = train_df[train_df['is_holiday'] == 0]
    for dow in range(7):
        dow_means[dow] = {}
        dow_subset = non_hol[non_hol['dow'] == dow]
        for m in METRICS:
            dow_means[dow][m] = dow_subset[m].mean() if len(dow_subset) > 0 else non_hol[m].mean()

    for _, val_row in val_df.iterrows():
        target_date = val_row['Date']
        target_dow = val_row['dow']
        target_is_holiday = val_row['is_holiday']

        pred = {
            'Date': target_date,
            'dow': target_dow,
            'is_holiday': target_is_holiday,
        }

        if target_is_holiday:
            for m in METRICS:
                pred[f'{m}_pred'] = holiday_means[m]
                pred[f'{m}_actual'] = val_row[m]
        else:
            for m in METRICS:
                pred[f'{m}_pred'] = dow_means[target_dow][m]
                pred[f'{m}_actual'] = val_row[m]

        predictions.append(pred)

    return pd.DataFrame(predictions)


baseline2_results = {}
for p in PORTFOLIOS:
    baseline2_results[p] = dow_group_mean_predict(train_data[p], val_data[p])

# =============================================================================
# EVALUATION METRICS
# =============================================================================
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

def compute_metrics(results_df, metric):
    """Compute WAPE, MAE, Bias, and underprediction rate."""
    actual_col = f'{metric}_actual'
    pred_col = f'{metric}_pred'

    # Drop NaN rows
    df = results_df.dropna(subset=[actual_col, pred_col])
    if len(df) == 0:
        return {'WAPE': np.nan, 'MAE': np.nan, 'Bias': np.nan, 'UnderPred%': np.nan}

    actual = df[actual_col].values
    pred = df[pred_col].values
    errors = actual - pred  # positive = underprediction

    wape = np.sum(np.abs(errors)) / np.sum(np.abs(actual)) if np.sum(np.abs(actual)) > 0 else np.nan
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)  # positive = model underpredicts on average
    underpred_rate = np.mean(errors > 0) * 100  # % of days where we underpredicted

    return {
        'WAPE': wape,
        'MAE': mae,
        'Bias': bias,
        'UnderPred%': underpred_rate,
        'N': len(df)
    }

# Print comparison table
for metric in METRICS:
    print(f"\n  --- {metric} ---")
    print(f"  {'Portfolio':<10} {'Model':<25} {'WAPE':>8} {'MAE':>10} {'Bias':>10} {'Under%':>8} {'N':>5}")
    print("  " + "-" * 78)

    for p in PORTFOLIOS:
        m1 = compute_metrics(baseline1_results[p], metric)
        m2 = compute_metrics(baseline2_results[p], metric)

        print(f"  {p:<10} {'Seasonal Naive':<25} {m1['WAPE']:>8.4f} {m1['MAE']:>10.2f} {m1['Bias']:>10.2f} {m1['UnderPred%']:>7.1f}% {m1['N']:>5.0f}")
        print(f"  {'':<10} {'DOW+Holiday Mean':<25} {m2['WAPE']:>8.4f} {m2['MAE']:>10.2f} {m2['Bias']:>10.2f} {m2['UnderPred%']:>7.1f}% {m2['N']:>5.0f}")

        # Mark winner
        winner = "Seasonal Naive" if m1['WAPE'] < m2['WAPE'] else "DOW+Holiday Mean"
        print(f"  {'':<10} {'→ Winner: ' + winner:<25}")

# =============================================================================
# COMPUTE THE ACTUAL SCORING FORMULA (from slides)
# =============================================================================
print("\n")
print("=" * 70)
print("SCORING FORMULA METRICS (matching competition scoring)")
print("=" * 70)

def compute_competition_errors(results_df):
    """
    Compute E_V, E_C, E_B as defined in the competition scoring slides.
    E_V = sum|A-F| / sum(A) for volume
    E_C = same but only for intervals where actual CCT > 0
    E_B = sum|A-F| / sum(A) for abandon rate
    """
    errors = {}

    # E_V: Volume WAPE
    df = results_df.dropna(subset=['Call Volume_actual', 'Call Volume_pred'])
    a = df['Call Volume_actual'].values
    f = df['Call Volume_pred'].values
    errors['E_V'] = np.sum(np.abs(a - f)) / np.sum(a) if np.sum(a) > 0 else np.nan

    # E_C: CCT WAPE (only where actual > 0)
    df = results_df.dropna(subset=['CCT_actual', 'CCT_pred'])
    mask = df['CCT_actual'] > 0
    a = df.loc[mask, 'CCT_actual'].values
    f = df.loc[mask, 'CCT_pred'].values
    errors['E_C'] = np.sum(np.abs(a - f)) / np.sum(a) if np.sum(a) > 0 else np.nan

    # E_B: Abandon Rate WAPE
    df = results_df.dropna(subset=['Abandon Rate_actual', 'Abandon Rate_pred'])
    a = df['Abandon Rate_actual'].values
    f = df['Abandon Rate_pred'].values
    errors['E_B'] = np.sum(np.abs(a - f)) / np.sum(a) if np.sum(a) > 0 else np.nan

    return errors

print(f"\n  {'Portfolio':<10} {'Model':<25} {'E_V':>8} {'E_C':>8} {'E_B':>8}")
print("  " + "-" * 62)

for p in PORTFOLIOS:
    e1 = compute_competition_errors(baseline1_results[p])
    e2 = compute_competition_errors(baseline2_results[p])

    print(f"  {p:<10} {'Seasonal Naive':<25} {e1['E_V']:>8.4f} {e1['E_C']:>8.4f} {e1['E_B']:>8.4f}")
    print(f"  {'':<10} {'DOW+Holiday Mean':<25} {e2['E_V']:>8.4f} {e2['E_C']:>8.4f} {e2['E_B']:>8.4f}")

# =============================================================================
# VISUALIZATION: Actual vs Predicted
# =============================================================================
print("\n\nGenerating comparison plots...")

for metric in METRICS:
    fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)

    for i, p in enumerate(PORTFOLIOS):
        ax = axes[i]
        r1 = baseline1_results[p].dropna(subset=[f'{metric}_actual'])
        r2 = baseline2_results[p].dropna(subset=[f'{metric}_actual'])

        ax.plot(r1['Date'], r1[f'{metric}_actual'], 'k-', linewidth=1.5, label='Actual', alpha=0.8)
        ax.plot(r1['Date'], r1[f'{metric}_pred'], 'b--', linewidth=1, label='B1: Seasonal Naive', alpha=0.7)
        ax.plot(r2['Date'], r2[f'{metric}_pred'], 'r--', linewidth=1, label='B2: DOW+Holiday Mean', alpha=0.7)

        # Mark holidays
        hol_dates = r1[r1['is_holiday'] == 1]['Date']
        for h in hol_dates:
            ax.axvline(h, color='green', alpha=0.2, linewidth=2)

        ax.set_ylabel(f'Portfolio {p}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f'{metric}: Actual vs Baselines (Oct-Dec 2025)')
    plt.tight_layout()
    safe_metric = metric.replace(' ', '_')
    plt.savefig(f'baseline_plots/val_{safe_metric}.png', dpi=150)
    plt.close()

# Error distribution plots
for metric in METRICS:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    for j, p in enumerate(PORTFOLIOS):
        r1 = baseline1_results[p].dropna(subset=[f'{metric}_actual', f'{metric}_pred'])
        r2 = baseline2_results[p].dropna(subset=[f'{metric}_actual', f'{metric}_pred'])

        err1 = r1[f'{metric}_actual'] - r1[f'{metric}_pred']
        err2 = r2[f'{metric}_actual'] - r2[f'{metric}_pred']

        # Baseline 1 error histogram
        ax = axes[0, j]
        ax.hist(err1, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--')
        ax.axvline(err1.mean(), color='blue', linestyle='--', label=f'Bias={err1.mean():.2f}')
        ax.set_title(f'{p}: Seasonal Naive')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Baseline 2 error histogram
        ax = axes[1, j]
        ax.hist(err2, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(0, color='red', linestyle='--')
        ax.axvline(err2.mean(), color='blue', linestyle='--', label=f'Bias={err2.mean():.2f}')
        ax.set_title(f'{p}: DOW+Holiday Mean')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{metric}: Error Distributions (positive = underprediction)', fontsize=14)
    plt.tight_layout()
    safe_metric = metric.replace(' ', '_')
    plt.savefig(f'baseline_plots/err_dist_{safe_metric}.png', dpi=150)
    plt.close()

# DOW breakdown of errors
for metric in METRICS:
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for j, p in enumerate(PORTFOLIOS):
        r1 = baseline1_results[p].dropna(subset=[f'{metric}_actual', f'{metric}_pred'])
        r2 = baseline2_results[p].dropna(subset=[f'{metric}_actual', f'{metric}_pred'])

        # WAPE by DOW for each baseline
        for bi, (results, name) in enumerate([(r1, 'Seasonal Naive'), (r2, 'DOW+Holiday Mean')]):
            ax = axes[bi, j]
            dow_wape = []
            for dow in range(7):
                ddf = results[results['dow'] == dow]
                if len(ddf) > 0:
                    a = ddf[f'{metric}_actual'].values
                    f_vals = ddf[f'{metric}_pred'].values
                    w = np.sum(np.abs(a - f_vals)) / np.sum(np.abs(a)) if np.sum(np.abs(a)) > 0 else 0
                    dow_wape.append(w)
                else:
                    dow_wape.append(0)
            ax.bar(range(7), dow_wape, alpha=0.7)
            ax.set_xticks(range(7))
            ax.set_xticklabels(dow_names, fontsize=8)
            ax.set_title(f'{p}: {name}')
            ax.set_ylabel('WAPE')
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{metric}: WAPE by Day of Week', fontsize=14)
    plt.tight_layout()
    safe_metric = metric.replace(' ', '_')
    plt.savefig(f'baseline_plots/dow_wape_{safe_metric}.png', dpi=150)
    plt.close()

print("\nPlots saved to ./baseline_plots/")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n")
print("=" * 70)
print("SUMMARY: WHICH BASELINE WINS?")
print("=" * 70)

wins = {'Seasonal Naive': 0, 'DOW+Holiday Mean': 0}
print(f"\n  {'Portfolio':<10} {'Metric':<15} {'B1 WAPE':>10} {'B2 WAPE':>10} {'Winner':<20}")
print("  " + "-" * 68)

for p in PORTFOLIOS:
    for metric in METRICS:
        m1 = compute_metrics(baseline1_results[p], metric)
        m2 = compute_metrics(baseline2_results[p], metric)

        if np.isnan(m1['WAPE']) or np.isnan(m2['WAPE']):
            winner = 'N/A'
        elif m1['WAPE'] < m2['WAPE']:
            winner = 'Seasonal Naive'
            wins['Seasonal Naive'] += 1
        else:
            winner = 'DOW+Holiday Mean'
            wins['DOW+Holiday Mean'] += 1

        print(f"  {p:<10} {metric:<15} {m1['WAPE']:>10.4f} {m2['WAPE']:>10.4f} {winner:<20}")

print(f"\n  Overall wins: Seasonal Naive={wins['Seasonal Naive']}, DOW+Holiday Mean={wins['DOW+Holiday Mean']}")
print(f"  → The floor to beat is whichever has lower WAPE per metric.")
print(f"\n  Any future model must beat BOTH of these to justify its complexity.")
