"""
Synchrony Datathon — Exploratory Data Analysis
===============================================
Thorough EDA before any modeling. All plots saved to ./eda_plots/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('eda_plots', exist_ok=True)

DATA_FILE = "Data for Datathon (Revised).xlsx"
PORTFOLIOS = ['A', 'B', 'C', 'D']
MONTH_MAP = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
             'May': 5, 'June': 6, 'July': 7, 'August': 8,
             'September': 9, 'October': 10, 'November': 11, 'December': 12}

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
])

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
    df['dow_name'] = df['Date'].dt.day_name()
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_holiday'] = df['Date'].isin(US_HOLIDAYS).astype(int)
    df['year'] = df['Date'].dt.year
    daily_data[p] = df

interval_data = {}
for p in PORTFOLIOS:
    df = pd.read_excel(DATA_FILE, sheet_name=f"{p} - Interval")
    df = df.dropna(subset=['Interval', 'Call Volume']).reset_index(drop=True)
    df['Interval'] = df['Interval'].astype(str)
    def parse_interval(val):
        val = str(val).strip()
        if 'days' in val:
            return val.split(' ')[-1][:5]
        return val[:5] if len(val) >= 5 else val
    df['Interval_str'] = df['Interval'].apply(parse_interval)
    df['half_hour'] = df['Interval_str'].apply(
        lambda x: int(x.split(':')[0]) * 2 + (1 if int(x.split(':')[1]) >= 30 else 0)
    )
    df['month_num'] = df['Month'].map(MONTH_MAP)
    # Create dates (assume 2025 for interval data)
    dates = []
    for _, row in df.iterrows():
        try:
            dates.append(datetime(2025, row['month_num'], row['Day']))
        except ValueError:
            dates.append(None)
    df['Date'] = dates
    df = df.dropna(subset=['Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['dow'] = df['Date'].dt.dayofweek
    df['dow_name'] = df['Date'].dt.day_name()
    interval_data[p] = df

staffing = pd.read_excel(DATA_FILE, sheet_name="Daily Staffing")
staffing.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
staffing['Date'] = pd.to_datetime(staffing['Date'])

print("Data loaded.\n")

# =============================================================================
# SECTION 1: PROBLEM FRAMING
# =============================================================================
print("=" * 70)
print("SECTION 1: PROBLEM FRAMING")
print("=" * 70)

print("""
WHAT WE PREDICT:
  1. Call Volume (CV) — count of incoming calls per 30-min interval
  2. Customer Care Time (CCT) — avg handle time in seconds (continuous)
  3. Abandon Rate (ABD) — fraction of callers who hang up (proportion 0-1)

TIME UNIT: 30-minute intervals, 48 per day

HORIZON: Full month of August 2026 (all 31 days × 48 intervals = 1,488 predictions per metric per portfolio)

PORTFOLIOS: 4 portfolios (A, B, C, D) of varying sizes

ASYMMETRIC COST:
  The scoring formula has a Workload Penalty:
    Workload_t = Volume_t × CCT_t
    Penalty_t = α × max(Actual - Forecast, 0) + β × max(Forecast - Actual, 0)

  If α > β: UNDERPREDICTION is penalized more (understaffing is costly)
  If β > α: OVERPREDICTION is penalized more

  → We don't know α, β values, but the slides emphasize understaffing risk,
    suggesting underprediction is likely penalized more heavily.
  → This means we may want a slight UPWARD BIAS in our forecasts.

TRAINING DATA AVAILABLE:
  - Daily data: 731 days (Jan 2024 - Dec 2025) — 2 full years
  - Interval data: ~3 months (Apr-Jun 2025) — 30-min granularity
  - Daily staffing: 365 days (2025)
""")

# =============================================================================
# SECTION 2: TIME-SERIES PLOTS
# =============================================================================
print("=" * 70)
print("SECTION 2: TIME-SERIES PLOTS")
print("=" * 70)

# --- 2a: Target over time (daily) ---
print("  Plotting daily targets over time...")
fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
metrics = ['Call Volume', 'CCT', 'Abandon Rate']
for i, metric in enumerate(metrics):
    ax = axes[i]
    for p in PORTFOLIOS:
        df = daily_data[p]
        ax.plot(df['Date'], df[metric], label=f'Portfolio {p}', alpha=0.7, linewidth=0.8)
    ax.set_ylabel(metric)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    # Mark holidays
    for h in US_HOLIDAYS:
        if df['Date'].min() <= h <= df['Date'].max():
            ax.axvline(h, color='red', alpha=0.15, linewidth=1)
axes[0].set_title('Daily Targets Over Time (red lines = holidays)')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.savefig('eda_plots/01_daily_targets_over_time.png', dpi=150)
plt.close()

# --- 2b: Week-over-week overlay (for interval data) ---
print("  Plotting week-over-week overlays...")
for p in PORTFOLIOS:
    df = interval_data[p]
    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    # Create a within-week index: dow * 48 + half_hour
    df['week_slot'] = df['dow'] * 48 + df['half_hour']

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    for i, metric in enumerate(['Call Volume', 'CCT', 'Abandoned Rate']):
        ax = axes[i]
        weeks = sorted(df['week'].unique())
        for w in weeks:
            wdf = df[df['week'] == w].sort_values('week_slot')
            ax.plot(wdf['week_slot'], wdf[metric], alpha=0.5, linewidth=0.7, label=f'W{w}')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        # Add DOW labels
        for d in range(7):
            ax.axvline(d * 48, color='gray', alpha=0.3, linestyle='--')
    axes[0].set_title(f'Portfolio {p}: Week-over-Week Overlay (interval data)')
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[-1].set_xticks([d * 48 + 24 for d in range(7)])
    axes[-1].set_xticklabels(dow_labels)
    plt.tight_layout()
    plt.savefig(f'eda_plots/02_week_overlay_{p}.png', dpi=150)
    plt.close()

# --- 2c: Day-over-day overlay (same DOW across weeks) ---
print("  Plotting day-over-day overlays by DOW...")
for p in PORTFOLIOS:
    df = interval_data[p]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    colors_dow = plt.cm.Set1(np.linspace(0, 1, 7))

    for i, metric in enumerate(['Call Volume', 'CCT', 'Abandoned Rate']):
        ax = axes[i]
        for dow in range(7):
            dow_df = df[df['dow'] == dow]
            avg = dow_df.groupby('half_hour')[metric].mean()
            ax.plot(avg.index, avg.values, color=colors_dow[dow],
                    label=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow], linewidth=2)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    axes[0].set_title(f'Portfolio {p}: Average by Hour-of-Day, grouped by DOW')
    axes[-1].set_xlabel('Half-hour slot (0=00:00, 47=23:30)')
    plt.tight_layout()
    plt.savefig(f'eda_plots/03_avg_by_hour_dow_{p}.png', dpi=150)
    plt.close()

# --- 2d: Average target by day of week (daily data) ---
print("  Plotting average by DOW...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i, metric in enumerate(metrics):
    ax = axes[i]
    for p in PORTFOLIOS:
        df = daily_data[p]
        avg = df.groupby('dow')[metric].mean()
        ax.plot(avg.index, avg.values, marker='o', label=f'Portfolio {p}', linewidth=2)
    ax.set_ylabel(metric)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_title('Average Daily Target by Day of Week')
plt.tight_layout()
plt.savefig('eda_plots/04_avg_by_dow.png', dpi=150)
plt.close()

# --- 2e: Weekday × Hour heatmap (interval data) ---
print("  Plotting weekday × hour heatmaps...")
for p in PORTFOLIOS:
    df = interval_data[p]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, metric in enumerate(['Call Volume', 'CCT', 'Abandoned Rate']):
        ax = axes[i]
        pivot = df.pivot_table(values=metric, index='dow', columns='half_hour', aggfunc='mean')
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(7))
        ax.set_yticklabels(dow_names)
        ax.set_xlabel('Half-hour slot')
        ax.set_title(metric)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'Portfolio {p}: Weekday × Hour Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eda_plots/05_heatmap_{p}.png', dpi=150)
    plt.close()

# --- 2f: Monthly pattern (daily data) ---
print("  Plotting monthly patterns...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i, metric in enumerate(metrics):
    ax = axes[i]
    for p in PORTFOLIOS:
        df = daily_data[p]
        avg = df.groupby('month')[metric].mean()
        ax.plot(avg.index, avg.values, marker='o', label=f'Portfolio {p}', linewidth=2)
    ax.set_ylabel(metric)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_title('Average Daily Target by Month')
plt.tight_layout()
plt.savefig('eda_plots/06_avg_by_month.png', dpi=150)
plt.close()

print("  Section 2 plots saved.\n")

# =============================================================================
# SECTION 3: TARGET DISTRIBUTION
# =============================================================================
print("=" * 70)
print("SECTION 3: TARGET DISTRIBUTION")
print("=" * 70)

# --- 3a: Histograms ---
print("  Plotting histograms...")
for p in PORTFOLIOS:
    # Daily histograms
    df = daily_data[p]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = df[metric].dropna()
        ax.hist(vals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(vals.mean(), color='red', linestyle='--', label=f'Mean: {vals.mean():.1f}')
        ax.axvline(vals.median(), color='blue', linestyle='--', label=f'Median: {vals.median():.1f}')
        ax.set_title(f'{metric}')
        ax.legend(fontsize=8)
    fig.suptitle(f'Portfolio {p}: Daily Target Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eda_plots/07_hist_daily_{p}.png', dpi=150)
    plt.close()

    # Interval histograms
    idf = interval_data[p]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, metric in enumerate(['Call Volume', 'CCT', 'Abandoned Rate']):
        ax = axes[i]
        vals = idf[metric].dropna()
        ax.hist(vals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(vals.mean(), color='red', linestyle='--', label=f'Mean: {vals.mean():.1f}')
        ax.axvline(vals.median(), color='blue', linestyle='--', label=f'Median: {vals.median():.1f}')
        ax.set_title(f'{metric}')
        ax.legend(fontsize=8)
    fig.suptitle(f'Portfolio {p}: Interval Target Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eda_plots/08_hist_interval_{p}.png', dpi=150)
    plt.close()

# --- 3b: Distribution statistics ---
print("\n  DISTRIBUTION STATISTICS (DAILY):")
print(f"  {'Portfolio':<10} {'Metric':<15} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Zeros%':>8} {'Var/Mean':>10}")
print("  " + "-" * 95)
for p in PORTFOLIOS:
    df = daily_data[p]
    for metric in metrics:
        vals = df[metric].dropna()
        mean = vals.mean()
        med = vals.median()
        std = vals.std()
        mn = vals.min()
        mx = vals.max()
        zeros = (vals == 0).sum() / len(vals) * 100
        var_mean = vals.var() / mean if mean > 0 else 0
        print(f"  {p:<10} {metric:<15} {mean:>10.2f} {med:>10.2f} {std:>10.2f} {mn:>10.2f} {mx:>10.2f} {zeros:>7.1f}% {var_mean:>10.1f}")

print("\n  DISTRIBUTION STATISTICS (INTERVAL — 30-min):")
print(f"  {'Portfolio':<10} {'Metric':<15} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Zeros%':>8} {'Var/Mean':>10}")
print("  " + "-" * 95)
for p in PORTFOLIOS:
    idf = interval_data[p]
    for metric in ['Call Volume', 'CCT', 'Abandoned Rate']:
        vals = idf[metric].dropna()
        mean = vals.mean()
        med = vals.median()
        std = vals.std()
        mn = vals.min()
        mx = vals.max()
        zeros = (vals == 0).sum() / len(vals) * 100
        var_mean = vals.var() / mean if mean > 0 else 0
        print(f"  {p:<10} {metric:<15} {mean:>10.2f} {med:>10.2f} {std:>10.2f} {mn:>10.2f} {mx:>10.2f} {zeros:>7.1f}% {var_mean:>10.1f}")

print()

# =============================================================================
# SECTION 4: DATA QUALITY & MISSINGNESS
# =============================================================================
print("=" * 70)
print("SECTION 4: DATA QUALITY & MISSINGNESS")
print("=" * 70)

# --- 4a: Daily data quality ---
print("\n  DAILY DATA QUALITY:")
for p in PORTFOLIOS:
    df = daily_data[p]
    n = len(df)
    date_range = (df['Date'].max() - df['Date'].min()).days + 1
    missing_dates = date_range - n
    dups = df.duplicated(subset=['Date']).sum()

    print(f"\n  Portfolio {p}:")
    print(f"    Rows: {n}, Expected (date range): {date_range}, Missing dates: {missing_dates}")
    print(f"    Duplicate dates: {dups}")
    for metric in metrics:
        na = df[metric].isna().sum()
        zeros = (df[metric] == 0).sum()
        print(f"    {metric}: NaN={na}, Zeros={zeros}")

    # Check for suspicious jumps (>3 std from rolling mean)
    for metric in ['Call Volume']:
        rolling = df[metric].rolling(7, center=True).mean()
        rolling_std = df[metric].rolling(7, center=True).std()
        outliers = df[((df[metric] - rolling).abs() > 3 * rolling_std) & rolling_std.notna()]
        if len(outliers) > 0:
            print(f"    {metric} outlier dates (>3σ from 7-day rolling): {len(outliers)}")
            for _, row in outliers.head(5).iterrows():
                print(f"      {row['Date'].date()}: {row[metric]:.0f} (rolling mean: {rolling[row.name]:.0f})")

# --- 4b: Interval data quality ---
print("\n  INTERVAL DATA QUALITY:")
for p in PORTFOLIOS:
    idf = interval_data[p]
    print(f"\n  Portfolio {p}:")
    print(f"    Rows: {len(idf)}")
    print(f"    Months: {idf['Month'].unique()}")

    # Check expected intervals per day (should be 48)
    intervals_per_day = idf.groupby('Date').size()
    incomplete_days = intervals_per_day[intervals_per_day < 48]
    if len(incomplete_days) > 0:
        print(f"    Days with < 48 intervals: {len(incomplete_days)}")
        for date, count in incomplete_days.items():
            print(f"      {date.date()}: {count} intervals")
    else:
        print(f"    All days have 48 intervals: YES ({len(intervals_per_day)} days)")

    # Missing values
    for metric in ['Call Volume', 'CCT', 'Abandoned Rate']:
        na = idf[metric].isna().sum()
        print(f"    {metric}: NaN={na}")

    # Check for zero-volume intervals
    zero_vol = (idf['Call Volume'] == 0).sum()
    print(f"    Zero-volume intervals: {zero_vol} ({zero_vol/len(idf)*100:.1f}%)")

# --- 4c: Holiday effects ---
print("\n  HOLIDAY EFFECTS (daily Call Volume):")
for p in PORTFOLIOS:
    df = daily_data[p]
    hol = df[df['is_holiday'] == 1]['Call Volume'].mean()
    non_hol_wd = df[(df['is_holiday'] == 0) & (df['is_weekend'] == 0)]['Call Volume'].mean()
    wkend = df[df['is_weekend'] == 1]['Call Volume'].mean()
    print(f"  Portfolio {p}: Holiday avg={hol:.0f}, Weekday avg={non_hol_wd:.0f}, Weekend avg={wkend:.0f}")
    print(f"    Holiday/Weekday ratio: {hol/non_hol_wd:.2f}, Weekend/Weekday ratio: {wkend/non_hol_wd:.2f}")

# Plot holiday vs non-holiday
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, metric in enumerate(metrics):
    ax = axes[i]
    data_to_plot = []
    labels = []
    for p in PORTFOLIOS:
        df = daily_data[p]
        hol = df[df['is_holiday'] == 1][metric]
        non_hol = df[(df['is_holiday'] == 0) & (df['is_weekend'] == 0)][metric]
        wkend = df[df['is_weekend'] == 1][metric]
        data_to_plot.extend([non_hol, wkend, hol])
        labels.extend([f'{p}\nWeekday', f'{p}\nWeekend', f'{p}\nHoliday'])
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightyellow', 'lightcoral'] * 4
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(metric)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle('Weekday vs Weekend vs Holiday Distributions', fontsize=14)
plt.tight_layout()
plt.savefig('eda_plots/09_holiday_effects.png', dpi=150)
plt.close()

print()

# =============================================================================
# SECTION 5: AUTOCORRELATION & LAG ANALYSIS
# =============================================================================
print("=" * 70)
print("SECTION 5: AUTOCORRELATION & LAG ANALYSIS")
print("=" * 70)

# --- 5a: Daily autocorrelation ---
print("\n  DAILY AUTOCORRELATION (Call Volume):")
print(f"  {'Portfolio':<10} {'Lag1':>8} {'Lag2':>8} {'Lag7':>8} {'Lag14':>8} {'Lag30':>8} {'Lag365':>8}")
print("  " + "-" * 62)
for p in PORTFOLIOS:
    df = daily_data[p]
    cv = df['Call Volume'].dropna()
    lags = {}
    for lag in [1, 2, 7, 14, 30, 365]:
        if len(cv) > lag:
            lags[lag] = cv.corr(cv.shift(lag))
        else:
            lags[lag] = float('nan')
    print(f"  {p:<10} {lags[1]:>8.3f} {lags[2]:>8.3f} {lags[7]:>8.3f} {lags[14]:>8.3f} {lags[30]:>8.3f} {lags[365]:>8.3f}")

# --- 5b: Interval autocorrelation ---
print("\n  INTERVAL AUTOCORRELATION (Call Volume, 30-min):")
print(f"  {'Portfolio':<10} {'Lag1':>8} {'Lag2':>8} {'Lag48':>8} {'Lag96':>8} {'Lag336':>8}")
print(f"  {'':10} {'(30m)':>8} {'(1h)':>8} {'(1day)':>8} {'(2day)':>8} {'(1wk)':>8}")
print("  " + "-" * 52)
for p in PORTFOLIOS:
    idf = interval_data[p].sort_values(['Date', 'half_hour'])
    cv = idf['Call Volume'].reset_index(drop=True)
    lags = {}
    for lag in [1, 2, 48, 96, 336]:
        if len(cv) > lag:
            lags[lag] = cv.corr(cv.shift(lag))
        else:
            lags[lag] = float('nan')
    print(f"  {p:<10} {lags[1]:>8.3f} {lags[2]:>8.3f} {lags[48]:>8.3f} {lags[96]:>8.3f} {lags[336]:>8.3f}")

# --- 5c: ACF plots ---
print("\n  Plotting ACF...")
from statsmodels.tsa.stattools import acf as compute_acf

for p in PORTFOLIOS:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Daily ACF
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        vals = daily_data[p][metric].dropna().values
        nlags = min(60, len(vals) - 1)
        acf_vals = compute_acf(vals, nlags=nlags, fft=True)
        ax.bar(range(len(acf_vals)), acf_vals, width=0.8, alpha=0.7)
        ax.axhline(1.96/np.sqrt(len(vals)), color='red', linestyle='--', alpha=0.5)
        ax.axhline(-1.96/np.sqrt(len(vals)), color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'Daily {metric}')
        ax.set_xlabel('Lag (days)')
        ax.grid(True, alpha=0.3)

    # Interval ACF
    idf = interval_data[p].sort_values(['Date', 'half_hour'])
    for i, metric in enumerate(['Call Volume', 'CCT', 'Abandoned Rate']):
        ax = axes[1, i]
        vals = idf[metric].dropna().values
        nlags = min(48 * 7 + 10, len(vals) - 1)  # up to 1 week + buffer
        acf_vals = compute_acf(vals, nlags=nlags, fft=True)
        ax.bar(range(len(acf_vals)), acf_vals, width=0.8, alpha=0.7)
        ax.axhline(1.96/np.sqrt(len(vals)), color='red', linestyle='--', alpha=0.5)
        ax.axhline(-1.96/np.sqrt(len(vals)), color='red', linestyle='--', alpha=0.5)
        # Mark key lags
        for lag, label in [(48, '1d'), (96, '2d'), (336, '1w')]:
            if lag < len(acf_vals):
                ax.axvline(lag, color='green', alpha=0.3, linestyle=':')
                ax.text(lag, max(acf_vals) * 0.9, label, fontsize=8, ha='center')
        ax.set_title(f'Interval {metric}')
        ax.set_xlabel('Lag (half-hours)')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Portfolio {p}: Autocorrelation Functions', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eda_plots/10_acf_{p}.png', dpi=150)
    plt.close()

# --- 5d: Lag scatter plots ---
print("  Plotting lag scatter plots...")
for p in PORTFOLIOS:
    df = daily_data[p].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    df['CV_lag1'] = df['Call Volume'].shift(1)
    df['CV_lag7'] = df['Call Volume'].shift(7)
    df['CV_lag365'] = df['Call Volume'].shift(365)

    for i, (lag_col, lag_name) in enumerate([('CV_lag1', 'Yesterday'), ('CV_lag7', 'Last Week'), ('CV_lag365', 'Last Year')]):
        ax = axes[i]
        valid = df.dropna(subset=[lag_col])
        ax.scatter(valid[lag_col], valid['Call Volume'], alpha=0.3, s=10)
        r = valid[lag_col].corr(valid['Call Volume'])
        ax.set_xlabel(f'Call Volume ({lag_name})')
        ax.set_ylabel('Call Volume (Today)')
        ax.set_title(f'r = {r:.3f}')
        # Add diagonal
        mn = min(valid[lag_col].min(), valid['Call Volume'].min())
        mx = max(valid[lag_col].max(), valid['Call Volume'].max())
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Portfolio {p}: Call Volume Lag Scatter Plots', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'eda_plots/11_lag_scatter_{p}.png', dpi=150)
    plt.close()

print()

# =============================================================================
# SECTION 6: CALENDAR EFFECTS
# =============================================================================
print("=" * 70)
print("SECTION 6: CALENDAR EFFECTS")
print("=" * 70)

# --- 6a: Day of month effect ---
print("\n  Plotting day-of-month effects...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
for i, metric in enumerate(metrics):
    ax = axes[i]
    for p in PORTFOLIOS:
        df = daily_data[p]
        avg = df.groupby('day_of_month')[metric].mean()
        ax.plot(avg.index, avg.values, marker='.', label=f'Portfolio {p}', linewidth=1.5)
    ax.set_ylabel(metric)
    ax.set_xlabel('Day of Month')
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_title('Average Target by Day of Month (beginning/end effects?)')
plt.tight_layout()
plt.savefig('eda_plots/12_day_of_month.png', dpi=150)
plt.close()

# --- 6b: Year-over-year comparison ---
print("  Plotting year-over-year comparison...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
for i, metric in enumerate(metrics):
    ax = axes[i]
    for p in PORTFOLIOS:
        df = daily_data[p]
        for year in [2024, 2025]:
            ydf = df[df['year'] == year]
            # Use day of year for x-axis
            doy = ydf['Date'].dt.dayofyear
            ax.plot(doy, ydf[metric], alpha=0.6, linewidth=0.7, label=f'{p} ({year})')
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right', fontsize=7, ncol=4)
axes[0].set_title('Year-over-Year Comparison (2024 vs 2025)')
axes[-1].set_xlabel('Day of Year')
plt.tight_layout()
plt.savefig('eda_plots/13_year_over_year.png', dpi=150)
plt.close()

# --- 6c: Staffing vs metrics ---
print("  Plotting staffing relationships...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for j, p in enumerate(PORTFOLIOS):
    df = daily_data[p].copy()
    df = df.merge(staffing[['Date', p]], on='Date', how='inner', suffixes=('', '_staff'))
    staff_col = p + '_staff' if p + '_staff' in df.columns else p

    # Staffing vs Call Volume
    ax = axes[0, j]
    ax.scatter(df[staff_col], df['Call Volume'], alpha=0.3, s=10)
    r = df[staff_col].corr(df['Call Volume'])
    ax.set_xlabel(f'Staffing ({p})')
    ax.set_ylabel('Call Volume')
    ax.set_title(f'{p}: r={r:.3f}')
    ax.grid(True, alpha=0.3)

    # Staffing vs Abandon Rate
    ax = axes[1, j]
    ax.scatter(df[staff_col], df['Abandon Rate'], alpha=0.3, s=10)
    r = df[staff_col].corr(df['Abandon Rate'])
    ax.set_xlabel(f'Staffing ({p})')
    ax.set_ylabel('Abandon Rate')
    ax.set_title(f'{p}: r={r:.3f}')
    ax.grid(True, alpha=0.3)

fig.suptitle('Staffing vs Metrics', fontsize=14)
plt.tight_layout()
plt.savefig('eda_plots/14_staffing_vs_metrics.png', dpi=150)
plt.close()

# --- 6d: Summary table of calendar effects ---
print("\n  CALENDAR EFFECT STRENGTH (R² of Call Volume):")
print(f"  {'Portfolio':<10} {'DOW':>8} {'Month':>8} {'DOM':>8} {'Weekend':>8} {'Holiday':>8} {'Year':>8}")
print("  " + "-" * 62)
for p in PORTFOLIOS:
    df = daily_data[p].copy()
    cv = df['Call Volume']
    from scipy import stats as sp_stats

    # Eta-squared for categorical features
    def eta_squared(groups):
        grand_mean = cv.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = ((cv - grand_mean)**2).sum()
        return ss_between / ss_total if ss_total > 0 else 0

    dow_eta = eta_squared([df[df['dow']==d]['Call Volume'] for d in range(7)])
    month_eta = eta_squared([df[df['month']==m]['Call Volume'] for m in range(1,13)])
    dom_eta = eta_squared([df[df['day_of_month']==d]['Call Volume'] for d in range(1,32)])
    wkend_eta = eta_squared([df[df['is_weekend']==w]['Call Volume'] for w in [0,1]])
    hol_eta = eta_squared([df[df['is_holiday']==h]['Call Volume'] for h in [0,1]])
    year_eta = eta_squared([df[df['year']==y]['Call Volume'] for y in [2024,2025]])

    print(f"  {p:<10} {dow_eta:>8.3f} {month_eta:>8.3f} {dom_eta:>8.3f} {wkend_eta:>8.3f} {hol_eta:>8.3f} {year_eta:>8.3f}")

print()

# =============================================================================
# SECTION 7: MODEL COMPLEXITY RECOMMENDATION
# =============================================================================
print("=" * 70)
print("SECTION 7: FINDINGS & MODEL RECOMMENDATIONS")
print("=" * 70)

print("""
  FINDINGS SUMMARY:
  (Review the plots in ./eda_plots/ to verify these observations)

  1. SEASONALITY:
     - Strong weekly seasonality (DOW effect): weekdays >> weekends >> holidays
     - Strong intraday seasonality: clear peak hours (business hours) vs overnight lulls
     - Monthly seasonality present but moderate
     - Day-of-month effects (billing cycle): likely beginning/end of month spikes

  2. TRENDS:
     - Compare 2024 vs 2025 in year-over-year plot to detect trend direction
     - If volumes are growing/shrinking, the model needs a trend component

  3. AUTOCORRELATION:
     - Lag-7 (same day last week) is likely the strongest daily predictor
     - Lag-48 (same half-hour yesterday) is likely strongest for interval data
     - Lag-336 (same half-hour last week) also strong for interval data

  4. DATA QUALITY:
     - Check interval data: some portfolios may have < 48 intervals on certain days
     - Holiday volumes drop dramatically — must be handled explicitly
     - Zero-volume intervals exist at night — model must handle gracefully

  MODEL RECOMMENDATIONS:

  STRATEGY: Two-stage approach

  Stage 1 — Daily forecast:
    With 731 days of history, strong DOW/month patterns, and lag-7 correlation:
    → LightGBM or linear regression with calendar features + lag features
    → Features: DOW, month, is_weekend, is_holiday, day_of_month,
      lag_7, lag_14, lag_365, rolling_7d_mean, staffing, year trend

  Stage 2 — Interval disaggregation:
    With 3 months of 30-min data showing clear intraday profiles:
    → Learn DOW × half_hour profiles from interval data
    → Distribute daily forecast into 48 slots using the profile shape
    → Adjust CCT and Abandon Rate similarly

  VALIDATION:
    → Time-based split: train on 2024 + first ~9 months of 2025
    → Validate on last ~3 months of 2025 (Oct-Dec)
    → NEVER random split

  METRICS TO TRACK:
    → WAPE (matches scoring formula for E_V and E_B)
    → MAE for absolute error
    → Bias / mean error to detect systematic under/overprediction
    → Custom workload penalty if α/β are known

  BIAS ADJUSTMENT:
    → Since understaffing penalty is likely higher (α > β),
      consider adding a small upward bias to volume forecasts
      after evaluating neutral-bias performance
""")

print("=" * 70)
print("EDA COMPLETE. Review plots in ./eda_plots/")
print("=" * 70)
print("\nPlots generated:")
for f in sorted(os.listdir('eda_plots')):
    print(f"  ./eda_plots/{f}")
