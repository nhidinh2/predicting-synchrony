"""
Synchrony Datathon — Data Cleaning Pipeline
=============================================
Cleans daily and interval data, saves cleaned versions as pickle files.

Daily data issues:
  - NaN streaks of 5-13 consecutive days (data collection outages)
  - Metrics go missing independently (different recording systems)
  - Call Volume & CCT NaN together, Abandon Rate sometimes separate

Interval data issues:
  - Missing half-hour slots (off-hours dropped from source)
  - CCT/Abandon Rate NaN where Call Volume = 0 (expected — no calls)
  - CCT/Abandon Rate NaN where Call Volume > 0 (recording failure)

Strategy:
  - Daily Call Volume & CCT: impute using same-DOW median from +/- 8 weeks
  - Daily Abandon Rate: LEAVE AS NaN — too noisy (60-186% CV within DOW),
    nearly uncorrelated with volume/staffing, DOW median is meaningless
  - Interval missing slots: fill Call Volume with 0, CCT and AR with NaN
  - Interval CCT NaN at zero volume: set to 0 (no calls = no handle time)
  - Interval CCT NaN at nonzero volume: impute with same DOW x slot median
  - Interval Abandon Rate: LEAVE AS NaN everywhere it's missing — same
    reasoning as daily, and the model should learn to skip these rows
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = "Data for Datathon (Revised).xlsx"
PORTFOLIOS = ['A', 'B', 'C', 'D']
MONTH_MAP = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
             'May': 5, 'June': 6, 'July': 7, 'August': 8,
             'September': 9, 'October': 10, 'November': 11, 'December': 12}

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

DAILY_METRICS = ['Call Volume', 'CCT', 'Abandon Rate']
INTERVAL_METRICS = ['Call Volume', 'CCT', 'Abandoned Rate']

# =============================================================================
# LOAD RAW DATA
# =============================================================================
print("Loading raw data...")

daily_data = {}
for p in PORTFOLIOS:
    df = pd.read_excel(DATA_FILE, sheet_name=f"{p} - Daily")
    df['Date'] = pd.to_datetime(df['Date'].str.strip().str[:8], format='%m/%d/%y')
    df = df.sort_values('Date').reset_index(drop=True)
    df['dow'] = df['Date'].dt.dayofweek
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

print("Raw data loaded.\n")

# =============================================================================
# CLEAN DAILY DATA
# =============================================================================
print("=" * 70)
print("CLEANING DAILY DATA")
print("=" * 70)

IMPUTE_METRICS = ['Call Volume', 'CCT']  # NOT Abandon Rate — too noisy

cleaned_daily = {}
for p in PORTFOLIOS:
    df = daily_data[p].copy()
    n_before = df[DAILY_METRICS].isna().sum()

    for metric in IMPUTE_METRICS:
        nan_mask = df[metric].isna()
        if nan_mask.sum() == 0:
            continue

        # Impute each NaN using median of same DOW from 8 weeks before and after
        for idx in df[nan_mask].index:
            row = df.loc[idx]
            target_dow = row['dow']
            target_date = row['Date']

            # Get same-DOW rows within 8 weeks of the NaN, excluding NaN rows
            window = pd.Timedelta(weeks=8)
            candidates = df[
                (df['dow'] == target_dow) &
                (df[metric].notna()) &
                (df['Date'] >= target_date - window) &
                (df['Date'] <= target_date + window)
            ]

            if len(candidates) > 0:
                df.loc[idx, metric] = candidates[metric].median()
            else:
                # Fallback: global same-DOW median
                fallback = df[(df['dow'] == target_dow) & (df[metric].notna())]
                df.loc[idx, metric] = fallback[metric].median()

    # Abandon Rate: left as NaN intentionally
    n_after = df[DAILY_METRICS].isna().sum()

    print(f"\n  Portfolio {p}:")
    for metric in DAILY_METRICS:
        action = "imputed" if metric in IMPUTE_METRICS else "left as NaN (too noisy)"
        print(f"    {metric}: {n_before[metric]} -> {n_after[metric]} NaN ({action})")

    cleaned_daily[p] = df

# =============================================================================
# CLEAN INTERVAL DATA
# =============================================================================
print("\n" + "=" * 70)
print("CLEANING INTERVAL DATA")
print("=" * 70)

cleaned_interval = {}
for p in PORTFOLIOS:
    df = interval_data[p].copy()

    # --- Step 1: Fill missing half-hour slots with zero-volume rows ---
    all_dates = sorted(df['Date'].unique())
    all_slots = list(range(48))
    full_grid = pd.MultiIndex.from_product([all_dates, all_slots], names=['Date', 'half_hour'])
    full_df = pd.DataFrame(index=full_grid).reset_index()

    df = full_df.merge(df, on=['Date', 'half_hour'], how='left')

    # Fill missing slots: CV=0, CCT/AR=NaN for new rows (off-hours)
    n_added = df['Call Volume'].isna().sum()
    df['Call Volume'] = df['Call Volume'].fillna(0)
    # CCT at zero volume = 0 (no calls = no handle time)
    # CCT at missing slots (also zero volume) = 0
    df.loc[df['Call Volume'] == 0, 'CCT'] = df.loc[df['Call Volume'] == 0, 'CCT'].fillna(0)
    # Abandoned Rate: leave as NaN everywhere it's missing

    # Recompute calendar columns for new rows
    df['dow'] = df['Date'].dt.dayofweek
    df['dow_name'] = df['Date'].dt.day_name()

    # Fill other columns for the new rows
    if 'Month' in df.columns:
        df['Month'] = df['Month'].fillna(df['Date'].dt.month_name())
    if 'Day' in df.columns:
        df['Day'] = df['Day'].fillna(df['Date'].dt.day)

    # --- Step 2: Fix CCT NaN where Call Volume > 0 ---
    # Abandon Rate is left as NaN — too noisy to impute reliably
    orig = interval_data[p].copy()
    nonzero_cct_nan = orig[(orig['Call Volume'] > 0) & (orig['CCT'].isna())]

    # Compute DOW x half_hour medians from clean rows
    clean_rows = orig[(orig['Call Volume'] > 0) & (orig['CCT'].notna())]
    cct_medians = clean_rows.groupby(['dow', 'half_hour'])['CCT'].median()

    cct_fixed = 0
    for _, row in nonzero_cct_nan.iterrows():
        mask = (df['Date'] == row['Date']) & (df['half_hour'] == row['half_hour'])
        key = (row['dow'], row['half_hour'])
        if key in cct_medians.index:
            df.loc[mask, 'CCT'] = cct_medians[key]
            cct_fixed += 1

    df = df.sort_values(['Date', 'half_hour']).reset_index(drop=True)

    print(f"\n  Portfolio {p}:")
    print(f"    Missing slots filled (zero-volume): {n_added}")
    print(f"    CCT imputed (nonzero vol): {cct_fixed}")
    print(f"    Abandon Rate: left as NaN where missing (not imputed)")
    print(f"    Final rows: {len(df)} (expected: {len(all_dates)} days x 48 = {len(all_dates)*48})")
    print(f"    Remaining NaN - CV: {df['Call Volume'].isna().sum()}, CCT: {df['CCT'].isna().sum()}, AR: {df['Abandoned Rate'].isna().sum()}")

    cleaned_interval[p] = df

# =============================================================================
# VERIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

print("\n  DAILY — spot check imputed values make sense:")
for p in PORTFOLIOS:
    orig = daily_data[p]
    clean = cleaned_daily[p]
    changed = orig['Call Volume'].isna() & clean['Call Volume'].notna()
    ar_still_nan = clean['Abandon Rate'].isna().sum()
    if changed.sum() > 0:
        print(f"\n  Portfolio {p} — imputed daily rows (AR remains NaN: {ar_still_nan}):")
        sample = clean[changed].head(5)
        for _, row in sample.iterrows():
            dow_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            same_dow = clean[(clean['dow'] == row['dow']) & (clean['Call Volume'].notna())]
            dow_med = same_dow['Call Volume'].median()
            dow_std = same_dow['Call Volume'].std()
            ar_str = f"{row['Abandon Rate']:.4f}" if pd.notna(row['Abandon Rate']) else "NaN"
            print(f"    {row['Date'].date()} ({dow_names[row['dow']]}): "
                  f"CV={row['Call Volume']:.0f} (DOW median={dow_med:.0f}, std={dow_std:.0f}), "
                  f"CCT={row['CCT']:.1f}, AR={ar_str}")

print("\n  INTERVAL — verify grid completeness:")
for p in PORTFOLIOS:
    df = cleaned_interval[p]
    per_day = df.groupby('Date').size()
    incomplete = per_day[per_day != 48]
    print(f"  Portfolio {p}: {len(per_day)} days, all with 48 slots: {'YES' if len(incomplete) == 0 else f'NO ({len(incomplete)} incomplete)'}")

print("\n  INTERVAL — zero volume distribution:")
for p in PORTFOLIOS:
    df = cleaned_interval[p]
    zero_pct = (df['Call Volume'] == 0).sum() / len(df) * 100
    # By hour bucket
    df['hour'] = df['half_hour'] // 2
    zero_by_hour = df.groupby('hour').apply(lambda x: (x['Call Volume'] == 0).mean() * 100)
    peak_zero = zero_by_hour.idxmax()
    print(f"  Portfolio {p}: {zero_pct:.1f}% zero-vol overall, highest zero rate at hour {peak_zero}:00 ({zero_by_hour.max():.0f}%)")

# =============================================================================
# SAVE
# =============================================================================
print("\n" + "=" * 70)
print("SAVING CLEANED DATA")
print("=" * 70)

for p in PORTFOLIOS:
    cleaned_daily[p].to_pickle(f'cleaned_daily_{p}.pkl')
    cleaned_interval[p].to_pickle(f'cleaned_interval_{p}.pkl')
staffing.to_pickle('cleaned_staffing.pkl')

print("  Saved cleaned_daily_{A,B,C,D}.pkl")
print("  Saved cleaned_interval_{A,B,C,D}.pkl")
print("  Saved cleaned_staffing.pkl")
print("\nDone.")
