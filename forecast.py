"""
Synchrony Datathon - Intraday Contact Center Forecasting
Predicts Call Volume, CCT, and Abandon Rate at 30-min intervals for August 2025
for 4 portfolios (A, B, C, D).

Approach:
  1. Daily-level forecast using historical patterns (2 yrs of daily data)
  2. Interval disaggregation using learned intraday profiles (Apr-Jun interval data)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

DATA_FILE = "Data for Datathon (Revised).xlsx"
TEMPLATE_FILE = "template_forecast_v00.csv"
PORTFOLIOS = ['A', 'B', 'C', 'D']

# Load daily data for each portfolio
daily_data = {}
for p in PORTFOLIOS:
    df = pd.read_excel(DATA_FILE, sheet_name=f"{p} - Daily")
    # Parse date: format is "01/01/24 Mon"
    df['Date'] = pd.to_datetime(df['Date'].str.strip().str[:8], format='%m/%d/%y')
    df = df.sort_values('Date').reset_index(drop=True)
    daily_data[p] = df
    print(f"  Portfolio {p} daily: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")

# Load interval data for each portfolio
interval_data = {}
for p in PORTFOLIOS:
    df = pd.read_excel(DATA_FILE, sheet_name=f"{p} - Interval")
    # Convert Interval to string and extract hour:minute
    df = df.dropna(subset=['Interval', 'Call Volume']).reset_index(drop=True)
    df['Interval'] = df['Interval'].astype(str)
    # Handle timedelta format (0 days HH:MM:SS) and time format (HH:MM:SS)
    def parse_interval(val):
        val = str(val).strip()
        if 'days' in val:
            # timedelta string like "0 days 00:30:00"
            parts = val.split(' ')
            return parts[-1][:5]  # "00:30"
        elif len(val) >= 5:
            return val[:5]
        return val
    df['Interval_str'] = df['Interval'].apply(parse_interval)
    # Create hour index (0-47)
    df['half_hour'] = df['Interval_str'].apply(
        lambda x: int(x.split(':')[0]) * 2 + (1 if int(x.split(':')[1]) >= 30 else 0)
    )
    interval_data[p] = df
    print(f"  Portfolio {p} interval: {len(df)} rows, months: {df['Month'].unique()}")

# Load staffing data
staffing = pd.read_excel(DATA_FILE, sheet_name="Daily Staffing")
staffing.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
staffing['Date'] = pd.to_datetime(staffing['Date'])
print(f"  Staffing: {len(staffing)} rows, {staffing['Date'].min().date()} to {staffing['Date'].max().date()}")

# Load template
template = pd.read_csv(TEMPLATE_FILE)
print(f"  Template: {template.shape[0]} rows, {template.shape[1]} columns")
print(f"  Forecast month: {template['Month'].iloc[0]}, days: {template['Day'].min()}-{template['Day'].max()}")

print()

# =============================================================================
# STEP 2: Feature Engineering
# =============================================================================
print("=" * 60)
print("STEP 2: Feature engineering...")
print("=" * 60)

# US Federal holidays (approximate)
US_HOLIDAYS_2025 = [
    datetime(2025, 1, 1),   # New Year's Day
    datetime(2025, 1, 20),  # MLK Day
    datetime(2025, 2, 17),  # Presidents' Day
    datetime(2025, 5, 26),  # Memorial Day
    datetime(2025, 7, 4),   # Independence Day
    datetime(2025, 9, 1),   # Labor Day
    datetime(2025, 10, 13), # Columbus Day
    datetime(2025, 11, 11), # Veterans Day
    datetime(2025, 11, 27), # Thanksgiving
    datetime(2025, 12, 25), # Christmas
]
US_HOLIDAYS_2024 = [
    datetime(2024, 1, 1),   # New Year's Day
    datetime(2024, 1, 15),  # MLK Day
    datetime(2024, 2, 19),  # Presidents' Day
    datetime(2024, 5, 27),  # Memorial Day
    datetime(2024, 7, 4),   # Independence Day
    datetime(2024, 9, 2),   # Labor Day
    datetime(2024, 10, 14), # Columbus Day
    datetime(2024, 11, 11), # Veterans Day
    datetime(2024, 11, 28), # Thanksgiving
    datetime(2024, 12, 25), # Christmas
]
US_HOLIDAYS_2026 = [
    datetime(2026, 1, 1),   # New Year's Day
    datetime(2026, 1, 19),  # MLK Day
    datetime(2026, 2, 16),  # Presidents' Day
    datetime(2026, 5, 25),  # Memorial Day
    datetime(2026, 7, 3),   # Independence Day (observed, July 4 is Sat)
    datetime(2026, 7, 4),   # Independence Day
    datetime(2026, 9, 7),   # Labor Day
    datetime(2026, 10, 12), # Columbus Day
    datetime(2026, 11, 11), # Veterans Day
    datetime(2026, 11, 26), # Thanksgiving
    datetime(2026, 12, 25), # Christmas
]
ALL_HOLIDAYS = set(US_HOLIDAYS_2024 + US_HOLIDAYS_2025 + US_HOLIDAYS_2026)

def add_daily_features(df):
    """Add time-based features to a daily DataFrame."""
    df = df.copy()
    df['dow'] = df['Date'].dt.dayofweek       # 0=Mon, 6=Sun
    df['month'] = df['Date'].dt.month
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_holiday'] = df['Date'].isin(ALL_HOLIDAYS).astype(int)
    # Day after holiday
    df['is_day_after_holiday'] = df['Date'].apply(
        lambda d: 1 if (d - timedelta(days=1)) in ALL_HOLIDAYS else 0
    )
    return df

for p in PORTFOLIOS:
    daily_data[p] = add_daily_features(daily_data[p])

print("  Added daily features (DOW, month, holiday flags, etc.)")

# =============================================================================
# STEP 3: Learn Intraday Profiles from Interval Data
# =============================================================================
print()
print("=" * 60)
print("STEP 3: Learning intraday profiles from interval data...")
print("=" * 60)

# Map month names to numbers for interval data
MONTH_MAP = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
             'May': 5, 'June': 6, 'July': 7, 'August': 8,
             'September': 9, 'October': 10, 'November': 11, 'December': 12}

def build_intraday_profiles(interval_df, daily_df):
    """
    Build intraday profiles: for each DOW × half_hour, what fraction of daily
    volume falls in that interval?

    Returns profiles for Call Volume, CCT, and Abandon Rate.
    """
    idf = interval_df.copy()
    idf['month_num'] = idf['Month'].map(MONTH_MAP)

    # We need to figure out which year these belong to. The interval data has
    # Month + Day but no year. From the daily data we know data spans 2024-2025.
    # The interval months are April, May, June. Let's match to 2025 since
    # the daily data goes through Dec 2025 and interval data is likely the most recent.
    # We'll try to create dates and match with daily data.

    dates_list = []
    for _, row in idf.iterrows():
        month_num = row['month_num']
        day = row['Day']
        # Try 2025 first, fall back to 2024
        for year in [2025, 2024]:
            try:
                dt = datetime(year, month_num, day)
                dates_list.append(dt)
                break
            except ValueError:
                if year == 2024:
                    dates_list.append(None)
    idf['Date'] = dates_list
    idf['dow'] = idf['Date'].apply(lambda d: d.weekday() if d else None)

    # Compute daily totals from interval data
    daily_totals = idf.groupby('Date').agg(
        daily_cv=('Call Volume', 'sum'),
        daily_abd_calls=('Abandoned Calls', 'sum')
    ).reset_index()

    idf = idf.merge(daily_totals, on='Date', how='left')

    # Volume profile: fraction of daily volume per interval
    idf['cv_frac'] = idf['Call Volume'] / idf['daily_cv'].replace(0, np.nan)

    # Build profiles by DOW × half_hour
    # For volume: average fraction by DOW × half_hour
    vol_profile = idf.groupby(['dow', 'half_hour'])['cv_frac'].mean().reset_index()
    vol_profile.columns = ['dow', 'half_hour', 'cv_frac']

    # For CCT: average CCT by DOW × half_hour (weighted by call volume)
    def weighted_mean(group):
        weights = group['Call Volume'].fillna(0)
        if weights.sum() == 0:
            return group['CCT'].mean()
        return np.average(group['CCT'].fillna(0), weights=weights)

    cct_profile = idf.groupby(['dow', 'half_hour']).apply(weighted_mean).reset_index()
    cct_profile.columns = ['dow', 'half_hour', 'cct_avg']

    # For Abandon Rate: average by DOW × half_hour (weighted by call volume)
    def weighted_mean_abd(group):
        weights = group['Call Volume'].fillna(0)
        if weights.sum() == 0:
            return group['Abandoned Rate'].mean()
        return np.average(group['Abandoned Rate'].fillna(0), weights=weights)

    abd_profile = idf.groupby(['dow', 'half_hour']).apply(weighted_mean_abd).reset_index()
    abd_profile.columns = ['dow', 'half_hour', 'abd_avg']

    return vol_profile, cct_profile, abd_profile

intraday_profiles = {}
for p in PORTFOLIOS:
    vol_prof, cct_prof, abd_prof = build_intraday_profiles(interval_data[p], daily_data[p])
    intraday_profiles[p] = {
        'volume': vol_prof,
        'cct': cct_prof,
        'abandon': abd_prof
    }
    print(f"  Portfolio {p}: volume profile shape={vol_prof.shape}, "
          f"CCT profile shape={cct_prof.shape}, abandon profile shape={abd_prof.shape}")

# =============================================================================
# STEP 4: Daily-Level Forecasting for August 2026
# =============================================================================
print()
print("=" * 60)
print("STEP 4: Daily-level forecasting for August 2026...")
print("=" * 60)

# August 2026 dates (the forecast target — data ends Dec 2025)
FORECAST_MONTH = 'August'
forecast_dates = pd.date_range('2026-08-01', '2026-08-31', freq='D')
aug_df = pd.DataFrame({'Date': forecast_dates})
aug_df = add_daily_features(aug_df)

# No staffing data for 2026 — estimate from Aug 2025 staffing
aug_2025_staffing = staffing[(staffing['Date'].dt.month == 8)].copy()
aug_2025_staffing['dow'] = aug_2025_staffing['Date'].dt.dayofweek
aug_staff_by_dow = aug_2025_staffing.groupby('dow')[PORTFOLIOS].mean()

for p in PORTFOLIOS:
    aug_df[p] = aug_df['dow'].map(aug_staff_by_dow[p])
    aug_df[p] = aug_df[p].fillna(aug_2025_staffing[p].mean())

def forecast_daily(daily_df, portfolio, staffing_df, target_dates):
    """
    Forecast daily metrics using historical pattern matching.

    Strategy:
    - For each target date, find similar historical days based on:
      DOW, month proximity, holiday status
    - Weight by recency and similarity
    - Use staffing as an adjustment factor
    """
    hist = daily_df.copy()
    results = []

    for _, target_row in target_dates.iterrows():
        t_date = target_row['Date']
        t_dow = target_row['dow']
        t_month = target_row['month']  # August = 8
        t_is_holiday = target_row['is_holiday']
        t_is_weekend = target_row['is_weekend']
        t_dom = target_row['day_of_month']

        # Find similar days: same DOW, similar time of year
        similar = hist.copy()

        # Must match: weekend/weekday
        similar = similar[similar['is_weekend'] == t_is_weekend]

        # If holiday, match holidays; if not, exclude holidays
        if t_is_holiday:
            similar = similar[similar['is_holiday'] == 1]
        else:
            similar = similar[similar['is_holiday'] == 0]

        # Prefer same DOW
        same_dow = similar[similar['dow'] == t_dow]
        if len(same_dow) >= 8:
            similar = same_dow

        # Weight by month proximity (prefer July/Aug/Sep from both years)
        similar['month_dist'] = similar['month'].apply(
            lambda m: min(abs(m - t_month), 12 - abs(m - t_month))
        )

        # Weight by recency (more recent = higher weight)
        similar['days_ago'] = (t_date - similar['Date']).dt.days
        similar = similar[similar['days_ago'] > 0]  # only past data

        # Composite weight: month proximity + recency
        similar['weight'] = np.exp(-similar['month_dist'] / 2.0) * np.exp(-similar['days_ago'] / 365.0)

        # Also boost same month from previous year
        similar.loc[similar['month'] == t_month, 'weight'] *= 2.0

        # Also consider day-of-month effects (beginning/end of month)
        dom_dist = abs(similar['day_of_month'] - t_dom)
        similar['weight'] *= np.exp(-dom_dist / 15.0)

        if len(similar) == 0:
            # Fallback: just use overall mean
            results.append({
                'Date': t_date,
                'Call Volume': hist['Call Volume'].mean(),
                'CCT': hist['CCT'].mean(),
                'Abandon Rate': hist['Abandon Rate'].mean()
            })
            continue

        # Weighted average
        w = similar['weight'].values
        w = w / w.sum()

        cv = np.average(similar['Call Volume'].fillna(0).values, weights=w)
        cct = np.average(similar['CCT'].fillna(0).values, weights=w)
        abd = np.average(similar['Abandon Rate'].fillna(0).values, weights=w)

        results.append({
            'Date': t_date,
            'Call Volume': cv,
            'CCT': cct,
            'Abandon Rate': abd
        })

    return pd.DataFrame(results)

# Adjust forecasts using staffing ratios
def adjust_for_staffing(forecast_df, portfolio, daily_df, staffing_df, target_df):
    """
    If August 2026 estimated staffing differs from historical staffing levels,
    adjust CCT and Abandon Rate accordingly.
    Uses estimated staffing from target_df (aug_df) which was derived from Aug 2025.
    """
    # Get average staffing for the historical period (2025)
    hist_2025 = daily_df[daily_df['Date'].dt.year == 2025].copy()
    hist_2025 = hist_2025.merge(staffing_df[['Date', portfolio]], on='Date', how='inner')

    if len(hist_2025) == 0:
        return forecast_df

    # Average staffing by DOW for full year 2025
    hist_staff_by_dow = hist_2025.groupby('dow')[portfolio].mean()

    forecast_df = forecast_df.copy()
    forecast_df['dow'] = forecast_df['Date'].dt.dayofweek

    for idx, row in forecast_df.iterrows():
        # Get estimated staffing from target_df (aug_df has staffing columns)
        target_match = target_df.loc[target_df['Date'] == row['Date'], portfolio]
        if len(target_match) == 0:
            continue
        aug_staff = target_match.values[0]
        hist_staff = hist_staff_by_dow.get(row['dow'], aug_staff)

        if hist_staff > 0 and aug_staff > 0:
            staff_ratio = aug_staff / hist_staff
            if staff_ratio != 1.0:
                forecast_df.at[idx, 'Abandon Rate'] *= (1.0 / staff_ratio) ** 0.5
                forecast_df.at[idx, 'CCT'] *= (1.0 / staff_ratio) ** 0.15

    return forecast_df

daily_forecasts = {}
for p in PORTFOLIOS:
    fc = forecast_daily(daily_data[p], p, staffing, aug_df)
    fc = adjust_for_staffing(fc, p, daily_data[p], staffing, aug_df)
    daily_forecasts[p] = fc
    print(f"  Portfolio {p} daily forecast:")
    print(f"    Call Volume: mean={fc['Call Volume'].mean():.0f}, "
          f"range=[{fc['Call Volume'].min():.0f}, {fc['Call Volume'].max():.0f}]")
    print(f"    CCT: mean={fc['CCT'].mean():.1f}, "
          f"range=[{fc['CCT'].min():.1f}, {fc['CCT'].max():.1f}]")
    print(f"    Abandon Rate: mean={fc['Abandon Rate'].mean():.4f}, "
          f"range=[{fc['Abandon Rate'].min():.4f}, {fc['Abandon Rate'].max():.4f}]")

# =============================================================================
# STEP 5: Interval Disaggregation
# =============================================================================
print()
print("=" * 60)
print("STEP 5: Disaggregating daily forecasts to 30-min intervals...")
print("=" * 60)

def disaggregate_to_intervals(daily_fc, profiles, portfolio):
    """
    Distribute daily forecasts into 30-min intervals using learned profiles.
    """
    vol_prof = profiles['volume']
    cct_prof = profiles['cct']
    abd_prof = profiles['abandon']

    rows = []
    for _, day_row in daily_fc.iterrows():
        date = day_row['Date']
        dow = date.weekday()
        daily_cv = day_row['Call Volume']
        daily_cct = day_row['CCT']
        daily_abd = day_row['Abandon Rate']

        for hh in range(48):
            # Get volume fraction for this DOW × half_hour
            mask = (vol_prof['dow'] == dow) & (vol_prof['half_hour'] == hh)
            cv_frac_vals = vol_prof.loc[mask, 'cv_frac']
            if len(cv_frac_vals) > 0:
                cv_frac = cv_frac_vals.values[0]
            else:
                # Fallback: uniform distribution
                cv_frac = 1.0 / 48.0

            # Interval call volume
            interval_cv = daily_cv * cv_frac

            # CCT from profile
            cct_mask = (cct_prof['dow'] == dow) & (cct_prof['half_hour'] == hh)
            cct_vals = cct_prof.loc[cct_mask, 'cct_avg']
            if len(cct_vals) > 0:
                interval_cct = cct_vals.values[0]
                # Adjust relative to profile's daily mean vs our forecasted daily mean
                # This preserves the intraday shape but matches the daily level
                dow_cct_mean = cct_prof.loc[cct_prof['dow'] == dow, 'cct_avg'].mean()
                if dow_cct_mean > 0:
                    interval_cct = interval_cct * (daily_cct / dow_cct_mean)
            else:
                interval_cct = daily_cct

            # Abandon rate from profile
            abd_mask = (abd_prof['dow'] == dow) & (abd_prof['half_hour'] == hh)
            abd_vals = abd_prof.loc[abd_mask, 'abd_avg']
            if len(abd_vals) > 0:
                interval_abd = abd_vals.values[0]
                # Adjust to match daily level
                dow_abd_mean = abd_prof.loc[abd_prof['dow'] == dow, 'abd_avg'].mean()
                if dow_abd_mean > 0:
                    interval_abd = interval_abd * (daily_abd / dow_abd_mean)
            else:
                interval_abd = daily_abd

            # Format interval string
            hour = hh // 2
            minute = (hh % 2) * 30
            interval_str = f"{hour}:{minute:02d}"

            rows.append({
                'Date': date,
                'Day': date.day,
                'half_hour': hh,
                'Interval': interval_str,
                'Call Volume': max(0, round(interval_cv, 2)),
                'CCT': max(0, round(interval_cct, 2)),
                'Abandon Rate': max(0, round(interval_abd, 6)),
                'Abandoned Calls': max(0, round(interval_cv * interval_abd, 2))
            })

    return pd.DataFrame(rows)

interval_forecasts = {}
for p in PORTFOLIOS:
    ifc = disaggregate_to_intervals(
        daily_forecasts[p], intraday_profiles[p], p
    )
    interval_forecasts[p] = ifc
    print(f"  Portfolio {p}: {len(ifc)} interval forecasts generated")
    print(f"    CV range: [{ifc['Call Volume'].min():.1f}, {ifc['Call Volume'].max():.1f}]")
    print(f"    CCT range: [{ifc['CCT'].min():.1f}, {ifc['CCT'].max():.1f}]")
    print(f"    ABD range: [{ifc['Abandon Rate'].min():.6f}, {ifc['Abandon Rate'].max():.6f}]")

# =============================================================================
# STEP 6: Fill the Template
# =============================================================================
print()
print("=" * 60)
print("STEP 6: Filling the output template...")
print("=" * 60)

output = template.copy()

# Normalize template interval format for matching
def normalize_interval(val):
    """Convert template interval like '0:00' or '13:30' to match our format."""
    val = str(val).strip()
    return val

output['Interval_match'] = output['Interval'].apply(normalize_interval)

for p in PORTFOLIOS:
    ifc = interval_forecasts[p]

    for idx, row in output.iterrows():
        day = row['Day']
        interval = row['Interval_match']

        # Find matching forecast row
        match = ifc[(ifc['Day'] == day) & (ifc['Interval'] == interval)]

        if len(match) == 0:
            continue

        m = match.iloc[0]
        output.at[idx, f'Calls_Offered_{p}'] = round(m['Call Volume'], 2)
        output.at[idx, f'Abandoned_Calls_{p}'] = round(m['Abandoned Calls'], 2)
        output.at[idx, f'Abandoned_Rate_{p}'] = round(m['Abandon Rate'], 6)
        output.at[idx, f'CCT_{p}'] = round(m['CCT'], 2)

# Drop helper column
output.drop(columns=['Interval_match'], inplace=True)

# Validate: no NaN values
nan_count = output.iloc[:, 3:].isna().sum().sum()
if nan_count > 0:
    print(f"  WARNING: {nan_count} NaN values remain in output!")
    # Fill remaining NaN with 0
    output.iloc[:, 3:] = output.iloc[:, 3:].fillna(0)
else:
    print("  All cells filled successfully!")

# Validate: no negative values
neg_count = (output.iloc[:, 3:] < 0).sum().sum()
print(f"  Negative values: {neg_count}")

# Save output
OUTPUT_FILE = "submission_forecast.csv"
output.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved to {OUTPUT_FILE}")

# Print summary statistics
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for p in PORTFOLIOS:
    cv_col = f'Calls_Offered_{p}'
    cct_col = f'CCT_{p}'
    abd_col = f'Abandoned_Rate_{p}'
    print(f"\n  Portfolio {p}:")
    print(f"    Call Volume:  total={output[cv_col].sum():.0f}, "
          f"daily avg={output[cv_col].sum()/31:.0f}")
    print(f"    CCT:          mean={output[cct_col].mean():.1f}")
    print(f"    Abandon Rate: mean={output[abd_col].mean():.4f}")

print()
print("Done! Output file:", OUTPUT_FILE)
