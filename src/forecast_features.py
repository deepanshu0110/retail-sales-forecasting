"""
Leakage-free features for multi-day sales forecasting.

Every feature here is known before the day being forecast:
calendar fields, the holiday calendar, planned promotions, and sales from
the same weekday one year earlier (lag 364). Nothing uses the target day's
own sales or recent sales that would not exist yet when the forecast is made.

The same function builds features for training, evaluation and the API, so
the model sees identical inputs in all three places.
"""

import numpy as np
import pandas as pd

YEAR_LAG = 364  # same weekday, one year earlier

MAJOR_HOLIDAYS = {(12, 25), (12, 24), (1, 1), (7, 4), (11, 24), (11, 25)}
MINOR_HOLIDAYS = {(2, 14), (3, 17), (10, 31)}

FEATURE_COLUMNS = [
    'time_idx', 'day_of_week', 'month', 'day_of_month', 'day_of_year',
    'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'is_holiday', 'promotion', 'promotion_weekend', 'promotion_holiday',
    'holiday_weekend', f'sales_lag_{YEAR_LAG}',
]


def is_holiday(date):
    """Same holiday rules used to generate the synthetic data."""
    if (date.month, date.day) in MAJOR_HOLIDAYS or (date.month, date.day) in MINOR_HOLIDAYS:
        return 1
    if date.month in (12, 8, 9):
        return 1
    return 0


def build_features(dates, sales_history, start_date, promotions=None):
    """
    Build model features for the given dates.

    dates          : sequence of dates to build features for
    sales_history  : pd.Series of past sales indexed by date (only real
                     observations; the lag lookup never reads the target day)
    start_date     : first date of the store's history, used for time_idx
    promotions     : optional dict/Series {date: 0 or 1} of planned promotions
    """
    dates = pd.to_datetime(pd.Series(list(dates)))
    start_date = pd.Timestamp(start_date)
    promo_lookup = pd.Series(promotions) if promotions is not None else pd.Series(dtype=float)
    if len(promo_lookup):
        promo_lookup.index = pd.to_datetime(promo_lookup.index)

    out = pd.DataFrame({'date': dates.values})
    d = out['date']
    out['time_idx'] = (d - start_date).dt.days
    out['day_of_week'] = d.dt.dayofweek
    out['month'] = d.dt.month
    out['day_of_month'] = d.dt.day
    out['day_of_year'] = d.dt.dayofyear
    out['week_of_year'] = d.dt.isocalendar().week.astype(int).values
    out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)
    out['is_month_start'] = d.dt.is_month_start.astype(int)
    out['is_month_end'] = d.dt.is_month_end.astype(int)
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    out['day_of_week_sin'] = np.sin(2 * np.pi * out['day_of_week'] / 7)
    out['day_of_week_cos'] = np.cos(2 * np.pi * out['day_of_week'] / 7)
    out['is_holiday'] = [is_holiday(x) for x in d]
    out['promotion'] = [int(promo_lookup.get(x, 0)) for x in d]
    out['promotion_weekend'] = out['promotion'] * out['is_weekend']
    out['promotion_holiday'] = out['promotion'] * out['is_holiday']
    out['holiday_weekend'] = out['is_holiday'] * out['is_weekend']

    history = sales_history.copy()
    history.index = pd.to_datetime(history.index)
    # Same weekday one year earlier. Cleaning removes some outlier days, so if that
    # exact day is missing, use the same weekday one or two weeks either side.
    def year_lag(x):
        for offset in (YEAR_LAG, YEAR_LAG + 7, YEAR_LAG - 7, YEAR_LAG + 14, YEAR_LAG - 14):
            value = history.get(x - pd.Timedelta(days=offset), np.nan)
            if not pd.isna(value):
                return value
        return np.nan
    out[f'sales_lag_{YEAR_LAG}'] = [year_lag(x) for x in d]

    return out.set_index('date')[FEATURE_COLUMNS]
