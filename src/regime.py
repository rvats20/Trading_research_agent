from __future__ import annotations
import pandas as pd


def expanding_median_flag(rv: pd.Series, min_periods: int = 120) -> pd.Series:
    """True if rv[t] >= expanding median of rv[0..t-1] (excludes current value).

    Uses only information available at time t. Days before `min_periods`
    observations are NaN (invalid — no regime call possible yet).
    """
    past_med = rv.shift(1).expanding(min_periods=min_periods).median()
    return (rv >= past_med).where(past_med.notna())
