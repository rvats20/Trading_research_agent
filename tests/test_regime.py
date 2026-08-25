from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.regime import expanding_median_flag


def test_no_current_value_leakage():
    # past median at t uses only values BEFORE t
    rv = pd.Series([10, 20, 30, 40, 50], dtype=float)
    flag = expanding_median_flag(rv, min_periods=2)
    # t=2 (first row with 2 past obs): past=[10,20] med 15, rv=30 -> True
    assert pd.isna(flag.iloc[1])
    assert flag.iloc[2] == True  # noqa: E712
    assert flag.iloc[3] == True  # noqa: E712


def test_low_value_false():
    rv = pd.Series([50, 40, 30, 20, 10], dtype=float)
    flag = expanding_median_flag(rv, min_periods=2)
    assert pd.isna(flag.iloc[1])
    assert flag.iloc[2] == False  # noqa: E712
    assert flag.iloc[4] == False  # noqa: E712


def test_nan_until_min_periods_observations():
    # min_periods counts PAST observations; after shift(1), row i has i past obs.
    rv = pd.Series([10.0, 20.0, 30.0, 40.0], dtype=float)
    flag = expanding_median_flag(rv, min_periods=3)
    # row 2 has 2 past obs < 3 -> NaN; row 3 has 3 past obs -> valid
    assert pd.isna(flag.iloc[1])
    assert flag.notna().iloc[3]


def test_exact_median_boundary_is_true():
    rv = pd.Series([10.0, 20.0, 20.0, 5.0], dtype=float)
    flag = expanding_median_flag(rv, min_periods=2)
    # t=3 (rv=5), past [10,20,20] med=20 -> False
    assert flag.iloc[3] == False  # noqa: E712
