import numpy as np
import pandas as pd

from src.features import true_range, rolling_percentile, add_volatility_features

def sample_ohlc(n=100):
    close = pd.Series(np.linspace(100, 120, n))
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
        "open": close,
        "high": close + 2,
        "low": close - 2,
        "close": close,
    })

def test_true_range_first_row_uses_intraday_range():
    df = sample_ohlc(3)
    tr = true_range(df)
    assert tr.iloc[0] == 4

def test_rolling_percentile_is_bounded():
    s = pd.Series(range(10), dtype=float)
    result = rolling_percentile(s, 5).dropna()
    assert result.between(0, 1).all()

def test_volatility_features_create_expected_columns():
    result = add_volatility_features(sample_ohlc())
    expected = {
        "atr_14", "atr_pct_price", "realized_vol_20",
        "bb_width_20", "atr_pct_price_pct_20",
        "realized_vol_20_pct_60",
    }
    assert expected.issubset(result.columns)
