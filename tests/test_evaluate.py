import numpy as np
import pandas as pd

from src.evaluate import compare_bucket, evaluate

def test_evaluator_returns_expected_number_of_tests():
    n = 300
    df = pd.DataFrame()
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    df["close"] = close

    # Minimal synthetic feature/outcome columns required by evaluator.
    features = [
        "atr_pct_price_pct_20", "atr_pct_price_pct_60",
        "realized_vol_20_pct_20", "realized_vol_20_pct_60",
        "bb_width_20_pct_20", "bb_width_20_pct_60",
    ]
    for f in features:
        df[f] = rng.uniform(0, 1, n)
    for h in [1, 3, 5, 10, 20]:
        df[f"fwd_return_{h}d"] = rng.normal(0, .02, n)

    result = evaluate(df)
    assert len(result["tests"]) == 6 * 3 * 5

def test_compare_bucket_excludes_unknown_feature_state():
    df = pd.DataFrame({"outcome": [100.0, 1.0, 2.0, 3.0]})
    mask = pd.Series([pd.NA, True, False, False], dtype="boolean")

    result = compare_bucket(df, mask, "outcome")

    assert result["contraction"]["n"] == 1
    assert result["contraction"]["mean"] == 1.0
    assert result["non_contraction"]["n"] == 2
    assert result["non_contraction"]["mean"] == 2.5

def test_compare_bucket_requires_both_groups_for_mean_difference():
    df = pd.DataFrame({"outcome": [1.0, 2.0]})
    mask = pd.Series([True, True], dtype="boolean")

    result = compare_bucket(df, mask, "outcome")

    assert result["mean_difference"] is None
