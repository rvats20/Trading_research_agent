import numpy as np
import pandas as pd

from src.evaluate import evaluate

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
