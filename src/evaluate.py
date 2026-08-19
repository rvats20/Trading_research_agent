from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def _summary(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else None,
        "win_rate": float((x > 0).mean()),
        "q10": float(x.quantile(.10)),
        "q90": float(x.quantile(.90)),
    }

def compare_bucket(df: pd.DataFrame, mask: pd.Series, outcome: str) -> dict:
    a = df.loc[mask, outcome].dropna()
    b = df.loc[~mask, outcome].dropna()
    result = {
        "contraction": _summary(a),
        "non_contraction": _summary(b),
    }
    if len(a) >= 8 and len(b) >= 8:
        t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        result["welch_t"] = float(t.statistic)
        result["p_value"] = float(t.pvalue)
    else:
        result["welch_t"] = None
        result["p_value"] = None
    if len(a):
        result["mean_difference"] = float(a.mean() - b.mean())
    return result

def evaluate(df: pd.DataFrame) -> dict:
    tests = []
    definitions = [
        "atr_pct_price_pct_20",
        "atr_pct_price_pct_60",
        "realized_vol_20_pct_20",
        "realized_vol_20_pct_60",
        "bb_width_20_pct_20",
        "bb_width_20_pct_60",
    ]
    for feature in definitions:
        for threshold in [0.10, 0.20, 0.30]:
            mask = df[feature] <= threshold
            for h in [1, 3, 5, 10, 20]:
                outcome = f"fwd_return_{h}d"
                r = compare_bucket(df, mask, outcome)
                tests.append({
                    "feature": feature,
                    "threshold": threshold,
                    "horizon_days": h,
                    "outcome": outcome,
                    **r
                })
    return {"tests": tests}
