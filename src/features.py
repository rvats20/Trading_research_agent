from __future__ import annotations
import numpy as np
import pandas as pd

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

def rolling_percentile(s: pd.Series, window: int) -> pd.Series:
    """
    Percentile rank of the current observation inside its trailing window.
    The current observation is included by design; this is a state descriptor,
    not a predictive feature. Use shift(1) if the research question requires
    the state to be defined strictly from prior observations.
    """
    def pct(x):
        last = x[-1]
        return np.nan if np.isnan(last) else float((x <= last).mean())

    return s.rolling(window, min_periods=window).apply(pct, raw=True)

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tr = true_range(out)

    out["atr_14"] = tr.rolling(14, min_periods=14).mean()
    out["atr_pct_price"] = out["atr_14"] / out["close"]

    logret = np.log(out["close"]).diff()
    out["realized_vol_20"] = (
        logret.rolling(20, min_periods=20).std() * np.sqrt(252)
    )

    mid = out["close"].rolling(20, min_periods=20).mean()
    sd = out["close"].rolling(20, min_periods=20).std()
    out["bb_width_20"] = (4 * sd) / mid

    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["range_mean_20"] = out["range_pct"].rolling(
        20, min_periods=20
    ).mean()

    for source in ["atr_pct_price", "realized_vol_20", "bb_width_20"]:
        for window in [20, 60]:
            out[f"{source}_pct_{window}"] = rolling_percentile(
                out[source], window
            )

    return out
