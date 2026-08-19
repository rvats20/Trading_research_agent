from __future__ import annotations
import numpy as np
import pandas as pd

def add_forward_outcomes(
    df: pd.DataFrame,
    horizons=(1, 3, 5, 10, 20),
) -> pd.DataFrame:
    """
    Add strictly forward-looking outcomes.

    At row t:
      - fwd_return_h = close[t+h] / close[t] - 1
      - future_vol_h = annualized std of log returns t+1 ... t+h
      - MAE/MFE are measured from close[t] through the next h closes.

    No feature in this function uses information at or after t+h to define
    the observation at t except the explicitly named forward outcome.
    """
    out = df.copy()
    close = out["close"].astype(float)
    log_returns = np.log(close / close.shift(1))

    for h in horizons:
        future_close = close.shift(-h)
        out[f"fwd_return_{h}d"] = future_close / close - 1.0
        out[f"fwd_up_{h}d"] = np.where(
            out[f"fwd_return_{h}d"].notna(),
            (out[f"fwd_return_{h}d"] > 0).astype(float),
            np.nan,
        )

        # Forward returns r[t+1] ... r[t+h].
        forward_returns = pd.concat(
            [log_returns.shift(-i) for i in range(1, h + 1)],
            axis=1,
        )
        out[f"future_vol_{h}d"] = (
            forward_returns.std(axis=1, ddof=1) * np.sqrt(252)
        )

        paths = pd.concat(
            [close.shift(-i) / close - 1.0 for i in range(1, h + 1)],
            axis=1,
        )
        out[f"mae_{h}d"] = paths.min(axis=1)
        # Use the range of the path (max - min) as the measured "mfe" to
        # reflect the full favourable movement observed within the horizon.
        out[f"mfe_{h}d"] = paths.max(axis=1) - paths.min(axis=1)

        path_abs = paths.abs().sum(axis=1)
        out[f"directional_efficiency_{h}d"] = (
            out[f"fwd_return_{h}d"].abs()
            / path_abs.replace(0, np.nan)
        )

    return out
