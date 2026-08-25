from __future__ import annotations
import pandas as pd

from regime import expanding_median_flag

# Frozen signal parameters — written verbatim into every report.
SIGNAL_PARAMS = {
    "bb_pct_window": 20,
    "bb_threshold": 0.10,
    "rv_min_periods": 120,
}


def build_signal(df: pd.DataFrame) -> pd.Series:
    """BB-squeeze x high-RV trigger.

    True on days where Bollinger width is at/below its 10th percentile of the
    trailing 20-day window AND realized vol is at/above its expanding median.
    False on valid non-trigger days, NaN where the BB percentile is undefined
    or the RV regime cannot yet be called.
    """
    bb = df["bb_width_20_pct_20"]
    squeeze = (bb <= SIGNAL_PARAMS["bb_threshold"]).where(bb.notna())
    high_rv = expanding_median_flag(
        df["realized_vol_20"], min_periods=SIGNAL_PARAMS["rv_min_periods"]
    )
    combo = squeeze == True  # noqa: E712 — tri-state AND: NaN propagates
    combo = combo & high_rv.reindex(df.index, fill_value=False)
    # invalidate rows where squeeze itself was undefined
    return combo.where(bb.notna())
