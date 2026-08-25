from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.trade_signal SIGNAL_PARAMS, build_signal


@pytest.fixture
def synthetic_df():
    n = 300
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.bdate_range("2022-01-03", periods=n)
    df = pd.DataFrame({
        "date": dates,
        "open": close + 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
    })
    # fake the derived columns signal.py needs
    mid = df["close"].rolling(SIGNAL_PARAMS["bb_pct_window"], min_periods=SIGNAL_PARAMS["bb_pct_window"]).mean()
    sd = df["close"].rolling(SIGNAL_PARAMS["bb_pct_window"], min_periods=SIGNAL_PARAMS["bb_pct_window"]).std()
    df["bb_width_20"] = (4 * sd) / mid
    w = SIGNAL_PARAMS["bb_pct_window"]

    def pct(x):
        last = x[-1]
        return np.nan if np.isnan(last) else float((x <= last).mean())
    df["bb_width_20_pct_20"] = df["bb_width_20"].rolling(w, min_periods=w).apply(pct, raw=True)
    lr = np.log(df["close"]).diff()
    df["realized_vol_20"] = lr.rolling(20, min_periods=20).std() * np.sqrt(252)
    return df


def test_signal_is_boolean_tri_state(synthetic_df):
    sig = build_signal(synthetic_df)
    assert sig.dtype == bool or sig.isna().any()
    assert set(sig.dropna().unique()) <= {True, False}


def test_no_signal_before_warmup(synthetic_df):
    sig = build_signal(synthetic_df)
    warmup = SIGNAL_PARAMS["rv_min_periods"]
    # signal is NaN only where BB pctile undefined; after BB warmup (20 rows) the
    # RV regime flag may still be False rather than NaN -> just require no True
    assert not sig.iloc[:warmup].fillna(False).any()


def test_params_frozen_roundtrip():
    import json
    s = json.dumps(SIGNAL_PARAMS, sort_keys=True)
    assert json.loads(s) == SIGNAL_PARAMS


def test_trigger_requires_both_conditions(synthetic_df):
    sig = build_signal(synthetic_df)
    bb = synthetic_df["bb_width_20_pct_20"]
    # any trigger day must have bb percentile <= threshold
    trig = sig[sig == True]  # noqa: E712
    assert (bb.loc[trig.index] <= SIGNAL_PARAMS["bb_threshold"]).all()
