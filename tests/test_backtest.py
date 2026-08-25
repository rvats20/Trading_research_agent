from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest


@pytest.fixture
def toy_df():
    """10 days, trigger on day 0. Open next day = 100, close +20d = 110."""
    n = 25
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = [100.0] * n
    df = pd.DataFrame({"date": dates, "open": close, "high": close,
                       "low": close, "close": close})
    sig = pd.Series([False] * n, index=df.index)
    sig.iloc[0] = True
    # make exit-day close 110 (index pos 1+20=21)
    df.loc[df.index[21], "close"] = 110.0
    df.loc[df.index[1], "open"] = 100.0
    return df


def test_exact_trade_pnl(toy_df):
    res = backtest(toy_df, toy_df["signal"] if "signal" in toy_df else _sig(toy_df),
                   horizon=20, cost_rt=0.0)
    assert res["summary"]["n_trades"] == 1
    t = res["trades"][0]
    assert t["entry"] == 100.0 and t["exit"] == 110.0
    assert abs(t["gross"] - 0.10) < 1e-9


def test_cost_applied(toy_df):
    res = backtest(toy_df, _sig(toy_df), horizon=20, cost_rt=0.0005)
    t = res["trades"][0]
    assert abs(t["net"] - (0.10 - 0.0005)) < 1e-9


def test_unclosed_trade_skipped():
    n = 10
    dates = pd.bdate_range("2023-01-02", periods=n)
    df = pd.DataFrame({"date": dates, "open": 100.0, "high": 100.0,
                       "low": 100.0, "close": 100.0})
    sig = pd.Series([True] + [False] * (n - 1), index=df.index)
    res = backtest(df, sig, horizon=20, cost_rt=0.0005)
    assert res["n_trades"] == 0


def _sig(df):
    s = pd.Series([False] * len(df), index=df.index)
    s.iloc[0] = True
    return s
