from __future__ import annotations
import pandas as pd


def backtest(df: pd.DataFrame, signal: pd.Series,
             horizon: int = 20, cost_rt: float = 0.0005) -> dict:
    """Entry next open after trigger; exit at close `horizon` sessions later.

    Overlapping triggers each open a position (research mode). Returns per-trade
    table and summary with costs applied.
    """
    opens = df["open"]
    closes = df["close"]
    trades = []
    trigger_idx = list(df.index[signal.fillna(False)])

    for i in trigger_idx:
        pos = df.index.get_loc(i)
        entry_idx = pos + 1
        exit_idx = pos + 1 + horizon
        if exit_idx >= len(df):
            continue  # trade not closed within data — skip (no look-ahead guessing)
        entry = float(opens.iloc[entry_idx])
        exit_ = float(closes.iloc[exit_idx])
        gross = exit_ / entry - 1
        net = gross - cost_rt
        trades.append({
            "trigger_date": str(df.loc[i, "date"].date()),
            "entry_date": str(df["date"].iloc[entry_idx].date()),
            "exit_date": str(df["date"].iloc[exit_idx].date()),
            "entry": entry, "exit": exit_,
            "gross": round(gross, 6), "net": round(net, 6),
        })

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"n_trades": 0, "trades": []}

    wins = tdf[tdf["net"] > 0]
    eq = (1 + tdf["net"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    summary = {
        "n_trades": int(len(tdf)),
        "win_rate": float((tdf["net"] > 0).mean()),
        "avg_net": float(tdf["net"].mean()),
        "median_net": float(tdf["net"].median()),
        "total_return_compound": float(eq.iloc[-1] - 1),
        "max_drawdown": float(dd),
        "cost_per_trade": cost_rt,
        "horizon_days": horizon,
    }
    return {"summary": summary, "trades": tdf.to_dict(orient="records")}
