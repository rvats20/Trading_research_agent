#!/usr/bin/env python3
"""EXP-002: Pre-registered walk-forward validation of BB-squeeze x high-RV signal."""
import sys, json
sys.path.insert(0, "/mnt/c/Users/Rahul/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/Trading_research_agent/src")
from data import load_ohlcv
from features import add_volatility_features
from outcomes import add_forward_outcomes
from trade_signal import build_signal, SIGNAL_PARAMS
from walkforward import walk_forward_folds, evaluate_fold
from backtest import backtest

DATA = "/mnt/c/Users/Rahul/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/Trading_research_agent/data/nifty_5y.csv"

df, snap = load_ohlcv(DATA)
df = add_volatility_features(df)
df = add_forward_outcomes(df)
signal = build_signal(df)

# --- Walk-forward folds ---
folds = []
for f in walk_forward_folds(df):
    sub_idx = f["index"]
    res = evaluate_fold(df.loc[sub_idx], signal.loc[sub_idx])
    folds.append({"year": f["year"], **res})

print("FOLD RESULTS")
print("-" * 70)
h1_pass_folds = 0
direction_consistent = 0
for f in folds:
    p = f.get("welch_p_one_sided")
    p_str = f"{p:.4f}" if p is not None else "n/a"
    h1_fold = p is not None and p < 0.05 and (f["mean_trigger"] or 0) < 0
    if h1_fold:
        h1_pass_folds += 1
    if f["mean_trigger"] is not None:
        direction_consistent += 1 if f["mean_trigger"] < 0 else 0
    print(f"{f['year']}: n={f['n_trigger']:3d} mean={f['mean_trigger'] if f['mean_trigger'] is not None else float('nan'):+.4f} "
          f"P(neg)={f['p_neg'] if f['p_neg'] is not None else float('nan'):.2f} "
          f"rest={f['mean_rest']:+.4f} one-sided p={p_str}")

# --- Pooled stats ---
sig_valid = signal.notna()
trig = signal.fillna(False).astype(bool)
a = df.loc[sig_valid & trig, "fwd_return_20d"].dropna()
b = df.loc[sig_valid & ~trig, "fwd_return_20d"].dropna()
from scipy import stats as st
tt = st.ttest_ind(a, b, equal_var=False, alternative="less")
pooled_p = float(tt.pvalue)
print(f"\nPOOLED: n={len(a)} mean {a.mean():+.4f} vs {b.mean():+.4f} | one-sided p={pooled_p:.4e}")
print(f"H2: P(neg)={(a<0).mean():.2%} (need >=58%)")

# --- Backtest ---
bt = backtest(df, signal, horizon=20, cost_rt=0.0005)
s = bt["summary"]
print(f"\nBACKTEST: {s['n_trades']} trades | win {s['win_rate']:.1%} | avg net {s['avg_net']:+.4f} "
      f"| cum {s['total_return_compound']:+.2%} | maxDD {s['max_drawdown']:.1%}")

# --- Verdict per pre-registration ---
H1 = pooled_p < 0.05 and a.mean() < 0
H2 = (a < 0).mean() >= 0.58
H3 = direction_consistent >= 4
verdict = "PROCEED to paper trading" if (H1 and (H2 or H3)) else "SIGNAL NOT VALIDATED"
print(f"\nVERDICT: H1={'PASS' if H1 else 'FAIL'} H2={'PASS' if H2 else 'FAIL'} "
      f"(direction consistent {direction_consistent}/5 folds; need>=4 for H3) H3={'PASS' if H3 else 'FAIL'}")
print(f"=> {verdict}")

out = {
    "experiment_id": "EXP-002",
    "preregistration": "See .hermes/plans/2026-08-25_exp002-bb-squeeze-validation.md",
    "signal_params": SIGNAL_PARAMS,
    "data_snapshot": snap.__dict__,
    "folds": folds,
    "pooled": {"n_trigger": len(a), "mean_trigger": a.mean(), "one_sided_p": pooled_p},
    "backtest_summary": s,
    "verdict": {"H1": H1, "H2": H2, "H3": H3, "decision": verdict},
}
path = "/mnt/c/Users/Rahul/Documents/Codex/2026-08-19/referenced-chatgpt-conversation-this-is-an/Trading_research_agent/results/experiment_002.json"
json.dump(out, open(path, "w"), indent=2, default=str)
print(f"written: {path}")
