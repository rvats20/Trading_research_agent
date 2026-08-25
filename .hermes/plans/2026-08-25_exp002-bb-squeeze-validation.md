# EXP-002: BB-Squeeze × High-RV Bearish Signal — Pre-Registered Validation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Rigorously validate the single surviving signal from EXP-001 (Bollinger-width squeeze during high realized-vol regime predicts negative 20d forward returns) as a pre-registered, walk-forward, cost-aware hypothesis test — before it's ever traded.

**Architecture:** Pure extension of the existing research framework. New module `exp002/` with signal builder, walk-forward engine, cost-aware backtest, and report generator. Reuses `src/data.py`, `src/features.py`. No strategy execution, no broker. Everything deterministic (fixed seeds, frozen data snapshots).

**Tech Stack:** Python 3.12, pandas 3.x, numpy, scipy (existing venv).

---

## Current Context / Assumptions

- Data: `data/nifty_5y.csv` (1,241 sessions, 2021-08-24 → 2026-08-24), sha256 recorded in EXP-001 report
- Signal (from EXP-001, post-hoc — this experiment is its out-of-sample trial):
  - **Trigger:** `bb_width_20_pct_20 <= 0.10` AND `realized_vol_20 >= median(realized_vol_20)`
  - **Claim:** forward 20d return mean ≈ −1.5%, P(neg) ≈ 61%, n≈104 occurrences in-sample
- Known caveats to address head-on:
  - Signal was discovered on this same dataset → must use walk-forward, not full-sample stats
  - Realized-vol median split uses full-sample median → must become *rolling/expanding* to avoid look-ahead
  - No transaction costs modeled yet
- Repo state: branch `fix-experiment-001-methodology` has EXP-001 findings; merge to main first

---

## Pre-Registration (write BEFORE looking at any new results)

**H1 (primary):** On each walk-forward fold, days meeting the trigger condition have mean forward-20d return significantly below non-trigger days (Welch t-test, one-sided α=0.05).

**H2 (secondary):** P(fwd 20d < 0) on trigger days ≥ 58%.

**H3 (robustness):** Effect direction (negative) consistent in ≥4 of 5 walk-forward folds.

**Falsification:** If H1 fails on the first split of holdout data → signal declared not-ready; no further tuning allowed within EXP-002 (prevents p-hacking loop).

**Decision rule:** Proceed to paper-trading only if H1 AND (H2 OR H3) pass.

---

## Task 1: Merge EXP-001 branch to main + create exp002 branch

**Objective:** Clean baseline for new work.

**Steps:**
1. `git checkout main && git merge fix-experiment-001-methodology`
2. Push main
3. `git checkout -b exp002-validation`

## Task 2: Fix look-ahead bias — rolling median for RV regime

**Objective:** The high-RV condition must use only information available at time t.

**Files:** Create `src/regime.py`

**Implementation:**
```python
def expanding_median_flag(rv: pd.Series, min_periods: int = 120) -> pd.Series:
    """True if rv[t] >= expanding median of rv[0..t-1] (excludes current value).
    Uses min_periods so early days are flagged NaN (invalid)."""
    past_med = rv.shift(1).expanding(min_periods=min_periods).median()
    return (rv >= past_med).where(past_med.notna())
```
Note: EXP-001 used full-sample median — that was look-ahead. EXP-002 corrects it.
This may weaken the signal slightly; that's honest science.

**Tests:** `tests/test_regime.py` — synthetic series, verify no current-value leakage, NaN before min_periods.

## Task 3: Signal builder module

**Objective:** Deterministic trigger construction.

**Files:** Create `src/signal.py`

**Implementation:**
```python
def build_signal(df: pd.DataFrame) -> pd.Series:
    """BB-squeeze x high-RV trigger. Returns True/False/NaN per day."""
    squeeze = (df["bb_width_20_pct_20"] <= 0.10).where(df["bb_width_20_pct_20"].notna())
    high_rv = expanding_median_flag(df["realized_vol_20"])
    return squeeze & high_rv.reindex(df.index, fill_value=False).where(df["bb_width_20_pct_20"].notna())
```
Also expose `SIGNAL_PARAMS = {"bb_pct_window": 20, "bb_threshold": 0.10, "rv_min_periods": 120}` as a frozen dict written into every output report.

**Tests:** mask counts on synthetic data; params serialization round-trip.

## Task 4: Walk-forward engine

**Objective:** Year-by-year folds: train nothing (signal is rule-based) but evaluate per-fold, simulating "would we have believed this at end of each year?"

**Files:** Create `src/walkforward.py`

**Design:**
- Folds by calendar year: evaluate trigger performance using ONLY data up to that point for the RV median expansion (already handled in Task 2)
- Per-fold output: n_trigger, mean fwd 20d, P(neg), Welch p (one-sided)
- Aggregate: sign consistency count, pooled mean, BH across folds is N/A (one primary test)

**Tests:** 3-year synthetic set → verify fold boundaries, no future data in any fold stat.

## Task 5: Cost-aware backtest

**Objective:** Translate statistical edge into net P&L estimate.

**Files:** Create `src/backtest.py`

**Design:**
- Entry: next open after trigger day; Exit: close 20 trading days later (or end of data)
- Overlapping signals allowed but tracked (max concurrent positions parameter, default unlimited for research)
- Costs: 0.05% round-trip (NIFTY futures est from EXP-001 README) — configurable
- Output: per-trade table + summary (n, win rate, avg net return, cum P&L in index points, max drawdown of the equity curve)

**Tests:** known toy scenario → exact expected P&L; cost application verified.

## Task 6: Run validation + generate report

**Objective:** Produce the verdict.

**Steps:**
1. Run walk-forward over nifty_5y.csv
2. Generate `results/experiment_002.json` + `results/EXP002_report.md`:
   - Pre-registration block (copied verbatim from this plan)
   - Fold table
   - Backtest summary
   - Verdict per decision rule
3. Commit everything on `exp002-validation`

**Verification:** JSON parses; report contains pre-registration text identical to plan.

## Task 7: PR + review + merge

1. Push branch, open PR
2. Review diff (check no look-ahead: grep full-sample medians, `.median()` on whole column)
3. Merge after tests green

---

## Tests / Validation Summary

- Unit tests per module (synthetic data, no network)
- Full suite: `pytest -q` green
- Look-ahead audit: search for `.median()` / `.mean()` computed on non-shifted full columns
- Reproducibility: rerun produces byte-identical JSON (fixed sort, fixed seeds)

## Risks / Tradeoffs

- Rolling RV median may demote many early-sample days to invalid → fewer triggers (~60-70 vs 104). Acceptable; honesty > sample size.
- Walk-forward folds of ~250 sessions give modest power per fold — that's why H3 requires direction-consistency rather than per-fold significance.
- If H1 fails: stop. Do NOT tune thresholds and retry (that would invalidate the pre-registration). Any variant becomes EXP-003 with fresh data.

## Open Questions

- Position sizing / concurrency caps are out of scope until statistical validation passes.
