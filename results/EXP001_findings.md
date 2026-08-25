# Experiment 001 — Deep-Dive Findings (5-Year Dataset)

**Data:** NIFTY 50 daily OHLC, 2021-08-24 → 2026-08-24 (1,241 sessions)
**Question:** Does volatility contraction contain predictive information about subsequent directional movement?

## Executive Summary

The naive "volatility squeeze → breakout" hypothesis is **false** for NIFTY.
Contraction predicts **below-normal forward returns**, not directional breakouts.

However, the effect is **not uniform**: it concentrates overwhelmingly in
**high-volatility backdrops** — a squeeze while realized vol is above its median
is a strong bearish signal (p=2.45e-06, negative mean across every year tested).
A squeeze during low-vol regimes carries no information (p=0.95).

## Signal definitions compared

| Definition | Verdict |
|---|---|
| Realized vol percentile | ❌ Unstable — flips sign by year (2023 positive, 2025/26 negative) |
| ATR/price percentile | ❌ Same instability as realized vol |
| **Bollinger width percentile** | ✅ Robust: survives BH-FDR, threshold sweeps, alternate params |

## BB-width squeeze (≤10th percentile of 20d window) — core stats

- Incidence: ~18% of days (216 days)
- Forward 20d return: **−0.50% mean vs +0.58% normal** (p=5.7e-05)
- Threshold sweep monotone (0.05→0.30 all significant, strongest at tightest)
- Alternate parametrization (30d BB) confirms: p=1.9e-05

## The real signal: Squeeze × High-Vol backdrop

Conditioning on realized vol ≥ median:

- n=104, forward 20d mean **−1.48%** (median −1.07%), P(neg)=61.5%
- vs everything else +0.78%, **p=2.45e-06**
- Directionally consistent in all 6 years; strong in 2022 and 2026

Interpretation: a Bollinger squeeze *during an elevated-volatility regime* marks
the calm before renewed downside — consolidation within a bear phase, not a base
for reversal.

## Regime splits (h=20d)

- Uptrend: squeeze +0.03% vs normal +0.59% (p=0.034)
- Downtrend: squeeze **−1.60%** vs normal +0.57% (p=0.0003)

## Caveats

- Exploratory, single-instrument, single market. No out-of-sample holdout yet.
- Effect sizes are modest per-trade; costs matter.
- Multiple-testing handled via BH-FDR on the 90-test grid, but the combo
  finding was *post-hoc* — treat as Experiment 002 hypothesis to pre-register.

## Recommended next steps

1. Pre-register EXP-002: BB-squeeze × high-RV filter, h=20d, short bias
2. Walk-forward validation with yearly refits
3. Transaction-cost-aware backtest (NIFTY futures slippage model)
