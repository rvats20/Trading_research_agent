from __future__ import annotations
import pandas as pd

from regime import expanding_median_flag


def walk_forward_folds(df: pd.DataFrame) -> list[dict]:
    """Year-by-year folds. Signal uses only past data (expanding median handles
    this internally), so folds are evaluation windows, not training windows."""
    folds = []
    years = df["date"].dt.year.dropna().unique()
    for y in sorted(years):
        sub = df[df["date"].dt.year == y]
        folds.append({
            "year": int(y),
            "index": sub.index,
            "start": sub["date"].min().strftime("%Y-%m-%d"),
            "end": sub["date"].max().strftime("%Y-%m-%d"),
        })
    return folds


def evaluate_fold(df: pd.DataFrame, signal: pd.Series, outcome: str = "fwd_return_20d") -> dict:
    a = df.loc[signal.fillna(False), outcome].dropna()
    b = df.loc[signal.notna() & ~signal.fillna(False), outcome].dropna()
    res = {
        "n_trigger": int(len(a)),
        "mean_trigger": float(a.mean()) if len(a) else None,
        "p_neg": float((a < 0).mean()) if len(a) else None,
        "n_rest": int(len(b)),
        "mean_rest": float(b.mean()) if len(b) else None,
    }
    if len(a) >= 8 and len(b) >= 8:
        tt = st.ttest_ind(a, b, equal_var=False, alternative="less")
        res["welch_p_one_sided"] = float(tt.pvalue)
    else:
        res["welch_p_one_sided"] = None
    return res


from scipy import stats as st  # noqa: E402
