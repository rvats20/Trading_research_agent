from __future__ import annotations
import hashlib
from dataclasses import dataclass
import pandas as pd

REQUIRED = ["date", "open", "high", "low", "close"]

# Common NSE / niftyindices export column aliases -> canonical names.
_COLUMN_ALIASES = {
    "date": "date",
    "timestamp": "date",
    "trading date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "open price": "open",
    "high price": "high",
    "low price": "low",
    "close price": "close",
    "shares traded": "volume",
    "volume": "volume",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common NSE export headers to canonical lowercase names."""
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        target = _COLUMN_ALIASES.get(key)
        if target:
            renamed[col] = target
        elif key in REQUIRED or key == "volume":
            renamed[col] = key
    return df.rename(columns=renamed)

def _clean_number(series: pd.Series) -> pd.Series:
    """Handle Indian number formatting: '1,234.56', stray spaces, '-' placeholders."""
    if not pd.api.types.is_numeric_dtype(series):
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"": None, "-": None, "nan": None, "NaN": None})
        )
    return pd.to_numeric(series, errors="coerce")

@dataclass(frozen=True)
class DataSnapshot:
    rows: int
    start: str
    end: str

    sha256: str

def load_ohlcv(path: str) -> tuple[pd.DataFrame, DataSnapshot]:
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}. "
            "If this is an NSE export, check that the file starts with its header row."
        )

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")
    for c in ["open", "high", "low", "close"]:
        df[c] = _clean_number(df[c])
        bad = df[c].isna()
        if bad.any():
            raise ValueError(f"Column '{c}' has {int(bad.sum())} unparseable values.")
    if "volume" in df.columns:
        df["volume"] = _clean_number(df["volume"])

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if not (df["high"] >= df[["open", "close"]].max(axis=1)).all():
        raise ValueError("Invalid OHLC: high below open/close.")
    if not (df["low"] <= df[["open", "close"]].min(axis=1)).all():
        raise ValueError("Invalid OHLC: low above open/close.")
    if (df[["open","high","low","close"]] <= 0).any().any():
        raise ValueError("OHLC contains non-positive values.")

    raw = df.to_csv(index=False).encode()
    snap = DataSnapshot(
        rows=len(df),
        start=df["date"].min().isoformat(),
        end=df["date"].max().isoformat(),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    return df, snap
