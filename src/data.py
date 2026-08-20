from __future__ import annotations
import hashlib
from dataclasses import dataclass
import pandas as pd

REQUIRED = ["date", "open", "high", "low", "close"]

@dataclass(frozen=True)
class DataSnapshot:
    rows: int
    start: str
    end: str
    
    sha256: str

def load_ohlcv(path: str) -> tuple[pd.DataFrame, DataSnapshot]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.rename(columns={cols[c]: c for c in cols})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

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
