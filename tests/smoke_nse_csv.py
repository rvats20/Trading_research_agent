#!/usr/bin/env python3
"""Smoke test: NSE-style CSV (aliased columns, Indian numbers) loads cleanly."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from data import load_ohlcv

csv = (
    "Date,Open Price,High Price,Low Price,Close Price,Shares Traded\n"
    '01-Jan-2024,"21,500.20","21,700.55","21,400.10","21,625.70","123,456,789"\n'
    '02-Jan-2024,"21,630.00","21,850.40","21,600.05","21,842.15","98,765,432"\n'
)
f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
f.write(csv)
f.close()

df, snap = load_ohlcv(f.name)
print(f"rows={snap.rows}")
print(df[["date", "close", "volume"]].to_string(index=False))
os.unlink(f.name)
print("NSE-format load: OK")
