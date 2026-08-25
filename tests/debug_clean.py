#!/usr/bin/env python3
"""Debug: why does _clean_number fail in load_ohlcv but work in isolation?"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd
from data import _normalize_columns, _clean_number

csv = (
    "Date,Open Price,High Price,Low Price,Close Price,Shares Traded\n"
    '01-Jan-2024,"21,500.20","21,700.55","21,400.10","21,625.70","123,456,789"\n'
    '02-Jan-2024,"21,630.00","21,850.40","21,600.05","21,842.15","98,765,432"\n'
)
f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
f.write(csv)
f.close()

df = pd.read_csv(f.name)
print("dtypes after read_csv:")
print(df.dtypes)
print(df["Open Price"].tolist())
n = _normalize_columns(df)
print(n.columns.tolist())
print(repr(n["open"].iloc[0]), type(n["open"].iloc[0]))
cleaned = _clean_number(n["open"])
print("cleaned:", cleaned.tolist())
os.unlink(f.name)
