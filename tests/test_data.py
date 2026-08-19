import pandas as pd
import pytest

from src.data import load_ohlcv

def write_csv(tmp_path, rows):
    p = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p

def test_loader_sorts_and_deduplicates_dates(tmp_path):
    p = write_csv(tmp_path, [
        {"date":"2020-01-03","open":102,"high":104,"low":101,"close":103},
        {"date":"2020-01-01","open":100,"high":102,"low":99,"close":101},
        {"date":"2020-01-01","open":100,"high":102,"low":99,"close":101},
    ])
    df, snap = load_ohlcv(str(p))
    assert len(df) == 2
    assert df["date"].is_monotonic_increasing
    assert snap.rows == 2

def test_loader_rejects_invalid_ohlc(tmp_path):
    p = write_csv(tmp_path, [
        {"date":"2020-01-01","open":100,"high":90,"low":80,"close":95},
    ])
    with pytest.raises(ValueError):
        load_ohlcv(str(p))

def test_loader_requires_ohlc(tmp_path):
    p = write_csv(tmp_path, [
        {"date":"2020-01-01","open":100,"high":102,"low":99},
    ])
    with pytest.raises(ValueError):
        load_ohlcv(str(p))
