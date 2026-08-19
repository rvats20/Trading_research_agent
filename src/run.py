from __future__ import annotations
import argparse, json
from pathlib import Path
from .data import load_ohlcv
from .features import add_volatility_features
from .outcomes import add_forward_outcomes
from .evaluate import evaluate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/experiment_001.json")
    args = p.parse_args()

    df, snapshot = load_ohlcv(args.data)
    df = add_volatility_features(df)
    df = add_forward_outcomes(df)

    report = {
        "experiment_id": "EXP-001",
        "question": "Does volatility contraction contain predictive information about subsequent directional movement in NIFTY?",
        "data_snapshot": snapshot.__dict__,
        "method": {
            "contraction_percentiles": [0.10, 0.20, 0.30],
            "horizons_days": [1, 3, 5, 10, 20],
            "volatility_definitions": [
                "ATR / price", "realized volatility", "Bollinger width"
            ],
            "statistical_test": "Welch two-sample t-test",
            "note": "Exploratory research; no strategy optimization or holdout access."
        },
        "results": evaluate(df),
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")

if __name__ == "__main__":
    main()
