# Trading Research Experiment 001

Greenfield research slice for testing:

> Does volatility contraction in NIFTY contain predictive information about subsequent directional movement?

This implementation is deliberately research-first. It does not optimize a trading strategy or connect to a broker.

This is a financial research framework for testing whether volatility contraction in the NIFTY index contains predictive signal about future directional movement. It's a pure research tool (not a trading bot) that validates hypotheses using statistical testing.


## Inputs

A CSV containing at minimum:
- `date`
- `open`
- `high`
- `low`
- `close`

Optional:
- `volume`

## Run

```bash
pip install -r requirements.txt
python -m src.run --data /path/to/nifty.csv --output results/experiment_001.json
```

The runner:
1. validates and sorts the dataset;
2. computes multiple volatility-contraction definitions;
3. computes forward return, direction, volatility expansion, and directional-efficiency outcomes;
4. evaluates contraction buckets against non-contraction observations;
5. writes a reproducible research report.

Holdout data is intentionally not supported by the runner. Keep it outside the research dataset until the research process is frozen.

## Tests

Run the unit suite before using market data:

```bash
pytest -q
```

Expected result: all tests pass.

The tests use synthetic data only and specifically check temporal alignment of forward outcomes.
