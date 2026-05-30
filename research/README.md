# MTF Research Pipeline (Binance USD-M)

Leakage-safe supervised research toward a net +1% price move (10% margin @ 10x),
both directions. NOT an execution engine.

## Modules
- `mtf_research.py`   — kline loader + leakage-safe MTF features + triple-barrier labels
- `barrier_leverage.py` — both-side, 10x leverage-aware labels + liquidation guard
- `binance_meta.py`   — per-symbol funding + mmr (network, with fallback)
- `validation.py`     — purged folds, train-only theta, probes, verdict gates
- `run_research.py`   — orchestrator + CLI

## Run
    cd research
    python run_research.py --symbol SOLUSDT --days 45

Outputs `research/output/<symbol>_<date>.json` and a console PASS/FAIL per side.
A run whose funding/mmr came from the fallback path is marked NOT-VALIDATED.

## Tests
    ./venv/bin/pytest tests/test_validation.py tests/test_binance_meta.py tests/test_run_research.py -v
