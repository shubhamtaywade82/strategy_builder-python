# MTF Research Validation Harness — Design

**Date:** 2026-05-31
**Status:** Approved (brainstorming)
**Scope:** Complete the supervised MTF research pipeline in `research/` so its
metrics honor the pre-registered verification contract (advisor §4). No new
strategy invention — discipline the existing one.

## Problem

`research/train_model.py` runs but violates 8 of the advisor's own §4 contract
points and contains a fold-boundary bug. Its fold-by-fold expectancy numbers are
therefore not trustworthy. The two core modules
(`mtf_research.py`, `barrier_leverage.py`) are leakage-safe and unit-tested —
the gap is entirely in the validation/orchestration layer.

### Defects in current `train_model.py`

| # | Contract (§4) | Current state |
|---|---|---|
| 1 | Shuffle test (permuted labels → E collapses to ≈ −cost) | missing |
| 2 | Leakage probe (shift features +1 bar → perf drops) | missing |
| 3 | Cost ladder {0.0009, 0.0012, 0.0015} → breakeven | missing |
| 4 | Per-symbol funding fetch (`fundingInfo` + `fundingRate`) | hardcoded 0.0001 / 8h |
| 5 | mmr from `leverageBracket`/`exchangeInfo` | hardcoded 0.005 |
| 6 | Fold-variance gate (reject if one fold carries) | no aggregation |
| 7 | Liquidation-never-fires assertion | missing |
| 8 | Win-rate sanity (~10–35%) | not printed |

**Bug:** `walk_forward_folds` yields `range(0, tr_end - embargo)`. For fold 1,
`tr_end = n // (n_folds+1)`; if that is `< embargo` the train range is empty or
negative. Embargo purge is fragile.

**Rule extraction flaw:** the surrogate `DecisionTree` is fit on the full dataset
in-sample, then its rule is reported with in-sample expectancy — no OOS check.

## Decisions (approved)

1. **θ chosen on train only** — per fold, pick the probability threshold that
   maximizes E_margin on *train* rows, apply it unchanged to *test*. Never tune θ
   on test (prevents threshold-snooping). Report θ per fold.
2. **Two binary models** — `long_label` (win/not) and `short_label` (win/not),
   each with balanced class weighting. Replaces the brittle 3-class `best_side`
   target, which is ~70%+ class-0 and biases a 3-class model toward no-trade.
3. **Hard verdict gates** — emit PASS/FAIL per side; PASS requires all gates.
4. **Fallback ⇒ NOT-VALIDATED** — if funding/mmr came from the network-blocked
   fallback path, the verdict is forced to NOT-VALIDATED regardless of metrics.

## Architecture

Three new files in `research/`. Core modules untouched.

```
binance_meta.py   per-symbol economics (funding + mmr), disk cache + fallback   [network]
validation.py     purged folds, train-θ policy, probes, verdict gates           [pure, no network]
run_research.py   orchestrate one symbol end-to-end → report + JSON + verdict    [glue]
```

### 1. `binance_meta.py`

Removes hardcoded funding/mmr (contract #4, #5).

- `fetch_funding_profile(symbol) -> FundingProfile`
  - fields: `interval_h`, `avg_rate`, `p90_rate`, `anchor_hour`, `source`
  - sources: `GET /fapi/v1/fundingInfo` (interval, cap),
    `GET /fapi/v1/fundingRate` (realized history → avg + 90th pct)
- `fetch_mmr(symbol, notional) -> (mmr, source)`
  - `GET /fapi/v1/leverageBracket` for the notional tier
- **Fallback:** on network failure (geo-block etc.) log a loud WARNING, return
  documented defaults, and stamp `source="fallback"`. A fallback run is later
  forced to NOT-VALIDATED by the verdict (decision #4) so it cannot masquerade
  as a validated result.
- **Cache:** write/read `research/cache/<symbol>_meta.json` to avoid refetch.

### 2. `validation.py`

Pure functions, no network, fully unit-testable.

- `purged_folds(n, n_folds=5, embargo=120, min_train=500) -> list[(train_idx, test_idx)]`
  - anchored expanding train window; embargo gap on **both** sides of each test
    fold (train end purged of `embargo` bars; test starts `embargo` after train
    end). Folds with `len(train) < min_train` are skipped, not yielded empty.
  - invariant: train and test index sets never overlap; no negative ranges.
- `pick_threshold_on_train(probs_train, pnl_train) -> theta`
  - sweep θ over a grid; return θ maximizing mean E_margin on train rows
    (requires a minimum train trade count to avoid degenerate θ).
- `evaluate_side(ds, side, folds, clf_factory) -> SideReport`
  - per fold: fit binary model on train, pick θ on train, apply to test, record
    `{n_trades, E_margin, win_rate, theta}`. Uses the side's
    `{long|short}_margin_pnl` column for expectancy.
- Probes:
  - `shuffle_test(...)` — permute labels, expect E_margin ≈ −round_trip_cost×L.
  - `leakage_probe(...)` — shift all features +1 bar, expect E_margin to drop
    materially vs. baseline.
  - `cost_ladder(rebuild_labels_fn, costs) -> {cost: E_margin}` — report the
    breakeven cost where E_margin crosses zero.
  - `liquidation_check(labels) -> int` — count catastrophic-branch rows; must be 0.
- `verdict(side_report, probes) -> Verdict{passed: bool, reasons: list[str]}`
  - gates (all required for PASS):
    - `E_margin > 0` on **every** fold
    - `std(fold E_margin) / |mean| <= cv_max` (no single-fold carry)
    - mean `win_rate ∈ [0.10, 0.35]`
    - shuffle E_margin collapsed (≈ −cost×L within tolerance)
    - leakage probe E_margin dropped below a fraction of baseline
    - `liquidation_check == 0`
    - meta `source != "fallback"` (else forced NOT-VALIDATED)

### 3. `run_research.py`

Orchestrator + CLI.

- Flow:
  1. `fetch_funding_profile` + `fetch_mmr` → build `LevBarrierConfig` with real values
  2. load 5 TFs via `BinanceUMKlineLoader` (1m/15m/1h/4h/1d)
  3. `build_mtf_features` → `triple_barrier_both_sides` → join on `entry_idx` → dropna
  4. `purged_folds` → `evaluate_side` for long & short → run all probes
  5. extract human rule via surrogate `DecisionTree` fit **on train folds only**,
     report with its OOS E_margin
  6. emit:
     - console: PASS/FAIL per side with reasons + per-fold table + cost-ladder
       breakeven + win-rate
     - `research/output/<symbol>_<YYYY-MM-DD>.json`: folds, probes, verdict,
       meta source, config
- CLI: `python run_research.py --symbol SOLUSDT --days 45`; loop over SOL/BTC/ETH.

## Data flow

```
binance_meta ─┐
              ├─→ LevBarrierConfig ─→ triple_barrier_both_sides ─┐
klines(5 TF) ─┴─→ build_mtf_features ───────────────────────────┴─→ join → dropna → dataset
dataset → purged_folds → per-side eval (train-θ) → probes(shuffle/leak/cost) → verdict → report (console + JSON)
```

## Error handling

- Network: retries already in loader; `binance_meta` adds explicit fallback +
  `source` stamp.
- Insufficient data for a horizon/fold: skip fold, log; if no folds survive,
  verdict = NOT-VALIDATED with reason "insufficient data".
- `LevBarrierConfig.__post_init__` already rejects stop-too-near-liquidation; let
  it raise.

## Testing

`tests/test_validation.py` (pure, no network):
- fold boundaries never overlap, never negative, respect `min_train`
- train-θ selection never reads test rows
- shuffle test collapses E on synthetic labeled data
- leakage probe drops E on a planted leaky feature
- cost-ladder is monotonic non-increasing in cost
- verdict returns FAIL when any single gate fails (one test per gate)

`tests/test_binance_meta.py`:
- mocked `fundingInfo`/`fundingRate`/`leverageBracket` responses parse correctly
- network failure path returns `source="fallback"` with documented defaults

## Out of scope (YAGNI)

- Execution engine / live trading (separate Ruby layer per advisor).
- The old data-snooping track (`researcher.py`, `researcher_robust.py`,
  `build_mtf_edge.py`, `build_regime_switcher.py`) — left untouched here; a
  separate cleanup decision.
- Hyperparameter search on the classifier — fixed sane GB/LogReg config.
- Multi-symbol portfolio aggregation — one symbol at a time.
