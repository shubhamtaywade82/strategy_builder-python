# Strategy Builder — Report Alignment Design

**Date:** 2026-05-31
**Branch:** architecure-update
**Source of truth:** `docs/strategy_analysis_report/strategy_analysis_report.md`

## Problem

The SOLUSDT analysis report describes a system that is **more validated and more
honest than the code actually is**. Three concrete gaps:

1. **Overfit ranking.** `_ui_engine/grid_search.py` flags strategies `is_viable`
   and sorts them by **in-sample expectancy** (lines 98, 133). It already computes
   `walk_forward.is_valid` (avg test AUC > 0.52, stability ≥ 0.6) and
   `shuffle_test.is_significant` (p < 0.05) per side — but never uses them to gate.
   Result: `short_threshold_0.7` (test AUC 0.521, p = 0.10) ranks #1 with a 96.2%
   in-sample win rate — the exact "illusion" the report warns about (§2).

2. **Fabricated "reality".** `frontend/src/sections/StrategyPlayer.tsx` is 100%
   hardcoded mock data — `RESEARCH_RULES`, `FOLD_DATA`, and combined stats
   (57.6% WR / 1.45 PF / p = 0.003) are static literals (lines 12–56, 277–290) and
   ignore the `results` prop. The report's "honest reality" 5-condition metrics (§3)
   are these UI placeholders. They were never backtested.

3. **Un-backtested, duplicated rule.** `trading_bot.py` (untracked) implements the
   report's 5-condition rule *live* (signal logging only — no execution, no sizing,
   no risk controls). Its SMC logic is a 4th divergent copy versus `smc_features.py`,
   the frontend, and the report. The rule has no backtest behind it anywhere.

## Goal

Make the code earn the report's claims: rank strategies by out-of-sample
robustness, produce **real** backtested metrics for the 5-condition rule, surface
those real numbers in the UI, add the report's recommended filters (§6.2), and give
the live bot the risk guardrails the report prescribes (§6.3).

## Scope (confirmed)

Full vertical — backend/research + frontend + live bot. All filters including new
OI/funding data feeds and best-effort liquidation. Phased so each phase is
independently verifiable.

## Non-goals

- No retraining schedule/automation (report §9 item 8) — future round.
- No paper/live order **execution** in `trading_bot.py` — it stays a signal+risk
  engine; execution wiring is out of scope.
- No changes to the separate `strategy_builder` package (regime switcher, etc.)
  except the shared `risk_guard` consumed by `trading_bot.py`.

---

## Architecture

```
                          ┌─────────────────────────────┐
  data_fetcher.fetch_*  → │  features (MTF + SMC + new   │
  oi_funding.fetch_*    → │  pctile/regime/1h-BOS cols)  │
                          └──────────────┬──────────────┘
                                         │
                ┌────────────────────────┼───────────────────────┐
                ▼                        ▼                        ▼
        grid_search (model       rule_strategy (5-cond     filters (RVOL, taker,
        confidence) + robustness  long/short) + backtest    OI, funding, liq) +
        gate + score              + walk_forward            confidence_score
                │                        │                        │
                └────────────┬───────────┴────────────┬───────────┘
                             ▼                         ▼
                   ui_research.run_research      risk_guard (shared)
                   → {results, player, ...}            │
                             │                         ▼
                             ▼                    trading_bot.py
                   StrategyPlayer.tsx (real)      (live signals + guard)
```

---

## Components

### New: `_ui_engine/robustness.py`
`score_robustness(wf: dict, shuffle: dict, in_sample_metrics: dict) -> dict`

Returns `{robustness_score: 0-100, is_robust: bool, warnings: [str]}`.

- `robustness_score` = weighted blend:
  - test-AUC component: `clip((avg_test_auc - 0.5) / 0.15, 0, 1)` → 40 pts
  - stability (fraction of folds AUC > 0.5) → 25 pts
  - significance: `1 - min(p_value/0.05, 1)` → 25 pts
  - low degradation (train AUC − test AUC small) → 10 pts
- `is_robust` = `in_sample_metrics.expectancy > 0 AND wf.is_valid AND shuffle.is_significant`.
- `warnings`: append `"in-sample only — fails OOS"` when not robust;
  `"AUC≈random (overfit)"` when `avg_test_auc < 0.53`;
  `"not significant (p≥0.05)"` when `p_value ≥ 0.05`.

### Modified: `_ui_engine/grid_search.py`
- After computing `wf` and `st` per side, call `score_robustness(...)` and attach
  `robustness`, `is_robust`, `warnings` to **each** strategy dict (the wf/shuffle
  are per-side, valid to share across that side's threshold variants).
- `is_viable` becomes `is_robust` (NOT raw in-sample expectancy). Keep the old
  in-sample expectancy as `is_viable_in_sample` for transparency/debugging.
- Sort `strategies` by `robustness_score` desc, then expectancy desc as tiebreak.

### Modified: `_ui_engine/features.py`
Add the columns the 5-condition rule needs (currently missing):
- HTF EMA50/EMA200 trend: in `compute_htf_block`, add
  `{prefix}_ema_regime = sign(ema50 - ema200)` (so `4h_ema_regime` exists).
- Percentile-rank transforms (rolling, leakage-safe, expanding or long rolling
  window so it uses past data only): `ltf_atr_pctile`, `ltf_vol_pctile`.
- Ensure 1H + 15m SMC columns are present in the matrix `run_research` uses. Today
  `ui_research` calls `build_mtf_features` (no SMC). Switch it to
  `smc_features.compute_all_features`, and extend that to also join **1H** SMC
  (for `1h_bos`) the same merge-asof way it joins 15m.

### New: `_ui_engine/rule_strategy.py`
`backtest_rule_strategy(features, data_1m, side, rr_cfg, ...) -> dict`

- Define the 5 named conditions mapping to real columns:
  | # | Condition | Long expr | Short expr |
  |---|---|---|---|
  | 1 | HTF trend | `4h_ema_regime > 0` | `< 0` |
  | 2 | 1H BOS | `1h_smc_bos_recent == 1` | `== -1` |
  | 3 | 15m FVG | `15m_smc_fvg_above_unmitigated == 1` | `fvg_below == 1` |
  | 4 | ATR pctile | `ltf_atr_pctile > 0.60` | same |
  | 5 | Vol pctile | `ltf_vol_pctile > 0.70` | same |
- Build signals where all 5 AND-true → run existing `backtest()` for metrics
  (WR, PF, expectancy, avg win/loss) and `walk_forward()` for OOS AUC + per-fold
  table; derive `sharpe` from per-trade pnl; derive `p_value` from `shuffle_test`.
- Return `{name, side, conditions[], metrics, folds[], robustness, warnings}`.
- This is the single backtest behind both the UI and the live bot's rule.

### Modified: `strategy_research/ui_research.py`
Add `player` to the payload:
```python
return {"symbol": ..., "results": ..., "player": {
    "long": backtest_rule_strategy(..., side="long"),
    "short": backtest_rule_strategy(..., side="short"),
}}
```
Backend tests + API schema (`strategy_api/schemas.py`) updated to allow the new key.

### New: data feeds `_ui_engine/oi_funding.py`
- `fetch_open_interest_hist(symbol, period, days)` → `/futures/data/openInterestHist`
  (note: Binance caps history ~30 days for OI — log + clamp honestly).
- `fetch_funding_rate(symbol, days)` → `/fapi/v1/fundingRate`.
- Leakage-safe merge-asof(backward) onto the 1m grid in `compute_all_features`,
  producing `oi_chg_pct` (rolling % change) and `funding_rate` columns.
- Liquidations: **no clean public historical endpoint** (`allForceOrders`
  deprecated; `forceOrder` is a live websocket stream only). Honest handling:
  research uses a proxy (`liq_proxy` = large-wick + volume-spike flag) and logs the
  caveat; the live bot can subscribe to the `forceOrder` stream (live-only). The
  spec does not claim historical liquidation backtesting.

### New: `_ui_engine/filters.py`
- `rvol(volume, window=20) -> Series` (current vol / rolling mean), filter `> 1.2`.
- taker-imbalance filter reuses `ltf_taker_imb`.
- `oi_expansion` filter `oi_chg_pct > 0.03`; `funding_ok` filter (not extremely
  positive for longs); `liq_clear` (proxy/stream).
- `confidence_score(row) -> 0..100` composite per report §6.2; expose so the rule
  strategy and bot can require `≥ 70`.
- Filters degrade gracefully: a filter whose source column is missing logs once and
  passes-through (does not silently block all trades).

### New: `risk_guard.py` (repo root, importable by `trading_bot.py`)
`RiskGuard` dataclass tracking session state:
- `daily_loss_limit` (default 5% of account) → `halted_today` when breached.
- `max_consecutive_losses` (default 5) → halt.
- `leverage_step_down`: drop to 5x after 3 consecutive losses.
- `position_size(account, entry, stop)` → quarter-Kelly-clamped fixed-fractional
  (reuses `_ui_engine/position_size.py`), capped at 2% risk.
- `register_trade(pnl)` updates streak/daily counters; `can_trade() -> (bool, reason)`.

### Modified: `trading_bot.py`
- Instantiate `RiskGuard`; gate signal emission on `can_trade()`.
- On signal, compute size via `guard.position_size(...)` and log it (no execution).
- Apply the `confidence_score ≥ 70` filter + RVOL/taker/funding filters before
  declaring a signal. Keep the existing 5-condition core (it already matches the
  rule), but import condition logic from one place where practical to kill the
  4th-copy drift.

### Modified: `frontend/src/sections/StrategyPlayer.tsx`
- Read `results.player.long` / `.short` for conditions, metrics, fold table, and
  combined stats. Delete the hardcoded `RESEARCH_RULES`, `FOLD_DATA`, and literal
  stat array.
- Honest empty state: when `results?.player` is absent, render
  "Run research to populate live-validated rule metrics" instead of fake numbers.
- Keep the interactive condition-toggle UX (it's fine) but drive labels/metrics
  from data.

---

## Data flow

1. `run_research` fetches klines (+ OI/funding) → builds MTF+SMC+new-cols features.
2. `grid_search` produces model-confidence strategies, now robustness-gated/scored.
3. `rule_strategy` backtests the 5-condition long/short rule → real metrics+folds.
4. Payload `{results, player}` returned to API → frontend renders real numbers.
5. `trading_bot.py` evaluates the same rule live, filtered + risk-guarded.

## Error handling

- Each new fetcher wraps network calls, returns empty + logs on failure; feature
  build tolerates missing OI/funding columns (filters pass-through with a logged
  caveat — never silently block).
- One RR / one side failing must not kill the run (existing pattern preserved).
- `rule_strategy` returns `{error}` if insufficient signals (< 20 trades) rather
  than fabricating metrics; UI shows the empty state.

## Testing

- `tests/` (pytest) new/updated:
  - `test_robustness.py`: overfit case (AUC 0.52, p 0.10) → `is_robust False`,
    warnings present; strong case → True. Ranking puts robust above non-robust.
  - `test_rule_strategy.py`: synthetic features where all 5 conditions fire →
    signals generated; metrics keys present; insufficient-data → `{error}`.
  - `test_filters.py`: RVOL/confidence-score math; missing-column pass-through.
  - `test_features.py` additions: `ltf_atr_pctile`/`ltf_vol_pctile`/`4h_ema_regime`
    exist and are leakage-safe (value at t uses only ≤ t).
  - `tests/api/test_ui_research_shape.py`: payload includes `player.long/short`.
- Frontend: type-check + smoke that StrategyPlayer renders from a fixture payload
  and shows empty state without `player`.

## Phasing (each independently shippable)

- **P1 — Robustness gate** (WI-1): `robustness.py` + `grid_search.py` + tests.
  Highest value, smallest blast radius.
- **P2 — Feature additions** needed downstream: `features.py` pctile/regime cols,
  1H/15m SMC join, tests.
- **P3 — Real rule backtest** (WI-2 core): `rule_strategy.py` + `ui_research`
  payload + API schema + tests.
- **P4 — Frontend wiring** (WI-3): `StrategyPlayer.tsx` real data + empty state.
- **P5 — Filters + feeds** (WI-2 filters): `oi_funding.py`, `filters.py`,
  confidence score, tests.
- **P6 — Risk guard + bot** (WI-4): `risk_guard.py` + `trading_bot.py` wiring + tests.

## Open risks

- OI history window is short (~30d) on Binance; longer backtests will have NaN OI
  for older bars → filters pass-through there (documented, not hidden).
- Historical liquidation backtesting is not possible with public data; we use a
  proxy in research and live stream in the bot, and say so.
- Switching `ui_research` from `build_mtf_features` to `compute_all_features` adds
  SMC columns to the model-confidence grid search too — may shift its feature
  importances. Acceptable (richer features); covered by existing run tests.
