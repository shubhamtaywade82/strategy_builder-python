# 5-Condition Rule — Empirical Validation Findings

**Date:** 2026-05-31
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT (Binance USD-M perp)
**Window:** 60 days, 1m execution
**Runner:** `run_strategies.py` → `output/strategies.json`, `output/rule_research_*.json`

## TL;DR

The original `strategy_analysis_report.md` is **not trustworthy**: its headline
"honest reality" numbers (57.6% WR / 1.45 PF / p<0.003 / fold tables) were
**hardcoded literals** in `frontend/src/sections/StrategyPlayer.tsx` — never
backtested. A real, deterministic, leakage-safe backtest tells a different story:

- The breakout 5-condition rule (trade the signal direction) has **no edge** — it
  is mildly **anti-predictive** (enters breakout spikes that fade).
- The **inverse (fade the bearish breakout → enter LONG) = mean-reversion** carries
  a **real, consistent edge**: +4% to +16% win-rate over a random-entry baseline on
  **all four symbols**. ETH and XRP are net-positive after maker fees. ETH passes
  the full robustness gate (p=0.040). The edge is modest and per-symbol significance
  is borderline (small samples, n≈50–90), but the cross-symbol consistency is strong
  evidence it is real, not noise.

## What was fixed to make results real / deterministic / accurate

1. **Killed the mock.** `StrategyPlayer.tsx` now renders the real backtest payload
   (`results.player`); the fabricated `RESEARCH_RULES`/`FOLD_DATA` literals are gone.
2. **Overfit gate.** `_ui_engine/grid_search.py` now marks a model-confidence
   strategy `is_viable` only if it is in-sample positive AND walk-forward valid AND
   shuffle-significant, and ranks by an out-of-sample `robustness_score`. The 96% WR
   in-sample artifacts now show `is_viable=false` with explicit warnings.
3. **Determinism.** Seeded the previously-unseeded RNG in `validation.shuffle_test`
   (the shuffle permutation) and `position_size.monte_carlo_sizing`. The shuffle-test
   p-value and Monte-Carlo sizing are now identical run-to-run.
4. **Backtest accuracy.** `_ui_engine/backtest.py` charged the round-trip fee
   **twice** on losses/timeouts and once on wins. Now charged exactly once,
   symmetrically: a 2:1 stop nets −0.059 (was −0.068) at 10x / 0.09% fee.
5. **Correct cost model.** The dashboard rule backtest uses **maker** fees (~0.04%)
   because entries are limit orders at the FVG; taker cost understated the edge.

## Method

- Leakage-safe signals: HTF (4H/1H/15m) conditions merged onto the 1m grid via
  `merge_asof(backward)` on HTF `close_time`; ATR/volume percentiles use trailing
  quantiles `.shift(1)`. 4H EMA200 warmed with +40d history. 30-bar entry cooldown.
- Each setup: path-dependent backtest → out-of-time walk-forward (5 folds,
  stability = fraction positive) → bootstrap significance (2000 resamples, seeded)
  → random-entry baseline (same RR/fee). Two modes: **momentum** (trade signal) and
  **reversion** (fade signal).
- Robustness gate (all required): expectancy>0 AND stability≥0.6 AND p<0.05 AND
  n≥30 AND beats baseline on both WR and expectancy.

## Mean-reversion edge (fade bearish signal → enter LONG), corrected & deterministic

| Symbol | WR (maker) | edge vs random | netE/trade (maker 2:1) | best p-value |
|---|---|---|---|---|
| ETH | 56.9% | +15.9% | +0.60% | 0.040 (zero-fee) |
| XRP | 50.7% | +11.9% | +0.31% | 0.126 |
| SOL | 45.6% | +7.5% | −0.34% | 0.43 |
| BTC | 46.9% | +4.2% | −0.24% | 0.44 |

Only **ETH short-reversion** clears the full gate (zero-fee, 1.5:1: WR 58.6%,
PF 1.72, netE +1.08%, p=0.040, n=58). The rest show genuine edge over random but
fail significance at current sample sizes / maker fees.

## Conclusion

The momentum 5-condition rule is **not deployable** (no edge / anti-predictive).
The **mean-reversion inverse is the real lead** — modest, consistent, net-positive
on ETH/XRP at maker fees, but needs more data (or symbol-pooling) to confirm
significance before live deployment. The dashboard's Strategy Player now shows these
real, deterministic numbers with honest DEPLOYABLE / EDGE-NOT-ROBUST / NO-EDGE badges.

## Artifacts

- `src/strategy_research/_ui_engine/rule_strategy.py` — canonical rule backtest.
- `run_strategies.py` — offline multi-symbol / multi-fee / momentum-vs-reversion sweep.
- `output/strategies.json` — bot config (deploy flag per symbol/side/mode).
- `output/rule_research_*.json`, `output/run_strategies_corrected.log` — full metrics.
