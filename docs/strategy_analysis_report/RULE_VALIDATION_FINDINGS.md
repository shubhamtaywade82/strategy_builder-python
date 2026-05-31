# 5-Condition Rule — Empirical Validation Findings

**Date:** 2026-05-31
**Symbols:** BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT (Binance USD-M perp)
**Window:** 60 days, 1m execution
**Runner:** `run_strategies.py` → `output/strategies.json`, `output/rule_research_*.json`

## TL;DR

The report's 5-condition rule (4H trend + 1H BOS + 15m FVG + ATR>60p + Vol>70p)
has **no deployable edge** on any of the 4 symbols, either side, at any tested
risk:reward, even with **zero fees**. Tested 96 configurations (4 symbols × 2 sides
× 2 RR × 3 fee scenarios). **Zero passed the robustness gate.**
`output/strategies.json` is written with every strategy `deploy: false` — the
correct, safe config for the bots (deploy nothing).

The report's headline "honest reality" numbers (57.6% WR / 1.45 PF / p<0.003) were
**hardcoded mock data** in `frontend/src/sections/StrategyPlayer.tsx`
(`RESEARCH_RULES`, `FOLD_DATA`) — never backtested. This validation disproves them.

## Method

- Leakage-safe signal construction: HTF (4H/1H/15m) conditions merged onto the 1m
  grid via `merge_asof(backward)` keyed on HTF `close_time` (an HTF bar only
  influences 1m bars after it closes). ATR/volume percentiles use trailing-window
  quantiles `.shift(1)` (strictly past). 4H EMA200 warmed with +40d extra history.
- 30-bar entry cooldown to dedupe overlapping signals (→ realistic ~1–3 trades/day).
- Each setup backtested with the existing path-dependent engine, then:
  - **Out-of-time walk-forward** (5 sequential folds) → stability = fraction of
    folds with positive expectancy.
  - **Bootstrap significance** (2000 resamples) → p = P(mean per-trade pnl ≤ 0).
  - **Random-entry baseline** (3000 random bars, same RR/fee) → the null WR/expectancy.
- Robustness gate (all required): expectancy>0 AND stability≥0.6 AND p<0.05 AND
  n≥30 AND beats the random baseline on both WR and expectancy.

## Key findings

### 1. No edge even at zero fees
| Best zero-fee setup | netE/trade | edge vs random | p-value |
|---|---|---|---|
| XRP LONG 2:1 | +0.48% | +0.7% | 0.19 |
| SOL LONG 1.5:1 | +0.05% | +1.8% | 0.48 |
| ETH LONG 1.5:1 | +0.12% | −3.3% | 0.41 |

Nothing reaches p<0.05. The apparent edge is statistical noise.

### 2. The rule is mildly ANTI-predictive
At zero fees the **random baseline beats the rule** on most symbols — e.g. BTC LONG:
rule WR 36.3% vs random **47.9%**. The conditions (high ATR + high volume + recent
FVG + BOS) cluster entries at **breakout spikes that subsequently fade** — i.e. the
rule buys local tops. Timing is worse than random.

### 3. Shorts are the worst
Anti-predictive on every symbol (edge vs baseline −6% to −14% WR). The bearish
mirror logic actively selects bad entries.

### 4. Fees + 10x leverage would bury any real edge anyway
Taker round-trip ≈0.09% × 10x ≈ **0.9% margin drag per trade**. The largest gross
edge observed (XRP long, ~+0.4%) does not survive it. Maker fees (~0.04% round-trip)
halve the drag but do not create an edge that is not there.

## Conclusion

No rule variant is fit for live deployment on the current 60-day window. Decision:
**accept no-edge**, keep `strategies.json` at `deploy: false`, do not trade this rule.

## Leads not pursued (for a future session)

- **Inversion / mean-reversion:** since the rule is anti-predictive, fading the
  breakout (invert the signal side) is the obvious test of whether a real,
  tradeable edge lives on the *other* side of these conditions.
- **Lower frequency / higher TF:** 5m/15m execution with ATR/trailing exits, where
  the fee drag is proportionally smaller.
- **Rule redesign:** a different hypothesis grounded in what genuinely predicts —
  not conditions copied from a report whose validation numbers were fabricated.

## Artifacts

- `run_strategies.py` — the validation runner (reusable).
- `output/strategies.json` — bot config (all `deploy:false`).
- `output/rule_research_20260531_125308.json` — full per-config metrics, folds,
  baselines, fee scenarios.
- `output/run_strategies.log` — console run log.
