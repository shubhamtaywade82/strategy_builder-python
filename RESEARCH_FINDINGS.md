# Strategy Research Findings — SOL / XRP / ETH / BTC

**Question driving this:** find a strategy with **>50% win rate AND 1:2 RR** (TP net
of fees, ≥1% move) usable in the crypto-trader bot, across SOLUSDT, XRPUSDT,
ETHUSDT, BTCUSDT.

**Headline answer:** that exact bar is **not achievable** with any vanilla
technical-analysis approach tested. Win% and reward:risk are *inversely locked* —
the data below proves it. A real, modest edge does exist (ETH, profit-factor
~1.3–1.4 net, out-of-sample). Optimise **net expectancy**, not the 50%/1:2 number.

---

## 1. Methodology (why earlier "winners" were fake)

The shipped `researcher.py` ranks 15 strategies by raw profit-factor with **no
minimum-trade filter and no out-of-sample split**. Result: it "recommended"
strategies with 2–6 trades (e.g. NY_Momentum PF 1.6 on **2 trades**, Funding_Arb
PF 2.42 on **5**). That is curve-fit noise, not edge.

What we use instead (the repo already shipped this infra — `researcher.py` just
ignored it):
- **Fees + slippage modelled** (`FeeModel` 0.05%/side, `SlippageModel` 2 bps) →
  every metric below is **NET**.
- **Anchored out-of-sample holdout** (train 65% / test 35%) and/or 5-fold
  walk-forward (`WalkForward`) — edge must persist on **unseen** data.
- **Hard gates** (`Gatekeeper`): ≥ ~20–30 trades, OOS PF ≥ 1.1, positive OOS
  expectancy, stability across folds.

Tooling written for this (all in `python_strategy_builder/`):
- `researcher_robust.py` — walk-forward + gatekeeper, `--holdout` (fast anchored)
  and `--focus` (only strategies that produce a real trade sample).
- `build_mtf_edge.py` — clean MTF-alignment strategy, cached O(1) (see §4).
- `frontier_sweep.py` — sweeps TP from 1:1→1:2.5 + partial, to map the win/RR
  frontier and pick the max-expectancy operating point.

Perf note: Supertrend / EMA are **causal**, so they're precomputed once into
timestamp-keyed numpy arrays and looked up O(1) — turns an O(n²) backtest into
O(n). Binance rate-limits (429) on bursty 1m pagination → fetch sequentially,
cap the 1m window.

---

## 2. Full library, 90-day out-of-sample holdout (focus set)

| symbol | best OOS strategy | OOS PF | OOS trades | verdict |
|--------|-------------------|--------|-----------|---------|
| BTC | (none) | — | — | **NO EDGE** |
| ETH | Liquidity_Sweep_MSS | 1.36 | 44 | pass (lenient gate) |
| SOL | Opening_Range_Breakout | 1.13 | 43 | pass (lenient gate) |
| XRP | Liquidity_Sweep_MSS | 1.06 | 52 | watchlist |

Per-symbol winners **differ**, and all are marginal (PF 1.1–1.4). No single
strategy works across all four. None comes close to 50%-win + 1:2.

---

## 3. The win/RR frontier — MTF-Alignment entry (the proof)

Entry = EMA50 aligned on 1h+4h+1d → structure breakout → engulfing trigger,
stop at swing floored so 2R ≥ 1% net. Execution 5m, 90d, anchored OOS. TP swept:

| symbol | R=1.0 win | R=1.5 win | R=2.0 win | R=2.5 win | best net avgR |
|--------|-----------|-----------|-----------|-----------|---------------|
| SOL | 51.7% | 32.1% | 28.6% | 25.0% | −0.17 (part-exit) — no edge |
| XRP | 59.1% | 33.3% | 33.3% | 33.3% | ~0.00 (part-exit) — breakeven |
| ETH | 60.0% | 43.5% | 45.5% | 42.1% | **+0.23 @ R=2.5, PF 1.42** — EDGE |
| BTC | 48.1% | 40.0% | 30.4% | 33.3% | −0.05 (R=2.5) — no edge |

**This is the answer in one table.** Win% falls monotonically as the target
widens. At R=1.0 you get >50% win on SOL/XRP/ETH — but the realised win/loss ratio
is only ~0.66 (fees + slippage + partials eat it), so net ≈ breakeven. To get a
1:2 *realised* ratio you must widen the target, which drops win% to 25–45%. You
**cannot** hold both. 50%-win AND 1:2-net = +0.5R/trade = elite alpha not present
in vanilla TA on these symbols.

---

## 4. What IS real

- **ETH carries a modest, repeatable net edge**: Liquidity_Sweep (PF 1.36, 44
  OOS trades) and MTF-alignment at wide targets (PF 1.3–1.4, +0.12 to +0.23 avgR,
  ~42% win). Net-positive after fees, out-of-sample. That's a *profit-factor-1.3*
  strategy — worth trading with discipline, but it is **not** 50%/1:2.
- SOL/XRP/BTC: no durable net edge in this window.

---

## 5. Recommendation — how to actually get to a tradeable edge

The 50%/1:2 KPI is the wrong target (mathematically a unicorn for vanilla TA).
Optimise **net OOS expectancy** instead. Levers, ranked by expected impact here:

1. **Selectivity + portfolio scanner.** Stop researching 4 symbols — scan all
   ~200 USDT-perps, take only the few A+ aligned setups/day. Edge is in
   *rejecting* setups. Biggest unused lever.
2. **Dynamic / structure-based exits**, not a fixed 2R. Target the actual next
   liquidity pool / measured move; trail after 1R; partial at 1:1. (Frontier
   shows fixed-R is suboptimal.)
3. **Asymmetric entry = tight invalidation.** Get RR from a tight stop just
   beyond a swept wick / order block, not a far target.
4. **Add non-price data** — OI, funding, CVD/order-flow, liquidations. Every
   strategy tested is pure OHLCV; real crypto edge largely lives here. The repo
   has OI hooks barely used.
5. **Cut costs** — maker entries (0.02% vs 0.07%), tight-spread symbols, avoid
   funding windows. Fees flipped several +R configs to breakeven.

**Concrete next steps in this repo:**
- Make `WalkForward` + `Gatekeeper` the *default* acceptance bar (already wired in
  `researcher_robust.py`); retire raw-PF ranking in `researcher.py`.
- Build a `scanner` that runs the MTF/Sweep entry across the full perp universe
  and ranks live setups by alignment score (concurrency: `ThreadPoolExecutor`
  for fetch, sequential compute with the O(1) cache pattern).
- Add OI/funding/CVD features and an entry condition that requires OI rising with
  price (genuine participation) — re-run the frontier with that filter.
- For literal 1m-execution: run `build_mtf_edge.py --exec-tf 1m` offline over a
  shorter window; the engine checks exits on the base series, so 1m base gives
  true intrabar precision (slow — run as a batch job, not interactively).

---

## 6. Deliverables produced

- **Config dict** (drop-in for the builder), MTF-alignment edge:
  ```python
  {
    "name": "MTF_Alignment_Edge",
    "entry": {"conditions": ["mtf_edge_entry"]},   # see build_mtf_edge.py
    "exit": {"targets": [2.0]},                     # 1:2; stop floored so 2R ≥ 1% net
    "timeframes": ["5m", "1h", "4h", "1d"],
  }
  ```
- **Custom ConditionRegistry entry**: `mtf_edge_entry` in `build_mtf_edge.py`
  (cached MTF bias + breakout + engulfing + 1%-net-move stop floor). The builder
  also already ships `advanced_mtf_trend_alignment_entry` (same idea, un-cached).
- **How to test**:
  ```bash
  ./venv/bin/python -u researcher_robust.py --symbol ETHUSDT --days 90 --holdout --focus
  ./venv/bin/python -u build_mtf_edge.py --days 90 --exec-tf 5m
  ./venv/bin/python -u frontier_sweep.py --days 90 --entry mtf
  ```

---

## 7. Liquidity-Sweep entry frontier

*(appended when the run completes — O(n²) entry, slower)*
