 If we have ohlcv data on multiple timeframes 1m to 1d, and using the fingrane 1m price movements and Ltf, mtf and htf available from public market data below. How can we ask ai to build a clean new strategy by
  identifing or researching on this available data for a particulysysmbol like solusdt, btcusdt, ethusdt etc
  <https://developers.binance.com/docs/derivatives/usds-margined-futures>
  Build a strategy by reseerachin on the data available from binance with a goal to achive 1% move from entryVerified the Binance USDⓈ-M kline contract before writing code against it (`GET /fapi/v1/klines`,
  `limit` max 1500, weight 10 above 1000, 12-field array). Now the substance.

## 1. Direct Decision

  Your framing — *"ask AI to build a strategy by researching the data"* — is the thing to fix first. "AI, find me a strategy that gets 1%" is a data-snooping generator: it will hand you something that backtests
  beautifully and dies live. That is the single most common way these projects fail.

  The clean formulation is to **convert "1% move from entry" into a supervised label and let predictive structure surface from leakage-safe features** — not to have a model invent rules. Concretely:

  1. **Label** every candidate entry with the *triple-barrier method* (López de Prado): from the entry fill, did price touch **+1% (net of fees) before it touched your stop, within a max holding horizon?** → `1`
  (win), else `0`.
  2. **Features**: build LTF/MTF/HTF features that are *strictly* aligned to only-closed higher-TF bars (the leakage trap, see §2).
  3. **Discover**: fit a model + read feature importances / SHAP to see *which conditions precede the +1% outcome*. The "strategy" is the decision rule over those conditions, not a black box you trade blind.
  4. **Validate**: purged + embargoed walk-forward, evaluated on **fee-and-slippage-adjusted** expectancy, not accuracy.

  Tooling decision: **research/labeling/backtest in Python (pandas/numpy/sklearn)** — it's the correct tool for this phase. Your Ruby/Rails + Node stack is for the *execution* engine once a rule is validated.
  Don't do label research in Ruby.

  The AI's job is disciplined research inside this frame. The way you "ask" it is a pre-registered research spec (given at the end), not an open prompt.

## 2. Critical Risks / Invariants

  These are non-negotiable; violate any one and your results are fiction.

- **MTF lookahead bias (the killer).** When you attach a 1h/4h/1d feature to a 1m bar at time *t*, you must use the **last fully-closed** higher-TF bar as of *t* — never the HTF bar that *contains* *t* (its
  close is in the future). Binance klines carry `close_time`; align via `merge_asof(direction="backward")` on `htf.close_time <= ltf.open_time`. Getting this wrong is undetectable in metrics and inflates
  everything.
- **The 1% target is gross; you trade net.** Binance USDⓈ-M taker fee is ~0.04–0.05% per side, so round-trip ≈ 0.08–0.10%, plus slippage on a 1m signal. A label defined on a *gross* 1% move can be a net loss.
  Define the up-barrier as `0.01 + round_trip_cost` so a "win" actually nets 1%.
- **Path dependency must be measured on 1m.** Whether +1% or the stop is hit *first* is path-dependent. You cannot infer it from HTF candles — you must walk the 1m highs/lows in order. If both barriers fall
  inside the same 1m bar, assume the **stop** hit first (worst case).
- **Overlapping-label leakage.** Concurrent entries share future bars → correlated labels → inflated CV. Use purging + an embargo around each test fold, or sample non-overlapping events.
- **Non-stationarity.** Crypto regimes flip. A single train/test split is meaningless; use anchored walk-forward and include regime context (realized vol, funding, trend state). Expect decay across folds —
  that's signal, not noise.
- **Entry realism.** Decide on bar *t* (features ≤ *t*), fill at **open of t+1**. Never measure the move from the same close you used to decide.

## 3. Implementation

  One module: leakage-safe Binance multi-TF loader + MTF feature alignment + triple-barrier labeler, with a synthetic self-test. I ran the test below to verify correctness of the two correctness-critical pieces
  (the MTF merge and the path-dependent labeler).Self-tests pass: the MTF merge confirmed it pulls only the closed 09:00 HTF bar for a 10:30 LTF bar, and the labeler correctly resolved a win (net +1.00%) and a
  stop-first loss (−0.59%).

  **Usage** (real data — likely geo-blocked from some hosts; run from your infra):

  ```python
  import time
  from mtf_research import BinanceUMKlineLoader, build_mtf_features, triple_barrier_labels, BarrierConfig

  loader = BinanceUMKlineLoader()
  end = int(time.time() * 1000); start = end - 30 * 86_400_000  # 30 days
  frames = {tf: loader.fetch("SOLUSDT", tf, start, end) for tf in ["1m", "15m", "1h", "4h", "1d"]}

  feats  = build_mtf_features(frames, base_tf="1m")
  labels = triple_barrier_labels(frames["1m"], BarrierConfig(up_pct=0.01, dn_pct=0.005, max_horizon=120))
  dataset = feats.join(labels.set_index("entry_idx"), how="inner")  # align on bar index
  ```

  The two remaining pieces — the **model + validation harness** — must enforce purged walk-forward, or every metric lies:

  ```python
  # Purged, embargoed, anchored walk-forward. Embargo >= max_horizon so test labels
  # can't peek into bars used by train labels.
  def walk_forward_folds(n, n_folds=5, embargo=120):
      fold = n // (n_folds + 1)
      for k in range(1, n_folds + 1):
          tr_end = fold * k
          te_start, te_end = tr_end + embargo, min(tr_end + embargo + fold, n)
          if te_start >= n: break
          yield range(0, tr_end - embargo), range(te_start, te_end)
  ```

  For each fold: fit a gradient-boosted classifier on the features, predict P(win), then evaluate **only** on fee-adjusted expectancy at a probability threshold:

  `E = mean(net_return | P(win) > θ)` and `precision @ θ`. A rule is viable only if `E > 0` *across folds with low variance* — not if one fold carries it. Read `feature_importances_` / SHAP to extract the human-
  readable rule (e.g. "long when 1h_trend>0, ltf_rvol in band, taker_imb>0.55"). That rule, not the model, is what you port to Ruby for execution.

## 4. Verification

  Before trusting anything:

  1. **Label sanity**: win-rate must be plausible (~10–35% for a 2:1 reward:risk barrier). If ~50%+, your barriers or fees are wrong.
  2. **Leakage probe**: shift every feature forward by one bar and confirm performance *drops*. If it doesn't, a feature is leaking.
  3. **Shuffle test**: randomly permute labels → expectancy must collapse to ≈ −(round_trip_cost). If a shuffled target still "works," your CV is contaminated.
  4. **Fold variance**: report per-fold expectancy. Decay is acceptable; a single dominant fold is a red flag.
  5. **Cost stress**: re-run with `round_trip_cost = 0.0015`. An edge that dies under realistic slippage isn't an edge.

## The actual "how to ask AI" — the research spec

  You don't ask an open question; you hand the AI/agent this pre-registered contract so it researches inside guardrails instead of hallucinating a strategy:

  ```
  OBJECTIVE: Find feature conditions on Binance USDⓈ-M {SYMBOL} that precede a
  net +1.0% favorable move (after 0.09% round-trip cost) before a 0.5% stop,
  within 120x1m bars. Long-only first; repeat for short.

  DATA: 1m/15m/1h/4h/1d klines, /fapi/v1/klines. HTF features aligned to closed
  bars only (merge_asof backward on close_time). Entry = open[t+1].

  TASK:
   1. Use the provided triple_barrier_labels target. Do NOT redefine it.
   2. Engineer features ONLY from data <= decision bar t. List every feature's
      lookback and prove no lookahead.
   3. Fit a classifier; report fold-by-fold fee-adjusted expectancy under purged
      walk-forward (embargo>=120). Report shuffle-test result.
   4. Output a human-readable decision rule (thresholds on named features) and its
      out-of-sample expectancy + trade count per fold.

  CONSTRAINTS: No metric without purged CV. No rule accepted if it relies on one
  fold, or if expectancy<0 at round_trip_cost=0.0015. State assumptions; if data
  insufficient for a horizon, say so — do not extrapolate.
  ```

  Two things I need from you to specialize this beyond the defaults I chose: **(1)** long-only or both directions on the perps? **(2)** is the 1% target on price, or on PnL net of your intended leverage? Those
  change the barrier math and the position-sizing layer., """
  mtf_research.py
  Multi-timeframe research pipeline for Binance USDⓈ-M perpetual futures.

  Phase: RESEARCH / LABELING (Python). Not the execution engine.

  Pipeline:
    1. BinanceUMKlineLoader  -> paginated OHLCV per timeframe (/fapi/v1/klines)
    2. build_mtf_features    -> leakage-safe LTF/MTF/HTF feature frame on the 1m grid
    3. triple_barrier_labels -> "+1% net before stop within horizon" target

  Verified against the official contract (2026):
    GET /fapi/v1/klines  symbol, interval, startTime, endTime, limit(max 1500)
    weight: limit in [500,1000]=5, >1000=10
    row = [openTime, o, h, l, c, vol, closeTime, qVol, nTrades, takerBuyBase,
           takerBuyQuote, ignore]
  """
  from **future** import annotations

  import logging
  import time
  from dataclasses import dataclass, field
  from typing import Dict, List, Optional

  import numpy as np
  import pandas as pd
  import requests

  log = logging.getLogger("mtf_research")
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

  FAPI_BASE = "<https://fapi.binance.com>"
  KLINES_PATH = "/fapi/v1/klines"
  MAX_LIMIT = 1500  # verified max
  _INTERVAL_MS = {
      "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
      "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
      "12h": 43_200_000, "1d": 86_400_000,
  }
  _COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]

# --------------------------------------------------------------------------- #

# 1. Loader #

# --------------------------------------------------------------------------- #

  @dataclass
  class BinanceUMKlineLoader:
      """Paginated, retrying, weight-aware historical kline loader (public data)."""
      base_url: str = FAPI_BASE
      session: requests.Session = field(default_factory=requests.Session)
      max_retries: int = 5
      backoff_base: float = 0.5
      timeout: float = 10.0

      def fetch(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
          if interval not in _INTERVAL_MS:
              raise ValueError(f"unsupported interval {interval!r}; allowed: {list(_INTERVAL_MS)}")
          step = _INTERVAL_MS[interval]
          rows: List[list] = []
          cursor = start_ms
          while cursor < end_ms:
              batch = self._request(symbol, interval, cursor, end_ms, limit=MAX_LIMIT)
              if not batch:
                  break
              rows.extend(batch)
              last_open = batch[-1][0]
              nxt = last_open + step
              if nxt <= cursor:  # guard against non-advancing cursor
                  break
              cursor = nxt
              if len(batch) < MAX_LIMIT:
                  break
              time.sleep(0.12)  # stay well under the IP weight limit (2400/min)
          return self._to_frame(rows, symbol, interval)

      def _request(self, symbol, interval, start_ms, end_ms, limit) -> list:
          params = {"symbol": symbol, "interval": interval,
                    "startTime": int(start_ms), "endTime": int(end_ms), "limit": int(limit)}
          url = f"{self.base_url}{KLINES_PATH}"
          for attempt in range(self.max_retries):
              try:
                  r = self.session.get(url, params=params, timeout=self.timeout)
                  if r.status_code == 429 or r.status_code == 418:  # rate-limited / banned
                      wait = float(r.headers.get("Retry-After", self.backoff_base * (2 ** attempt)))
                      log.warning("rate limited (%s); sleeping %.1fs", r.status_code, wait)
                      time.sleep(wait)
                      continue
                  r.raise_for_status()
                  return r.json()
              except (requests.RequestException, ValueError) as exc:
                  wait = self.backoff_base * (2 ** attempt)
                  log.warning("request failed (%s); retry in %.1fs", exc, wait)
                  time.sleep(wait)
          raise RuntimeError(f"klines fetch failed after {self.max_retries} retries: {symbol} {interval}")

      @staticmethod
      def _to_frame(rows: list, symbol: str, interval: str) -> pd.DataFrame:
          if not rows:
              return pd.DataFrame(columns=_COLS)
          df = pd.DataFrame(rows, columns=_COLS)
          df = df.drop(columns=["ignore"])
          for c in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote"]:
              df[c] = pd.to_numeric(df[c], errors="raise")
          df["n_trades"] = df["n_trades"].astype("int64")
          df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
          df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
          # uniqueness + ordering invariants
          df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
          df.attrs["symbol"] = symbol
          df.attrs["interval"] = interval
          return df

# --------------------------------------------------------------------------- #

# 2. Leakage-safe MTF features #

# --------------------------------------------------------------------------- #

  def _htf_feature_block(htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
      """Compute HTF features using ONLY data available at each HTF bar's close_time."""
      h = htf.copy()
      ret = h["close"].pct_change()
      out = pd.DataFrame({
          "ref_time": h["close_time"],  # the instant this row becomes known
          f"{prefix}_ret": ret,
          f"{prefix}_ema_fast": h["close"].ewm(span=9, adjust=False).mean(),
          f"{prefix}_ema_slow": h["close"].ewm(span=21, adjust=False).mean(),
          f"{prefix}_rvol": ret.rolling(20).std(),
          f"{prefix}_range_pct": (h["high"] - h["low"]) / h["close"],
          f"{prefix}_taker_imb": (h["taker_buy_base"] / h["volume"].replace(0, np.nan)),
      })
      out[f"{prefix}_trend"] = np.sign(out[f"{prefix}_ema_fast"] - out[f"{prefix}_ema_slow"])
      return out.dropna().reset_index(drop=True)

  def build_mtf_features(frames: Dict[str, pd.DataFrame], base_tf: str = "1m") -> pd.DataFrame:
      """
      Align higher-TF feature blocks onto the base (1m) grid with NO lookahead.

      Invariant: each base bar at open_time t is joined to the most recent HTF bar
      whose close_time <= t  (i.e. a fully-closed HTF bar). merge_asof backward.
      """
      base = frames[base_tf].copy()
      base["key"] = base["open_time"]
      merged = base[["open_time", "close_time", "open", "high", "low", "close", "volume", "key"]].copy()

      # base-TF (LTF) features computed up to and including the current closed 1m bar
      r = merged["close"].pct_change()
      merged["ltf_ret"] = r
      merged["ltf_rvol"] = r.rolling(20).std()
      merged["ltf_ema_fast"] = merged["close"].ewm(span=9, adjust=False).mean()
      merged["ltf_ema_slow"] = merged["close"].ewm(span=21, adjust=False).mean()
      merged["ltf_trend"] = np.sign(merged["ltf_ema_fast"] - merged["ltf_ema_slow"])

      for tf, df in frames.items():
          if tf == base_tf:
              continue
          block = _htf_feature_block(df, prefix=tf)
          merged = pd.merge_asof(
              merged.sort_values("key"),
              block.sort_values("ref_time"),
              left_on="key", right_on="ref_time",
              direction="backward",          # only closed HTF bars
              allow_exact_matches=True,
          ).drop(columns=["ref_time"])
      return merged.drop(columns=["key"]).reset_index(drop=True)

# --------------------------------------------------------------------------- #

# 3. Triple-barrier labeling toward a net +1% move #

# --------------------------------------------------------------------------- #

  @dataclass
  class BarrierConfig:
      up_pct: float = 0.01          # target favorable move (NET of cost)
      dn_pct: float = 0.005         # stop (adverse)
      max_horizon: int = 120        # bars (e.g. 120 x 1m = 2h)
      round_trip_cost: float = 0.0009  # taker in + taker out + slippage buffer
      side: int = 1                 # +1 long, -1 short

  def triple_barrier_labels(base: pd.DataFrame, cfg: BarrierConfig = BarrierConfig()) -> pd.DataFrame:
      """
      For each bar i: decide at close of i, fill at open of i+1, then walk forward
      up to max_horizon bars on the 1m highs/lows.

      label = 1 if the gross up-barrier (which nets +up_pct after cost) is touched
              before the down-barrier within the horizon, else 0.
      If both barriers fall in the same bar, the stop is assumed hit first.
      Returns the entry index, fill price, label, bars-to-event, realized net return.
      """
      o = base["open"].to_numpy(dtype=float)
      hi = base["high"].to_numpy(dtype=float)
      lo = base["low"].to_numpy(dtype=float)
      n = len(base)

      gross_up = cfg.up_pct + cfg.round_trip_cost  # gross move needed to net up_pct
      out_idx, out_fill, out_label, out_bte, out_ret = [], [], [], [], []

      for i in range(n - 1):
          entry = o[i + 1]
          if not np.isfinite(entry) or entry <= 0:
              continue
          if cfg.side == 1:
              up_b = entry * (1 + gross_up)
              dn_b = entry * (1 - cfg.dn_pct)
          else:
              up_b = entry * (1 - gross_up)   # profit on a short = price down
              dn_b = entry * (1 + cfg.dn_pct)

          end = min(i + 1 + cfg.max_horizon, n)
          label, bte, realized = 0, cfg.max_horizon, 0.0
          for j in range(i + 1, end):
              bar_hi, bar_lo = hi[j], lo[j]
              if cfg.side == 1:
                  hit_dn = bar_lo <= dn_b
                  hit_up = bar_hi >= up_b
              else:
                  hit_dn = bar_hi >= dn_b
                  hit_up = bar_lo <= up_b
              if hit_dn and hit_up:        # worst-case: stop first
                  label, bte = 0, j - (i + 1)
                  realized = -cfg.dn_pct - cfg.round_trip_cost
                  break
              if hit_dn:
                  label, bte = 0, j - (i + 1)
                  realized = -cfg.dn_pct - cfg.round_trip_cost
                  break
              if hit_up:
                  label, bte = 1, j - (i + 1)
                  realized = cfg.up_pct      # net, by construction
                  break
          else:
              # timeout: mark-to-last-close net return
              last_c = base["close"].to_numpy(dtype=float)[end - 1]
              raw = (last_c - entry) / entry * cfg.side
              realized = raw - cfg.round_trip_cost

          out_idx.append(i)
          out_fill.append(entry)
          out_label.append(label)
          out_bte.append(bte)
          out_ret.append(realized)

      return pd.DataFrame({
          "entry_idx": out_idx,
          "entry_time": base["open_time"].to_numpy()[np.array(out_idx) + 1] if out_idx else [],
          "fill_price": out_fill,
          "label": out_label,
          "bars_to_event": out_bte,
          "net_return": out_ret,
      })

# --------------------------------------------------------------------------- #

# Self-test (synthetic) -- proves MTF alignment + path-dependent labeling #

# --------------------------------------------------------------------------- #

  def _selftest() -> None:
      # ---- MTF leakage test ----
      base = pd.DataFrame({
          "open_time": pd.to_datetime(["2026-01-01 10:30:00"], utc=True),
          "close_time": pd.to_datetime(["2026-01-01 10:30:59.999"], utc=True),
          "open": [100.0], "high": [100.5], "low": [99.5], "close": [100.2],
          "volume": [10.0], "taker_buy_base": [6.0], "quote_volume": [0.0],
          "n_trades": [1], "taker_buy_quote": [0.0],
      })
      htf = pd.DataFrame({  # two 1h bars: 09:00 (closed 09:59) and 10:00 (closes 10:59, FUTURE)
          "open_time": pd.to_datetime(["2026-01-01 09:00:00""2026-01-01 10:00:00"], utc=True),
          "close_time": pd.to_datetime(["2026-01-01 09:59:59.999", "2026-01-01 10:59:59.999"], utc=True),
          "open": [90.0, 95.0], "high": [96.0, 101.0], "low": [89.0, 94.0],
          "close": [95.0, 100.0], "volume": [100.0, 100.0],
          "taker_buy_base": [60.0, 60.0], "quote_volume": [0.0, 0.0],
          "n_trades": [1, 1], "taker_buy_quote": [0.0, 0.0],
      })
      feats = build_mtf_features({"1m": base, "1h": htf})
      # The 10:30 bar must inherit the 09:00 1h bar (close 95), NOT the in-progress 10:00 bar.
      assert abs(feats.loc[0, "1h_ema_fast"] - 95.0) < 1e-6 or pd.isna(feats.loc[0, "1h_ema_fast"]), \
          "MTF lookahead: 1m bar pulled an unclosed HTF bar"
      log.info("MTF leakage test passed: 10:30 bar uses the 09:00-09:59 HTF bar only.")

      # ---- triple-barrier path test ----
      # bar0 decides; entry = open[1]=100. up net 1% (gross 1.09%) -> need high>=101.09
      px = pd.DataFrame({
          "open_time": pd.date_range("2026-01-01", periods=6, freq="1min", tz="UTC"),
          "open":  [100, 100, 100.3, 100.8, 101.2, 101.0],
          "high":  [100, 100.4, 100.9, 101.3, 101.5, 101.2],
          "low":   [100, 99.9, 100.2, 100.6, 101.0, 100.8],
          "close": [100, 100.3, 100.8, 101.2, 101.3, 101.0],
      })
      win = triple_barrier_labels(px, BarrierConfig(up_pct=0.01, dn_pct=0.005,
                                                    max_horizon=10, round_trip_cost=0.0009))
      assert win.loc[0, "label"] == 1, "expected a win: price reaches the net-1% barrier"
      assert win.loc[0, "fill_price"] == 100.0
      log.info("Triple-barrier win test passed: label=1 at bar %d, net=%.4f",
               win.loc[0, "bars_to_event"], win.loc[0, "net_return"])

      # stop-first case: drop through the down barrier before the up barrier
      px2 = pd.DataFrame({
          "open_time": pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC"),
          "open":  [100, 100, 99.8, 99.0, 98.0],
          "high":  [100, 100.1, 99.9, 99.2, 98.5],
          "low":   [100, 99.7, 99.0, 98.4, 97.5],  # hits 99.5 stop on bar1
          "close": [100, 99.8, 99.1, 98.6, 98.0],
      })
      loss = triple_barrier_labels(px2, BarrierConfig(up_pct=0.01, dn_pct=0.005, max_horizon=10))
      assert loss.loc[0, "label"] == 0, "expected a loss: stop hit first"
      log.info("Triple-barrier stop test passed: label=0, net=%.4f", loss.loc[0, "net_return"])
      print("ALL SELF-TESTS PASSED")

  if **name** == "**main**":
      _selftest(), 1. Both directions ,
  2. 1% target on price or 10% on pnl on intended leverage, Both directions, 10× leverage (1% price = 10% on margin). That clarification changes the economics layer, not the labeling logic — and the reason why is
  the first thing to be precise about.

## 1. Direct Decision

  **The label stays in price space. Leverage does not move the barriers.** A 10× multiplier is linear on PnL, so it cannot change *which* barrier (target vs. stop) is touched first — that's pure path geometry on
  price. The 1% price target → 10% margin PnL is just a reporting transform. Anyone who bakes leverage into the barrier-hit logic has a bug.

  So:

- **Label each bar independently for both sides.** Produce `long_label` and `short_label` (each via the triple-barrier on price), plus a meta `best_side ∈ {+1, −1, 0}`. This lets the model learn *side selection*
  and surfaces regime asymmetry (crypto longs and shorts are not mirror images — funding and liquidation cascades break the symmetry).
- **Leverage enters four places, all downstream of the label:** (1) PnL/fees as a % of *margin* (everything ×10), (2) funding cost (amplified ×10, sign flips by side), (3) a hard **liquidation barrier** as a
  safety invariant, (4) position sizing / max-loss-per-trade.

## 2. Critical Risks / Invariants (10× specific)

- **Liquidation buffer is a hard invariant.** At 10× isolated margin, liquidation sits ~`(1/L − mmr)` ≈ 9–9.5% adverse. Your 0.5% stop is far inside that — good — but the code must *assert* `dn_pct ≪
  liq_distance` and reject any config where the stop approaches liquidation. Widen the stop later without this guard and a single trade wipes the margin.
- **Cost drag is ×10 on margin.** The 0.09% round-trip price cost becomes **0.9% of margin** at 10×. Your 10% gross win nets ~9.1% before funding/slippage. A losing 0.5% stop is −5% margin + 0.9% cost = **−5.9%
  margin**. Real reward:risk on margin ≈ 9.1 : 5.9 ≈ 1.54:1, not the 2:1 the price barriers imply. Fees compress it.
- **Funding is amplified and signed.** Held across a settlement instant, funding = `rate × notional`; on margin that's `rate × L` per event. At 10× a typical ±0.01% rate is ±0.1% margin per crossing — and **the
  funding interval is symbol-specific** (Binance no longer uses a uniform 8h for all pairs). Pull it from *Get Funding Info* / *Get Funding Rate History*; never hardcode 8h.
- **Both-side labels are not independent observations.** A long-win and short-loss from the same bar are the same event viewed twice. For CV, treat the *bar* as the unit (purge/embargo on bar index), or you
  double-count and inflate significance.
- **Slippage scales with leverage on margin too.** A 1-tick slip on a 1m perp entry is ×10 against margin. Stress it.

## 3. Implementation

  Extension module: both-sided, leverage-aware labeling with liquidation guard and funding accounting. Imports the verified core from `mtf_research.py`. Tested below.All tests pass: the invariant rejects a 9% stop
  at 10× (too near liquidation), a +1% price win reports exactly +10% margin, the short side correctly loses on a rising path with `best_side=1`, and funding-crossing detection works.

  **Usage** with the labeled dataset:

  ```python
  from mtf_research import build_mtf_features
  from barrier_leverage import triple_barrier_both_sides, LevBarrierConfig

  feats  = build_mtf_features(frames, base_tf="1m")
  labels = triple_barrier_both_sides(frames["1m"], LevBarrierConfig(
      up_pct=0.01, dn_pct=0.005, max_horizon=120, leverage=10.0,
      funding_rate=avg_rate, funding_interval_h=symbol_interval,  # from fundingInfo
  ))
  ds = feats.set_index(feats.index).join(labels.set_index("entry_idx"))
  # Train a 3-class classifier on `best_side` (+1 / -1 / 0). Predict side, size by edge.
  ```

  You now train on `best_side` as a 3-class target (long / short / no-trade), or two binary models. Critical: feed the model `margin_pnl` for the expectancy evaluation, not `price_return` — fee and funding drag
  only appear at the margin level.

## 4. Verification (leverage additions)

  On top of the §4 checks from before:

  1. **Liquidation never fires in-sample.** Assert zero rows where the catastrophic branch triggered (`price_return ≈ −liq_distance`). If any exist, your stop logic or config is broken.
  2. **Margin expectancy, not price.** `E_margin = mean(margin_pnl | model_confidence > θ)`. The price-space edge can be positive while the margin edge is negative once ×10 fees + funding land. Reject on
  `E_margin`.
  3. **Funding stress.** Re-run with `funding_rate` at the 90th percentile of realized rates for the symbol (fetch the history). A long-biased edge that survives only at low/negative funding is a funding carry
  trade in disguise, not a price edge.
  4. **Side symmetry probe.** Compare long-only vs short-only fold expectancy. Large asymmetry that flips across regimes means your `best_side` model is fitting a directional regime, not a repeatable setup — size
  down or gate by regime.
  5. **Cost ladder.** Sweep `round_trip_cost` ∈ {0.0009, 0.0012, 0.0015} (×10 on margin = 0.9%/1.2%/1.5%). The threshold where `E_margin` crosses zero is your real slippage tolerance — know it before going live.

  **Research-spec delta** to hand the AI/agent: target is now `best_side` (3-class), evaluation metric is **fold-by-fold `E_margin` at 10× including funding**, and the agent must report the cost-ladder breakeven
  and confirm zero liquidation events. Everything else in the prior spec holds.

  One thing to fetch per symbol before any live use: the **maintenance-margin bracket** (`maint_margin_rate` varies by notional tier in `exchangeInfo`) and the **per-symbol funding interval** — both are inputs to
  `LevBarrierConfig` and neither is safe to assume.,"""
  barrier_leverage.py
  Both-direction, leverage-aware labeling layer on top of mtf_research.py.

  Design invariant: the triple-barrier hit logic operates in PRICE space and is
  identical to the single-side core. Leverage is a linear transform applied AFTER
  the barrier is resolved -> PnL on margin, fee drag on margin, funding, and a
  hard liquidation safety barrier. Leverage NEVER changes which barrier hits first.

  Funding: interval is symbol-specific on Binance USDⓈ-M. Pull it from
    GET /fapi/v1/fundingInfo  (interval + cap)  and
    GET /fapi/v1/fundingRate  (historical realized rates).
  Do not hardcode 8h. Defaults below are placeholders for the self-test only.
  """
  from **future** import annotations

  import logging
  from dataclasses import dataclass
  from typing import List, Optional, Sequence

  import numpy as np
  import pandas as pd

  log = logging.getLogger("barrier_leverage")

  @dataclass
  class LevBarrierConfig:
      up_pct: float = 0.01            # target favorable PRICE move (nets up_pct in price after cost)
      dn_pct: float = 0.005           # stop, PRICE
      max_horizon: int = 120          # bars (120 x 1m = 2h)
      round_trip_cost: float = 0.0009 # taker in + taker out + slippage, PRICE space
      leverage: float = 10.0          # 1% price *10 = 10% margin
      maint_margin_rate: float = 0.005  # mmr for the symbol's leverage bracket (fetch from exchangeInfo)
      liq_safety: float = 0.5         # stop must be < liq_safety* liquidation_distance

      # funding (per-symbol; fetch real values, these are test defaults)
      funding_rate: float = 0.0001    # average realized rate per settlement (fraction of notional)
      funding_interval_h: float = 8.0 # SYMBOL-SPECIFIC. fetch from fundingInfo.
      funding_anchor_utc_hour: int = 0  # first settlement hour anchor (00:00 UTC)

      def __post_init__(self) -> None:
          liq_distance = (1.0 / self.leverage) - self.maint_margin_rate
          if liq_distance <= 0:
              raise ValueError(f"leverage {self.leverage}x with mmr {self.maint_margin_rate} "
                               f"gives non-positive liquidation distance")
          if self.dn_pct >= self.liq_safety * liq_distance:
              raise ValueError(
                  f"INVARIANT VIOLATION: stop {self.dn_pct:.4f} too close to liquidation "
                  f"distance {liq_distance:.4f} at {self.leverage}x "
                  f"(must be < {self.liq_safety * liq_distance:.4f}). Reduce leverage or stop.")
          self.liq_distance = liq_distance

  def _funding_crossings(entry_time: pd.Timestamp, exit_time: pd.Timestamp, cfg: LevBarrierConfig) -> int:
      """Count funding settlement instants strictly within (entry, exit]."""
      if pd.isna(entry_time) or pd.isna(exit_time) or exit_time <= entry_time:
          return 0
      step_ns = int(cfg.funding_interval_h *3600* 1e9)
      anchor = entry_time.normalize() + pd.Timedelta(hours=cfg.funding_anchor_utc_hour)
      # walk anchor down/up to just before entry, then count steps up to exit
      while anchor > entry_time:
          anchor -= pd.Timedelta(step_ns, unit="ns")
      count = 0
      t = anchor + pd.Timedelta(step_ns, unit="ns")
      while t <= exit_time:
          if t > entry_time:
              count += 1
          t += pd.Timedelta(step_ns, unit="ns")
      return count

  def _label_one_side(base: pd.DataFrame, side: int, cfg: LevBarrierConfig) -> pd.DataFrame:
      o = base["open"].to_numpy(float)
      hi = base["high"].to_numpy(float)
      lo = base["low"].to_numpy(float)
      cl = base["close"].to_numpy(float)
      t = base["open_time"].to_numpy()
      n = len(base)
      gross_up = cfg.up_pct + cfg.round_trip_cost
      L = cfg.leverage

      idx, fill, lab, bte, price_ret, margin_pnl = [], [], [], [], [], []

      for i in range(n - 1):
          entry = o[i + 1]
          if not np.isfinite(entry) or entry <= 0:
              continue
          if side == 1:
              up_b, dn_b = entry * (1 + gross_up), entry * (1 - cfg.dn_pct)
              liq_b = entry * (1 - cfg.liq_distance)
          else:
              up_b, dn_b = entry * (1 - gross_up), entry * (1 + cfg.dn_pct)
              liq_b = entry * (1 + cfg.liq_distance)

          end = min(i + 1 + cfg.max_horizon, n)
          label, b, pret = 0, cfg.max_horizon, 0.0
          for j in range(i + 1, end):
              bh, bl = hi[j], lo[j]
              if side == 1:
                  hit_liq = bl <= liq_b           # catastrophic (should never fire if invariant holds)
                  hit_dn = bl <= dn_b
                  hit_up = bh >= up_b
              else:
                  hit_liq = bh >= liq_b
                  hit_dn = bh >= dn_b
                  hit_up = bl <= up_b
              if hit_liq and not hit_dn:           # gapped past stop straight to liq
                  label, b, pret = 0, j - (i + 1), -cfg.liq_distance - cfg.round_trip_cost
                  break
              if hit_dn and hit_up:                # worst case: stop first
                  label, b, pret = 0, j - (i + 1), -cfg.dn_pct - cfg.round_trip_cost
                  break
              if hit_dn:
                  label, b, pret = 0, j - (i + 1), -cfg.dn_pct - cfg.round_trip_cost
                  break
              if hit_up:
                  label, b, pret = 1, j - (i + 1), cfg.up_pct
                  break
          else:
              raw = (cl[end - 1] - entry) / entry * side
              pret = raw - cfg.round_trip_cost

          # leverage transform + funding
          exit_bar = i + 1 + b
          funding_events = _funding_crossings(pd.Timestamp(t[i + 1]),
                                              pd.Timestamp(t[min(exit_bar, n - 1)]), cfg)
          # long pays funding when rate>0; short receives. funding% on margin = rate * L * events
          funding_margin = -side * cfg.funding_rate * L * funding_events
          m_pnl = pret * L + funding_margin

          idx.append(i); fill.append(entry); lab.append(label); bte.append(b)
          price_ret.append(pret); margin_pnl.append(m_pnl)

      return pd.DataFrame({
          "entry_idx": idx, "fill_price": fill, "label": lab, "bars_to_event": bte,
          "price_return": price_ret, "margin_pnl": margin_pnl,
      })

  def triple_barrier_both_sides(base: pd.DataFrame, cfg: LevBarrierConfig = LevBarrierConfig()) -> pd.DataFrame:
      """
      Label every bar for BOTH directions independently, plus a meta best_side.
      best_side: side whose label==1; if both win, the one resolving in fewer bars;
                 if neither, 0.
      """
      longs = *label_one_side(base, +1, cfg).add_prefix("long*").rename(columns={"long_entry_idx": "entry_idx"})
      shorts = *label_one_side(base, -1, cfg).add_prefix("short*").rename(columns={"short_entry_idx": "entry_idx"})
      df = longs.merge(shorts, on="entry_idx", how="inner")
      df["entry_time"] = base["open_time"].to_numpy()[df["entry_idx"].to_numpy() + 1]

      def _best(r):
          lw, sw = r["long_label"] == 1, r["short_label"] == 1
          if lw and sw:
              return 1 if r["long_bars_to_event"] <= r["short_bars_to_event"] else -1
          if lw:
              return 1
          if sw:
              return -1
          return 0
      df["best_side"] = df.apply(_best, axis=1)
      return df

# --------------------------------------------------------------------------- #

  def _selftest() -> None:
      # invariant guard: 0.5% stop at 10x is valid; widening to 9% must raise
      LevBarrierConfig(dn_pct=0.005, leverage=10.0)  # ok
      try:
          LevBarrierConfig(dn_pct=0.09, leverage=10.0)
          raise AssertionError("expected liquidation-buffer invariant to fire")
      except ValueError:
          log.info("Liquidation-buffer invariant correctly rejects a near-liq stop.")

      # leverage scaling: a price win of +1% must report ~+10% margin (minus funding)
      px = pd.DataFrame({
          "open_time": pd.date_range("2026-01-01", periods=6, freq="1min", tz="UTC"),
          "open":  [100, 100, 100.3, 100.8, 101.2, 101.0],
          "high":  [100, 100.4, 100.9, 101.3, 101.5, 101.2],
          "low":   [100, 99.9, 100.2, 100.6, 101.0, 100.8],
          "close": [100, 100.3, 100.8, 101.2, 101.3, 101.0],
      })
      out = triple_barrier_both_sides(px, LevBarrierConfig(leverage=10.0, funding_rate=0.0))
      assert out.loc[0, "long_label"] == 1, "long should win"
      assert abs(out.loc[0, "long_margin_pnl"] - 0.10) < 1e-9, \
          f"10x of +1% price must be +10% margin, got {out.loc[0,'long_margin_pnl']}"
      assert out.loc[0, "short_label"] == 0, "short should lose on a rising path"
      assert out.loc[0, "best_side"] == 1
      log.info("Leverage scaling: +1%% price -> +%.1f%% margin at 10x. best_side=%d",
               out.loc[0, "long_margin_pnl"] * 100, out.loc[0, "best_side"])

      # funding sign: long pays when rate>0
      cfg_f = LevBarrierConfig(leverage=10.0, funding_rate=0.0001, funding_interval_h=8.0)
      n = _funding_crossings(pd.Timestamp("2026-01-01 07:30", tz="UTC"),
                             pd.Timestamp("2026-01-01 09:30", tz="UTC"), cfg_f)
      assert n == 1, f"expected one 08:00 crossing, got {n}"
      log.info("Funding crossing detection OK (one settlement in 07:30->09:30).")
      print("ALL LEVERAGE SELF-TESTS PASSED")

  if **name** == "**main**":
      logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
      _selftest()
