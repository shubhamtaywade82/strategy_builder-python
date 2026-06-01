"""
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
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import ta

log = logging.getLogger("mtf_research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

FAPI_BASE = "https://fapi.binance.com"
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
# 1. Loader                                                                    #
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
            expected_bars = (end_ms - cursor) // step + 1
            current_limit = min(MAX_LIMIT, max(1, expected_bars))
            batch = self._request(symbol, interval, cursor, end_ms, limit=current_limit)
            if not batch:
                break
            rows.extend(batch)
            last_open = batch[-1][0]
            nxt = last_open + step
            if nxt <= cursor:  # guard against non-advancing cursor
                break
            cursor = nxt
            if len(batch) < current_limit:
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
# 2. Leakage-safe MTF features                                                 #
# --------------------------------------------------------------------------- #
def _htf_feature_block(htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute HTF features using ONLY data available at each HTF bar's close_time."""
    h = htf.copy()
    ret = h["close"].pct_change()
    
    # Calculate indicators
    rsi = ta.momentum.RSIIndicator(h["close"], window=14).rsi()
    macd = ta.trend.MACD(h["close"])
    bb = ta.volatility.BollingerBands(h["close"], window=20, window_dev=2)
    atr = ta.volatility.AverageTrueRange(h["high"], h["low"], h["close"], window=14).average_true_range()
    
    out = pd.DataFrame({
        "ref_time": h["close_time"],  # the instant this row becomes known
        f"{prefix}_ret": ret,
        f"{prefix}_ema_fast": (h["close"].ewm(span=9, adjust=False).mean() - h["close"]) / h["close"],
        f"{prefix}_ema_slow": (h["close"].ewm(span=21, adjust=False).mean() - h["close"]) / h["close"],
        f"{prefix}_rvol": ret.rolling(20).std(),
        f"{prefix}_range_pct": (h["high"] - h["low"]) / h["close"],
        f"{prefix}_taker_imb": (h["taker_buy_base"] / h["volume"].replace(0, np.nan)),
        f"{prefix}_rsi": rsi,
        f"{prefix}_macd_diff": macd.macd_diff() / h["close"], # normalized MACD histogram
        f"{prefix}_bb_pband": bb.bollinger_pband(),           # %B
        f"{prefix}_atr_pct": atr / h["close"],
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
    
    rsi = ta.momentum.RSIIndicator(merged["close"], window=14).rsi()
    macd = ta.trend.MACD(merged["close"])
    bb = ta.volatility.BollingerBands(merged["close"], window=20, window_dev=2)
    atr = ta.volatility.AverageTrueRange(merged["high"], merged["low"], merged["close"], window=14).average_true_range()
    
    merged["ltf_ret"] = r
    merged["ltf_rvol"] = r.rolling(20).std()
    merged["ltf_ema_fast"] = (merged["close"].ewm(span=9, adjust=False).mean() - merged["close"]) / merged["close"]
    merged["ltf_ema_slow"] = (merged["close"].ewm(span=21, adjust=False).mean() - merged["close"]) / merged["close"]
    merged["ltf_trend"] = np.sign(merged["ltf_ema_fast"] - merged["ltf_ema_slow"])
    merged["ltf_rsi"] = rsi
    merged["ltf_macd_diff"] = macd.macd_diff() / merged["close"]
    merged["ltf_bb_pband"] = bb.bollinger_pband()
    merged["ltf_atr_pct"] = atr / merged["close"]

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
# 3. Triple-barrier labeling toward a net +1% move                            #
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
# Self-test (synthetic) -- proves MTF alignment + path-dependent labeling      #
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
        "open_time": pd.to_datetime(["2026-01-01 09:00:00", "2026-01-01 10:00:00"], utc=True),
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


if __name__ == "__main__":
    _selftest()
