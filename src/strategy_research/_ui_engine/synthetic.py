"""
Deterministic Synthetic Market Data
===================================
Generates reproducible multi-timeframe OHLCV that mirrors the Binance USD-M
kline schema. Used for:

- Offline / geo-blocked environments where ``fapi.binance.com`` is unreachable.
- Deterministic tests and demos (same seed -> byte-identical frames).

This is NEVER passed off as real data: every consumer that uses it stamps
``data_source = "synthetic"`` so the UI/report can show an explicit badge. The
generator builds a single 1m path and *resamples* it up to 15m/1h/4h/1d, so the
higher timeframes are internally consistent with the 1m series (no contradictory
bars), exactly as real aggregated klines would be.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Binance kline interval -> milliseconds (mirrors data_fetcher.INTERVAL_MS).
_INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}

# Rough, plausible starting prices so synthetic SOL/BTC/etc. look familiar.
_REF_PRICE = {
    "BTCUSDT": 65_000.0, "ETHUSDT": 3_400.0, "SOLUSDT": 150.0,
    "XRPUSDT": 0.55, "BNBUSDT": 580.0, "DOGEUSDT": 0.12,
}


def _seed_from(symbol: str, days: int, end_ms: int) -> int:
    """Stable 63-bit seed from inputs (Python ``hash`` is salted per-process)."""
    raw = f"{symbol.upper()}|{days}|{end_ms}".encode()
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**63)


def _build_1m(symbol: str, n: int, end_ms: int, seed: int) -> pd.DataFrame:
    """Generate a deterministic 1m OHLCV path with mild regime structure."""
    rng = np.random.default_rng(seed)
    p0 = _REF_PRICE.get(symbol.upper(), 100.0)

    # Per-minute log-returns: low baseline drift + slow sinusoidal regime so the
    # trend filter (4H EMA50/200) actually flips across the window, plus noise.
    minutes = np.arange(n)
    regime = 0.00000035 * np.sin(2 * np.pi * minutes / (n / 3.0))  # ~3 regimes
    base_vol = 0.0009  # per-minute sigma (~3.5%/day), realistic for SOL
    shocks = rng.normal(0.0, base_vol, size=n)
    # occasional volatility clusters
    cluster = rng.random(n) < 0.01
    shocks[cluster] *= 3.0
    log_ret = regime + shocks
    close = p0 * np.exp(np.cumsum(log_ret))

    open_ = np.empty(n)
    open_[0] = p0
    open_[1:] = close[:-1]

    # Intrabar extremes proportional to per-bar move + a noise wick.
    span = np.abs(close - open_)
    wick = (np.abs(shocks) * close) * (0.5 + rng.random(n))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick * (0.6 + 0.4 * rng.random(n))
    low = np.clip(low, 1e-9, None)

    # Volume rises with absolute move (activity clusters with volatility).
    base_v = _REF_PRICE.get(symbol.upper(), 100.0)
    vol = (1.0 + 40.0 * np.abs(log_ret) / base_vol) * (500.0 / max(p0, 1.0))
    vol *= 0.7 + 0.6 * rng.random(n)
    quote_vol = vol * close
    n_trades = np.maximum(1, (vol * (3.0 + 5.0 * rng.random(n))).astype(np.int64))
    taker_frac = np.clip(0.5 + 0.25 * np.tanh(log_ret / base_vol) + 0.05 * rng.standard_normal(n), 0.05, 0.95)
    taker_buy_base = vol * taker_frac
    taker_buy_quote = taker_buy_base * close

    step = _INTERVAL_MS["1m"]
    open_time_ms = end_ms - n * step + np.arange(n) * step
    open_time = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    close_time = pd.to_datetime(open_time_ms + step - 1, unit="ms", utc=True)

    return pd.DataFrame({
        "open_time": open_time,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "close_time": close_time,
        "quote_volume": quote_vol, "n_trades": n_trades,
        "taker_buy_base": taker_buy_base, "taker_buy_quote": taker_buy_quote,
        # internal: integer ms of open_time, used for version-robust resampling
        # (datetime64 resolution differs across pandas versions). Dropped before return.
        "_open_ms": open_time_ms.astype("int64"),
    })


_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "n_trades": "sum",
    "taker_buy_base": "sum", "taker_buy_quote": "sum",
}


def _resample(base_1m: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate the 1m frame up to ``interval`` (Binance-consistent bars)."""
    step = _INTERVAL_MS[interval]
    df = base_1m.copy()
    df["_bucket"] = (df["_open_ms"] // step) * step
    g = df.groupby("_bucket", sort=True).agg(_AGG).reset_index()
    g["open_time"] = pd.to_datetime(g["_bucket"], unit="ms", utc=True)
    g["close_time"] = pd.to_datetime(g["_bucket"] + step - 1, unit="ms", utc=True)
    g["n_trades"] = g["n_trades"].astype("int64")
    return g.drop(columns=["_bucket"]).reset_index(drop=True)


def make_synthetic_mtf(
    symbol: str,
    days: int = 60,
    timeframes: Optional[List[str]] = None,
    end_ms: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Deterministic multi-timeframe OHLCV, same shape as ``fetch_symbol_mtf``.

    Same (symbol, days, end_ms) -> identical frames every call.
    """
    tfs = timeframes or ["1m", "15m", "1h", "4h", "1d"]
    # Snap end to a whole-minute boundary so the window is reproducible.
    if end_ms is None:
        end_ms = 1_700_000_000_000  # fixed epoch anchor for full determinism
    end_ms = (end_ms // _INTERVAL_MS["1m"]) * _INTERVAL_MS["1m"]

    n_1m = days * 1440
    seed = _seed_from(symbol, days, end_ms)
    base_1m = _build_1m(symbol, n_1m, end_ms, seed)

    out: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        if tf not in _INTERVAL_MS:
            continue
        frame = base_1m.copy() if tf == "1m" else _resample(base_1m, tf)
        out[tf] = frame.drop(columns=["_open_ms"], errors="ignore").reset_index(drop=True)
    return out
