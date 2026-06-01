"""
Binance USD-M Futures Data Fetcher
Public API - no key required.
Endpoint: GET /fapi/v1/klines
"""
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}

COLS = ["open_time","open","high","low","close","volume","close_time",
        "quote_volume","n_trades","taker_buy_base","taker_buy_quote","ignore"]

BASE_URL = "https://fapi.binance.com"


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch paginated klines for a single timeframe."""
    step = INTERVAL_MS[interval]
    rows = []
    cursor = start_ms
    session = requests.Session()

    while cursor < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": int(cursor),
            "endTime": int(end_ms),
            "limit": 1500,
        }
        try:
            r = session.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            nxt = batch[-1][0] + step
            if nxt <= cursor:
                break
            cursor = nxt
            if len(batch) < 1500:
                break
            time.sleep(0.15)
        except Exception as e:
            print(f"  Error fetching {interval} at {cursor}: {e}")
            time.sleep(1)
            break

    return _to_frame(rows, symbol, interval)


def _to_frame(rows: list, symbol: str, interval: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[c for c in COLS if c != "ignore"])
    df = pd.DataFrame(rows, columns=COLS).drop(columns=["ignore"])
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_trades"] = df["n_trades"].astype("int64")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df.attrs["symbol"] = symbol
    df.attrs["interval"] = interval
    return df


def fetch_symbol_mtf(symbol: str, days: int = 60, timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Fetch multi-timeframe data for a symbol."""
    tfs = timeframes or ["1m", "15m", "1h", "4h", "1d"]
    end = int(time.time() * 1000)
    start = end - days * 86_400_000
    print(f"Fetching {symbol} ({days}d) across {tfs}...")

    result = {}
    for tf in tfs:
        if tf not in INTERVAL_MS:
            continue
        df = fetch_klines(symbol, tf, start, end)
        result[tf] = df
        print(f"  {tf}: {len(df)} bars")
        time.sleep(0.2)
    return result
