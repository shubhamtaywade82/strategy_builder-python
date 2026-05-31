"""
Binance USD-M Futures Kline Loader
==================================
Paginated, rate-limit-aware historical OHLCV fetcher.
Based on verified Binance API contract (GET /fapi/v1/klines).

Public market data - no API key required.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from strategy_research.utils.config import DataConfig, TimeframeConfig

log = logging.getLogger("binance_loader")

_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


@dataclass
class BinanceDataManager:
    """Manages fetching and caching of multi-timeframe Binance data."""

    data_config: DataConfig = field(default_factory=DataConfig)
    tf_config: TimeframeConfig = field(default_factory=TimeframeConfig)
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_symbol(
        self,
        symbol: str,
        days: int = 60,
        timeframes: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch multi-timeframe OHLCV data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'SOLUSDT')
            days: Days of history to fetch
            timeframes: List of timeframe strings. Defaults to all MTF timeframes.

        Returns:
            Dict mapping timeframe -> DataFrame
        """
        symbol = symbol.upper()
        tfs = timeframes or self.tf_config.all_tfs
        end = int(time.time() * 1000)
        start = end - days * 86_400_000

        log.info(f"Fetching {symbol} data: {days}d across {tfs}")
        result = {}

        for tf in tfs:
            if tf not in self.tf_config.INTERVAL_MS:
                log.warning(f"Unsupported timeframe: {tf}")
                continue
            try:
                df = self._fetch_klines(symbol, tf, start, end)
                result[tf] = df
                log.info(f"  {tf}: {len(df)} bars")
                time.sleep(self.data_config.request_delay)
            except Exception as e:
                log.error(f"Failed to fetch {tf}: {e}")

        return result

    def _fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch paginated klines for a single timeframe."""
        step = self.tf_config.INTERVAL_MS[interval]
        rows: List[list] = []
        cursor = start_ms

        while cursor < end_ms:
            batch = self._request(symbol, interval, cursor, end_ms)
            if not batch:
                break

            rows.extend(batch)
            last_open = batch[-1][0]
            nxt = last_open + step
            if nxt <= cursor:
                break
            cursor = nxt

            if len(batch) < self.data_config.max_limit:
                break
            time.sleep(self.data_config.request_delay)

        return self._to_frame(rows, symbol, interval)

    def _request(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list:
        """Single API request with retry logic."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_ms),
            "endTime": int(end_ms),
            "limit": self.data_config.max_limit,
        }
        url = f"{self.data_config.base_url}/fapi/v1/klines"

        for attempt in range(self.data_config.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.data_config.timeout)

                if r.status_code in (429, 418):
                    wait = float(r.headers.get("Retry-After",
                        self.data_config.backoff_base * (2 ** attempt)))
                    log.warning(f"Rate limited ({r.status_code}); sleeping {wait:.1f}s")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json()

            except (requests.RequestException, ValueError) as exc:
                wait = self.data_config.backoff_base * (2 ** attempt)
                log.warning(f"Request failed ({exc}); retry in {wait:.1f}s")
                time.sleep(wait)

        raise RuntimeError(
            f"Klines fetch failed after {self.data_config.max_retries} retries: "
            f"{symbol} {interval}"
        )

    @staticmethod
    def _to_frame(rows: list, symbol: str, interval: str) -> pd.DataFrame:
        """Convert raw kline rows to a clean DataFrame."""
        if not rows:
            return pd.DataFrame(columns=[c for c in _COLS if c != "ignore"])

        df = pd.DataFrame(rows, columns=_COLS)
        df = df.drop(columns=["ignore"])

        # Numeric columns
        numeric_cols = [
            "open", "high", "low", "close", "volume", "quote_volume",
            "taker_buy_base", "taker_buy_quote"
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["n_trades"] = df["n_trades"].astype("int64")

        # Timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Deduplicate and sort
        df = df.drop_duplicates(subset="open_time")\
               .sort_values("open_time")\
               .reset_index(drop=True)

        df.attrs["symbol"] = symbol
        df.attrs["interval"] = interval

        return df

    def fetch_funding_info(self, symbol: str) -> Dict:
        """Fetch funding rate info for a symbol."""
        url = f"{self.data_config.base_url}/fapi/v1/fundingInfo"
        params = {"symbol": symbol.upper()}

        try:
            r = self.session.get(url, params=params, timeout=self.data_config.timeout)
            r.raise_for_status()
            data = r.json()
            return data[0] if isinstance(data, list) and data else {}
        except Exception as e:
            log.warning(f"Failed to fetch funding info: {e}")
            return {}

    def fetch_funding_rate_history(
        self,
        symbol: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical funding rates."""
        url = f"{self.data_config.base_url}/fapi/v1/fundingRate"
        params = {"symbol": symbol.upper(), "limit": limit}

        try:
            r = self.session.get(url, params=params, timeout=self.data_config.timeout)
            r.raise_for_status()
            data = r.json()

            df = pd.DataFrame(data)
            if df.empty:
                return df

            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            return df

        except Exception as e:
            log.warning(f"Failed to fetch funding history: {e}")
            return pd.DataFrame()

    def fetch_exchange_info(self) -> Dict:
        """Fetch exchange info including symbol specifications."""
        url = f"{self.data_config.base_url}/fapi/v1/exchangeInfo"

        try:
            r = self.session.get(url, timeout=self.data_config.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"Failed to fetch exchange info: {e}")
            return {}

    def get_symbol_specs(self, symbol: str) -> Dict:
        """Get specifications for a specific symbol."""
        info = self.fetch_exchange_info()
        symbols = info.get("symbols", [])

        for sym in symbols:
            if sym.get("symbol") == symbol.upper():
                return {
                    "symbol": sym.get("symbol"),
                    "status": sym.get("status"),
                    "baseAsset": sym.get("baseAsset"),
                    "quoteAsset": sym.get("quoteAsset"),
                    "filters": {f["filterType"]: f for f in sym.get("filters", [])},
                    " maintMarginPercent": sym.get("maintMarginPercent"),
                    "requiredMarginPercent": sym.get("requiredMarginPercent"),
                }
        return {}

    def get_funding_config(self, symbol: str) -> Tuple[float, float]:
        """Get per-symbol funding configuration (rate, interval_hours).

        Returns:
            (avg_funding_rate, interval_hours)
        """
        info = self.fetch_funding_info(symbol)
        history = self.fetch_funding_rate_history(symbol, limit=500)

        avg_rate = history["fundingRate"].mean() if not history.empty else 0.0001

        # Funding interval in hours
        interval_s = info.get("fundingIntervalHours", 8)
        if isinstance(interval_s, str):
            interval_s = int(interval_s)

        log.info(f"Funding config for {symbol}: rate={avg_rate:.6f}, interval={interval_s}h")
        return avg_rate, float(interval_s)


def fetch_and_align_mtf(
    symbol: str,
    days: int = 60,
    timeframes: Optional[List[str]] = None,
    config: Optional[DataConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience function: fetch multi-timeframe data for a symbol.

    Args:
        symbol: Trading pair
        days: Days of history
        timeframes: List of timeframes to fetch
        config: Optional DataConfig override

    Returns:
        Dict of timeframe -> DataFrame
    """
    manager = BinanceDataManager(data_config=config or DataConfig())
    return manager.fetch_symbol(symbol, days, timeframes)
