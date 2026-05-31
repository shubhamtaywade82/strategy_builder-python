"""
Binance USD-M Futures REST Client
==================================
Real-time historical + live data from Binance.
Drops into: src/strategy_builder/market_data/binance_rest_client.py
"""
import time
import hmac
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode
import requests
import pandas as pd

from ..domain import Candle
from ..exceptions import DataError


BASE_URL = "https://fapi.binance.com"
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}


class BinanceRestClient:
    """Production Binance USD-M Futures REST client.
    
    Public endpoints (no API key):
        - GET /fapi/v1/klines (historical OHLCV)
        - GET /fapi/v1/ticker/price (current price)
        - GET /fapi/v1/ticker/24hr (24h stats)
        - GET /fapi/v1/fundingRate (funding history)
    
    Private endpoints (API key required):
        - GET /fapi/v2/account (account info)
        - POST /fapi/v1/order (place order)
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "strategy_builder/1.0",
        })
        if api_key:
            self.session.headers["X-MBX-APIKEY"] = api_key

    # ── Public Endpoints ──────────────────────────────────────────

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1500,
    ) -> List[Dict]:
        """Fetch klines (paginated internally if needed).
        
        Args:
            symbol: e.g. "SOLUSDT"
            interval: e.g. "1m", "15m", "1h", "4h", "1d"
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            limit: Max 1500 per request
        
        Returns:
            List of raw kline dicts
        """
        if interval not in INTERVAL_MS:
            raise DataError(f"Invalid interval: {interval}")

        url = f"{BASE_URL}/fapi/v1/klines"
        all_rows: List[Dict] = []
        step = INTERVAL_MS[interval]
        cursor = start_ms or 0
        end = end_ms or int(time.time() * 1000)

        while cursor < end:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": end,
                "limit": limit,
            }
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    retry = float(resp.headers.get("Retry-After", 2))
                    time.sleep(retry)
                    continue
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                all_rows.extend(batch)
                nxt = batch[-1][0] + step
                if nxt <= cursor:
                    break
                cursor = nxt
                if len(batch) < limit:
                    break
                time.sleep(0.12)  # Rate limit: ~10 req/sec
            except requests.RequestException as e:
                raise DataError(f"Klines fetch failed: {e}")

        return all_rows

    def get_klines_df(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        """Fetch klines as a clean DataFrame."""
        rows = self.get_klines(symbol, interval, start_ms, end_ms, limit)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df = df.drop(columns=["ignore"])

        numeric = ["open", "high", "low", "close", "volume", "quote_volume",
                   "taker_buy_base", "taker_buy_quote"]
        for c in numeric:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["n_trades"] = df["n_trades"].astype("int64")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

        return df

    def get_klines_candles(
        self,
        symbol: str,
        interval: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1500,
    ) -> List[Candle]:
        """Fetch klines as domain Candle objects (integrates with existing pipeline)."""
        rows = self.get_klines(symbol, interval, start_ms, end_ms, limit)
        candles = []
        for row in rows:
            # row[0]=open_time, [1]=open, [2]=high, [3]=low, [4]=close, [5]=volume
            candles.append(Candle(
                timestamp=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            ))
        return candles

    def get_candles_days(self, symbol: str, interval: str, days: int) -> List[Candle]:
        """Convenience: fetch last N days of candles."""
        end = int(time.time() * 1000)
        start = end - days * 86_400_000
        return self.get_klines_candles(symbol, interval, start, end)

    def get_ticker_price(self, symbol: str) -> Dict:
        """Get current mark price."""
        url = f"{BASE_URL}/fapi/v1/ticker/price"
        resp = self.session.get(url, params={"symbol": symbol.upper()}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_ticker_24h(self, symbol: str) -> Dict:
        """Get 24-hour stats (volume, change, high/low)."""
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        resp = self.session.get(url, params={"symbol": symbol.upper()}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get funding rate history."""
        url = f"{BASE_URL}/fapi/v1/fundingRate"
        resp = self.session.get(url, params={"symbol": symbol.upper(), "limit": limit}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_exchange_info(self) -> Dict:
        """Get exchange info (filters, limits)."""
        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Private Endpoints (require API key) ───────────────────────

    def _sign(self, params: Dict) -> str:
        if not self.api_secret:
            raise DataError("API secret required for private endpoints")
        query = urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def get_account(self) -> Dict:
        """Get futures account info (margin, positions, balances)."""
        url = f"{BASE_URL}/fapi/v2/account"
        ts = int(time.time() * 1000)
        params = {"timestamp": ts, "recvWindow": 5000}
        params["signature"] = self._sign(params)
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def place_order(
        self,
        symbol: str,
        side: str,  # BUY or SELL
        order_type: str,  # MARKET, LIMIT, STOP_MARKET
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict:
        """Place a futures order."""
        url = f"{BASE_URL}/fapi/v1/order"
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "timestamp": ts,
            "recvWindow": 5000,
        }
        if price:
            params["price"] = price
        if stop_price:
            params["stopPrice"] = stop_price
        if reduce_only:
            params["reduceOnly"] = "true"

        params["signature"] = self._sign(params)
        resp = self.session.post(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders."""
        url = f"{BASE_URL}/fapi/v1/openOrders"
        ts = int(time.time() * 1000)
        params = {"timestamp": ts, "recvWindow": 5000}
        if symbol:
            params["symbol"] = symbol.upper()
        params["signature"] = self._sign(params)
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an order."""
        url = f"{BASE_URL}/fapi/v1/order"
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol.upper(),
            "orderId": order_id,
            "timestamp": ts,
            "recvWindow": 5000,
        }
        params["signature"] = self._sign(params)
        resp = self.session.delete(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
