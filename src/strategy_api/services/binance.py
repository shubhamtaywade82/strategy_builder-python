from __future__ import annotations

from typing import List, Dict, Any

import httpx
from strategy_api.config import settings

_TIMEOUT = httpx.Timeout(8.0)


async def _get(path: str) -> Any:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
        r = await c.get(f"{settings.binance_fapi}{path}", headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()


async def mark_price(symbol: str) -> Dict[str, Any]:
    d = await _get(f"/fapi/v1/premiumIndex?symbol={symbol}")
    return {
        "symbol": d["symbol"],
        "markPrice": float(d["markPrice"]),
        "indexPrice": float(d["indexPrice"]),
        "fundingRate": float(d["lastFundingRate"]),
        "nextFundingTime": d["nextFundingTime"],
        "time": d["time"],
    }


async def ticker_24h(symbol: str) -> Dict[str, Any]:
    d = await _get(f"/fapi/v1/ticker/24hr?symbol={symbol}")
    return {
        "symbol": d["symbol"],
        "priceChange": float(d["priceChange"]),
        "priceChangePct": float(d["priceChangePercent"]),
        "high24h": float(d["highPrice"]),
        "low24h": float(d["lowPrice"]),
        "volume24h": float(d["volume"]),
        "quoteVolume24h": float(d["quoteVolume"]),
        "lastPrice": float(d["lastPrice"]),
        "openPrice": float(d["openPrice"]),
        "count": d["count"],
    }


async def klines(symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
    rows = await _get(f"/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}")
    return [{
        "openTime": r[0],
        "open": float(r[1]),
        "high": float(r[2]),
        "low": float(r[3]),
        "close": float(r[4]),
        "volume": float(r[5]),
        "closeTime": r[6],
        "quoteVolume": float(r[7]),
        "nTrades": r[8],
        "takerBuyBase": float(r[9]),
        "takerBuyQuote": float(r[10]),
    } for r in rows]


async def top_symbols() -> List[Dict[str, Any]]:
    rows = await _get("/fapi/v1/ticker/24hr")
    usdt = [d for d in rows if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 0]
    usdt.sort(key=lambda d: float(d["quoteVolume"]), reverse=True)
    return [{
        "symbol": d["symbol"],
        "lastPrice": float(d["lastPrice"]),
        "priceChangePct": float(d["priceChangePercent"]),
        "quoteVolume": float(d["quoteVolume"]),
    } for d in usdt[:30]]
