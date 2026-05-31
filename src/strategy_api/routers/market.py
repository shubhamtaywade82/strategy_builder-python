from __future__ import annotations

from fastapi import APIRouter, Query
from strategy_api.services import binance

router = APIRouter(prefix="/api/market", tags=["market"])
_SYMBOL = r"^[A-Z0-9]{3,12}$"


@router.get("/mark-price")
async def mark_price(symbol: str = Query(..., pattern=_SYMBOL)):
    return await binance.mark_price(symbol)


@router.get("/ticker-24h")
async def ticker_24h(symbol: str = Query(..., pattern=_SYMBOL)):
    return await binance.ticker_24h(symbol)


@router.get("/klines")
async def klines(
    symbol: str = Query(..., pattern=_SYMBOL),
    interval: str = Query(..., pattern=r"^(1m|3m|5m|15m|30m|1h|4h|1d)$"),
    limit: int = Query(100, ge=1, le=1500),
):
    return await binance.klines(symbol, interval, limit)


@router.get("/top-symbols")
async def top_symbols():
    return await binance.top_symbols()
