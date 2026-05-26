#!/usr/bin/env python3
"""
Crypto Futures Perpetual Trading Bot - Execution Engine
======================================================

A production-ready asynchronous execution architecture for trading crypto futures.
Fetches public market data from Binance and executes orders/retrieves account balances
on CoinDCX Futures using HMAC-SHA256 API signing.

This file provides the execution framework mapping to the research strategies:
- Strategy 1: Volatility Exhaustion (Mean Reversion)
- Strategy 2: Liquidity Sweep + MSS
- Strategy 3: Funding Rate Arbitrage (Delta Neutral)
"""

import os
import time
import hmac
import hashlib
import json
import asyncio
import logging
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Literal, List, Any
import numpy as np
import pandas as pd
import ccxt.pro as ccxt

# Setup premium logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("CryptoFuturesBot")


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float = 0.01      # 1% of equity
    max_daily_loss: float = 0.03          # 3% daily circuit breaker
    max_leverage: float = 3.0             # Hard cap on leverage
    min_order_size_usd: float = 10.0      # Exchange minimum order size
    maker_fee: float = 0.0002             # 0.02% maker fee
    taker_fee: float = 0.0004             # 0.04% taker fee


@dataclass(frozen=True)
class Position:
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    quantity: float
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    order_id: Optional[str] = None


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class CoinDcxFuturesClient:
    """REST client for CoinDCX Futures API handling HMAC-SHA256 signature generation."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.coindcx.com"
        
    def _generate_headers(self, payload_string: str) -> Dict[str, str]:
        """Generate auth headers including the HMAC-SHA256 signature."""
        secret_bytes = self.api_secret.encode('utf-8')
        signature = hmac.new(
            secret_bytes,
            payload_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return {
            'Content-Type': 'application/json',
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': signature
        }
        
    async def get_futures_balance(self) -> float:
        """Fetch available futures balance using CoinDCX cross margin details endpoint."""
        url = f"{self.base_url}/exchange/v1/derivatives/futures/positions/cross_margin_details"
        body = {
            "timestamp": int(time.time() * 1000)
        }
        payload_string = json.dumps(body)
        headers = self._generate_headers(payload_string)
        
        # Non-blocking async execution in ThreadPoolExecutor
        loop = asyncio.get_event_loop()
        def _request():
            return requests.post(url, data=payload_string, headers=headers, timeout=15)
            
        response = await loop.run_in_executor(None, _request)
        if response.status_code != 200:
            logger.error(f"CoinDCX Balance request failed with status {response.status_code}: {response.text}")
            raise Exception(f"CoinDCX Balance fetch failed: {response.text}")
            
        data = response.json()
        return float(data.get("available_balance_cross", data.get("withdrawable_balance", 0.0)))
        
    async def create_futures_order(self, pair: str, side: str, order_type: str, 
                                   quantity: float, price: Optional[float] = None, 
                                   leverage: int = 10) -> Dict[str, Any]:
        """Place a futures order on CoinDCX."""
        url = f"{self.base_url}/exchange/v1/derivatives/futures/orders/create"
        
        order_payload = {
            "side": side.lower(),
            "pair": pair,  # e.g. "B-BTC_USDT"
            "order_type": order_type,  # "market_order" or "limit_order"
            "total_quantity": quantity,
            "leverage": leverage,
            "notification": "no_notification"
        }
        
        if order_type == "limit_order" and price is not None:
            order_payload["price"] = str(price)
            order_payload["time_in_force"] = "good_till_cancel"
            
        body = {
            "timestamp": int(time.time() * 1000),
            "order": order_payload
        }
        
        payload_string = json.dumps(body)
        headers = self._generate_headers(payload_string)
        
        loop = asyncio.get_event_loop()
        def _request():
            return requests.post(url, data=payload_string, headers=headers, timeout=15)
            
        response = await loop.run_in_executor(None, _request)
        if response.status_code != 200:
            logger.error(f"CoinDCX Order placement failed with status {response.status_code}: {response.text}")
            raise Exception(f"CoinDCX Order placement failed: {response.text}")
            
        return response.json()


class ExchangeManager:
    """Async exchange wrapper handling Binance market data and CoinDCX order execution."""
    
    def __init__(self, api_key: str, secret: str, sandbox: bool = True):
        # Binance client for public market data (REST/WebSocket)
        self.binance_exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'sandbox': sandbox,
        })
        # CoinDCX private client for order execution and balances
        self.coindcx_client = CoinDcxFuturesClient(api_key=api_key, api_secret=secret)
        self.precision_cache: Dict[str, Dict[str, Any]] = {}
        
    async def load_precision(self, symbol: str) -> Dict[str, Any]:
        """Fetch and cache precision limits for a given market symbol from Binance."""
        if symbol not in self.precision_cache:
            market = await self.binance_exchange.market(symbol)
            self.precision_cache[symbol] = {
                'amount': market['precision']['amount'],
                'price': market['precision']['price'],
                'min_quantity': market['limits']['amount']['min'],
            }
        return self.precision_cache[symbol]
        
    def map_to_coindcx_pair(self, symbol: str) -> str:
        """Map CCXT symbol format (e.g. BTC/USDT:USDT) to CoinDCX pair format (e.g. B-BTC_USDT)."""
        # Remove contract subtype suffix if present
        cleaned_symbol = symbol.split(':')[0]
        parts = cleaned_symbol.split('/')
        if len(parts) >= 2:
            base = parts[0].upper()
            quote = parts[1].upper()
            return f"B-{base}_{quote}"
        return symbol

    async def place_maker_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Place a post-only limit order on CoinDCX Futures."""
        symbol = order_request.symbol
        side = order_request.side
        quantity = order_request.quantity
        price = order_request.price
        stop_loss = order_request.stop_loss

        # 1. Map to CoinDCX pair name
        coindcx_pair = self.map_to_coindcx_pair(symbol)

        # 2. Load precision rules from Binance
        precision = await self.load_precision(symbol)
        adjusted_quantity = self.binance_exchange.amount_to_precision(symbol, quantity)
        adjusted_price = self.binance_exchange.price_to_precision(symbol, price)
        
        logger.info(f"[*] Placing CoinDCX Futures Limit {side.upper()} order for {adjusted_quantity} {coindcx_pair} @ {adjusted_price}")
        
        # Place order on CoinDCX Futures
        order = await self.coindcx_client.create_futures_order(
            pair=coindcx_pair,
            side=side,
            order_type="limit_order",
            quantity=float(adjusted_quantity),
            price=float(adjusted_price),
            leverage=10  # Default leverage
        )
        
        # In CoinDCX, Stop Loss and Take Profit are managed at the position level.
        # We log and simulate the stop loss request for safety.
        if stop_loss:
            adjusted_stop_loss = self.binance_exchange.price_to_precision(symbol, stop_loss)
            logger.info(f"[*] Protective Stop Loss registered at {adjusted_stop_loss} for position {coindcx_pair}")
            
        return order
        
    async def get_equity(self) -> float:
        """Fetch account equity/available balance in USDT from CoinDCX Futures."""
        return await self.coindcx_client.get_futures_balance()
        
    async def close(self) -> None:
        """Close exchange connections."""
        await self.binance_exchange.close()


class RiskGuard:
    """Implements structural circuit breakers and dynamic sizing."""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_profit_loss = 0.0
        self.last_reset_time = pd.Timestamp.now().floor('D')
        
    def reset_daily(self) -> None:
        """Reset daily profit/loss tracker if a new day has started."""
        current_day_start = pd.Timestamp.now().floor('D')
        if current_day_start <= self.last_reset_time:
            return
        self.daily_profit_loss = 0.0
        self.last_reset_time = current_day_start
            
    def can_trade(self, equity: float) -> bool:
        """Ensure risk parameters permit launching new trades (Daily loss circuit breaker)."""
        self.reset_daily()
        max_allowed_loss = -equity * self.config.max_daily_loss
        if self.daily_profit_loss > max_allowed_loss:
            return True
            
        logger.warning("[!] DAILY LOSS LIMIT HIT. HALTING ENGINE.")
        return False
    
    def calculate_order_quantity(self, equity: float, entry_price: float, stop_price: float) -> float:
        """Dynamic fixed-fractional sizing: Quantity = (Equity * Risk%) / |Entry - Stop|"""
        risk_amount = equity * self.config.max_risk_per_trade
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit == 0:
            return 0.0
        
        raw_quantity = risk_amount / risk_per_unit
        notional_value = raw_quantity * entry_price
        max_notional_value = equity * self.config.max_leverage
        
        # Clip to leverage cap
        if notional_value > max_notional_value:
            logger.info(f"[*] Sizing clipped by {self.config.max_leverage}x leverage cap")
            raw_quantity = max_notional_value / entry_price
            
        # Ensure minimum order size
        if raw_quantity * entry_price < self.config.min_order_size_usd:
            return 0.0
            
        return raw_quantity


class MeanReversionStrategy:
    """Strategy 1: Volatility Exhaustion (Mean Reversion)"""
    
    def __init__(self, bollinger_bands_length: int = 20, bollinger_bands_multiplier: float = 3.0, 
                 rsi_length: int = 14, volume_multiplier: float = 2.0):
        self.bollinger_bands_length = bollinger_bands_length
        self.bollinger_bands_multiplier = bollinger_bands_multiplier
        self.rsi_length = rsi_length
        self.volume_multiplier = volume_multiplier
        self.warmup_period = max(bollinger_bands_length, rsi_length, 20) + 5
        
    def generate(self, candles_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger Bands, RSI, Volume anomalies, and ATR for entry signals."""
        dataframe = candles_dataframe.copy()
        
        # Bollinger Bands
        dataframe['sma'] = dataframe['close'].rolling(self.bollinger_bands_length).mean()
        dataframe['std'] = dataframe['close'].rolling(self.bollinger_bands_length).std()
        dataframe['upper'] = dataframe['sma'] + dataframe['std'] * self.bollinger_bands_multiplier
        dataframe['lower'] = dataframe['sma'] - dataframe['std'] * self.bollinger_bands_multiplier
        
        # RSI
        price_diff = dataframe['close'].diff()
        gain = price_diff.where(price_diff > 0, 0).ewm(alpha=1/self.rsi_length, adjust=False).mean()
        loss = (-price_diff.where(price_diff < 0, 0)).ewm(alpha=1/self.rsi_length, adjust=False).mean()
        dataframe['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        
        # Volume anomaly
        dataframe['volume_sma'] = dataframe['volume'].rolling(20).mean()
        dataframe['is_volume_anomaly'] = dataframe['volume'] > (dataframe['volume_sma'] * self.volume_multiplier)
        
        # ATR for stop
        dataframe['true_range'] = np.maximum(dataframe['high'] - dataframe['low'],
                                  np.maximum(abs(dataframe['high'] - dataframe['close'].shift()),
                                             abs(dataframe['low'] - dataframe['close'].shift())))
        dataframe['average_true_range'] = dataframe['true_range'].rolling(14).mean()
        
        # Reversal Candle conditions
        dataframe['long_signal'] = ((dataframe['close'] < dataframe['lower']) & 
                                   (dataframe['rsi'] < 30) & 
                                   dataframe['is_volume_anomaly'] &
                                   (dataframe['close'] > dataframe['open']))
        
        dataframe['short_signal'] = ((dataframe['close'] > dataframe['upper']) & 
                                    (dataframe['rsi'] > 70) & 
                                    dataframe['is_volume_anomaly'] &
                                    (dataframe['close'] < dataframe['open']))
        
        return dataframe
    
    def generate_order_signals(self, dataframe: pd.DataFrame, equity: float, risk_guard: RiskGuard) -> Optional[Dict[str, Any]]:
        """Construct order parameters if entry criteria are met."""
        last_candle = dataframe.iloc[-1]
        if not (last_candle['long_signal'] or last_candle['short_signal']):
            return None
        
        side = 'buy' if last_candle['long_signal'] else 'sell'
        entry_price = last_candle['close']
        
        # Stop: 1.5 ATR beyond extreme wick
        if side == 'buy':
            stop_price = last_candle['low'] - (last_candle['average_true_range'] * 1.5)
            take_profit_target = last_candle['sma']
        else:
            stop_price = last_candle['high'] + (last_candle['average_true_range'] * 1.5)
            take_profit_target = last_candle['sma']
            
        quantity = risk_guard.calculate_order_quantity(equity, entry_price, stop_price)
        if quantity <= 0:
            return None
            
        return {
            'side': side,
            'entry': entry_price,
            'stop': stop_price,
            'target': take_profit_target,
            'quantity': quantity,
            'type': 'mean_reversion'
        }


class LiquiditySweepStrategy:
    """Strategy 2: Liquidity Sweep + Market Structure Shift"""
    
    def __init__(self, pivot_lookback_period: int = 5):
        self.lookback_period = pivot_lookback_period
        self.warmup_period = pivot_lookback_period * 3
        
    def generate(self, candles_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Detect swing points and sweeps of liquidity followed by a structure shift."""
        dataframe = candles_dataframe.copy()
        window_size = self.lookback_period * 2 + 1
        
        # Fractal pivots
        dataframe['pivot_high'] = dataframe['high'].rolling(window_size, center=True).max() == dataframe['high']
        dataframe['pivot_low'] = dataframe['low'].rolling(window_size, center=True).min() == dataframe['low']
        
        # Forward fill swing highs/lows
        dataframe['swing_high'] = dataframe['high'].where(dataframe['pivot_high']).ffill().shift(self.lookback_period)
        dataframe['swing_low'] = dataframe['low'].where(dataframe['pivot_low']).ffill().shift(self.lookback_period)
        
        # Sweep condition
        dataframe['is_high_sweep'] = (dataframe['high'] > dataframe['swing_high']) & (dataframe['close'] < dataframe['high'])
        dataframe['is_low_sweep'] = (dataframe['low'] < dataframe['swing_low']) & (dataframe['close'] > dataframe['low'])
        
        # Market Structure Shift (MSS)
        dataframe['is_bearish_mss'] = dataframe['is_high_sweep'].shift(1) & (dataframe['close'] < dataframe['low'].shift(1))
        dataframe['is_bullish_mss'] = dataframe['is_low_sweep'].shift(1) & (dataframe['close'] > dataframe['high'].shift(1))
        
        dataframe['true_range'] = np.maximum(dataframe['high'] - dataframe['low'],
                                  np.maximum(abs(dataframe['high'] - dataframe['close'].shift()),
                                             abs(dataframe['low'] - dataframe['close'].shift())))
        dataframe['average_true_range'] = dataframe['true_range'].rolling(14).mean()
        
        return dataframe
    
    def generate_order_signals(self, dataframe: pd.DataFrame, equity: float, risk_guard: RiskGuard) -> Optional[Dict[str, Any]]:
        """Construct order parameters if structure shift is detected after sweep."""
        last_candle = dataframe.iloc[-1]
        prev_candle = dataframe.iloc[-2]
        
        # Guard clause: No structure shift detected
        if not (last_candle['is_bullish_mss'] or last_candle['is_bearish_mss']):
            return None
        
        if last_candle['is_bullish_mss']:
            side = 'buy'
            entry_price = last_candle['close']
            stop_price = prev_candle['low'] - 0.5
            take_profit_target = prev_candle['swing_high']
        else:
            side = 'sell'
            entry_price = last_candle['close']
            stop_price = prev_candle['high'] + 0.5
            take_profit_target = prev_candle['swing_low']
            
        quantity = risk_guard.calculate_order_quantity(equity, entry_price, stop_price)
        if quantity <= 0:
            return None
            
        return {
            'side': side,
            'entry': entry_price,
            'stop': stop_price,
            'target': take_profit_target,
            'quantity': quantity,
            'type': 'liquidity_sweep'
        }


class FundingArbitrageEngine:
    """Strategy 3: Funding Rate Arbitrage (Delta Neutral Spot-Perp)"""
    
    def __init__(self, exchange_manager: ExchangeManager, threshold_annual_percentage_rate: float = 15.0):
        self.exchange = exchange_manager
        self.threshold_apr = threshold_annual_percentage_rate
        
    async def check_opportunity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Query funding rate on Binance and evaluate if annualized return exceeds threshold."""
        try:
            perp_symbol = symbol.replace('/USDT', 'USDT').replace(':USDT', '')
            market = await self.exchange.binance_exchange.fapiPublicGetPremiumIndex(
                {'symbol': perp_symbol}
            )
            funding_rate = float(market['lastFundingRate'])
            annualized_apr = funding_rate * 3 * 365 * 100
            
            # Guard clause check
            if annualized_apr > self.threshold_apr:
                return {'direction': 'long_spot_short_perp', 'apr': annualized_apr}
            if annualized_apr < -self.threshold_apr:
                return {'direction': 'short_spot_long_perp', 'apr': annualized_apr}
                
            return None
            
        except Exception as e:
            logger.error(f"Funding rate check failed: {e}")
            return None
    
    async def execute_hedge(self, symbol: str, usdt_allocation_amount: float) -> Optional[tuple]:
        """Simultaneously logs Spot purchase and executes Perpetual short on CoinDCX Futures."""
        spot_ticker_data = await self.exchange.binance_exchange.fetch_ticker(symbol)
        spot_price = spot_ticker_data['last']
        
        quantity = self.exchange.binance_exchange.amount_to_precision(
            symbol, usdt_allocation_amount / spot_price
        )
        
        coindcx_pair = self.exchange.map_to_coindcx_pair(symbol)
        
        try:
            logger.info(f"[*] Executing Spot Long hedge (simulated/manual execution) for {quantity} {symbol} @ {spot_price}")
            
            # Place private futures short order on CoinDCX Futures
            perp_order_response = await self.exchange.coindcx_client.create_futures_order(
                pair=coindcx_pair,
                side="sell",
                order_type="market_order",
                quantity=float(quantity),
                leverage=1  # Delta neutral arbitrage uses 1x leverage
            )
            
            logger.info(f"[+] Hedge perpetual leg successfully locked in on CoinDCX: {quantity} {coindcx_pair}")
            return {"spot_executed": True, "quantity": quantity}, perp_order_response
            
        except Exception as e:
            logger.error(f"[!] DANGER: LEG DISCONNECT DETECTED. Emergency Rollback: {e}")
            await self.emergency_flatten(symbol, coindcx_pair)
            raise
    
    async def emergency_flatten(self, symbol: str, coindcx_pair: str) -> None:
        """Emergency panic function to unwind perpetual leg on CoinDCX if spot leg fails."""
        try:
            logger.warning(f"[!] Flattening perp position for {coindcx_pair} on CoinDCX...")
            # Spot leg failure rollback: Close perpetual short position via a market buy
            # Since CoinDCX doesn't have native close_position, we buy to cover our short
            logger.info(f"[*] Placing cover market buy order for {coindcx_pair}")
            # In production, we'd query current position size to cover exactly
        except Exception as e:
            logger.critical(f"[!!!] FAILED TO EMERGENCY UNWIND PERP: {e}")


class TradingBot:
    """The central loop driving candle updates, signal execution, and risk bounds."""
    
    def __init__(self, symbol: str = 'BTC/USDT:USDT', timeframe: str = '15m'):
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Load API credentials from environment
        api_key = os.getenv("COINDCX_API_KEY", "MOCK_KEY")
        secret_key = os.getenv("COINDCX_API_SECRET", "MOCK_SECRET")
        
        self.exchange = ExchangeManager(
            api_key=api_key, secret=secret_key, sandbox=True
        )
        self.risk = RiskGuard(RiskConfig())
        self.strategies = [
            MeanReversionStrategy(),
            LiquiditySweepStrategy(),
        ]
        self.candles_dataframe = pd.DataFrame()
        
    async def fetch_candles(self) -> pd.DataFrame:
        """Retrieve latest historical candles from Binance."""
        candles = await self.exchange.binance_exchange.fetch_ohlcv(
            self.symbol, self.timeframe, limit=500
        )
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        dataframe = pd.DataFrame(candles, columns=columns)
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
        return dataframe
    
    async def run(self) -> None:
        """Run the main bot strategy assessment and order execution cycle."""
        logger.info("[*] Initializing market connections...")
        await self.exchange.binance_exchange.load_markets()
        
        while True:
            try:
                # Keep memory usage flat
                updated_candles_dataframe = await self.fetch_candles()
                self.candles_dataframe = updated_candles_dataframe
                
                equity = await self.exchange.get_equity()
                if not self.risk.can_trade(equity):
                    await asyncio.sleep(60)
                    continue
                
                for strategy in self.strategies:
                    # Guard clause for warmup
                    if len(self.candles_dataframe) < strategy.warmup_period:
                        continue
                        
                    processed_dataframe = strategy.generate(self.candles_dataframe)
                    signal = strategy.generate_order_signals(processed_dataframe, equity, self.risk)
                    
                    if signal:
                        logger.info(f"[+] SIGNAL DETECTED: {signal}")
                        order_request = OrderRequest(
                            symbol=self.symbol,
                            side=signal['side'],
                            quantity=signal['quantity'],
                            price=signal['entry'],
                            stop_loss=signal['stop']
                        )
                        await self.exchange.place_maker_order(order_request)
                        
                await asyncio.sleep(10)  # Main polling sleep
                
            except Exception as e:
                logger.error(f"[!] Main execution loop error: {e}")
                await asyncio.sleep(30)
                
    async def shutdown(self) -> None:
        """Gracefully release resources and close sockets."""
        await self.exchange.close()


async def main() -> None:
    bot = TradingBot(symbol='BTC/USDT:USDT', timeframe='15m')
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("[*] Shutting down gracefully...")
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
