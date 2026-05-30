"""
Supertrend Live Trading Bot
===========================

Async WebSocket bot that streams Binance Futures klines, computes the
Supertrend indicator on every confirmed closed candle, and executes orders
when a trend flip is detected.

Architecture (three decoupled coroutines):
  _stream_klines()     — WebSocket producer; gates on k["x"]==True (closed candle only)
  _evaluate_signals()  — Consumer; runs signal generator against the rolling buffer
  _execute_orders()    — Order worker; reads from asyncio.Queue, never blocks the WS recv loop

Key guarantees:
  • Anti-repainting: signals are only triggered by k["x"]==True candles
  • Bounded memory: KlineBuffer is a fixed-size deque (default 500 bars)
  • Gap filling: REST candles fetched on reconnect to maintain a continuous buffer
  • Exponential back-off reconnection: 2s → 4s → 8s → 16s

Usage:
    python examples/supertrend_live_bot.py --symbol BTCUSDT --timeframe 15m
    python examples/supertrend_live_bot.py --symbol ETHUSDT --timeframe 1h --length 7 --multiplier 2.5
"""

import argparse
import asyncio
import collections
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

import websockets

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from strategy_builder.domain import Candle
from strategy_builder.market_data.candle_loader import CandleLoader
from strategy_builder.strategies.supertrend import build_signal_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SupertrendBot")

BINANCE_FAPI_WS_BASE = "wss://fstream.binance.com/ws"
MAX_BUFFER_ROWS = 500
MAX_RECONNECT_ATTEMPTS = 20


# ---------------------------------------------------------------------------
# Bounded rolling candle buffer
# ---------------------------------------------------------------------------

class KlineBuffer:
    """
    Bounded rolling buffer of fully-closed OHLCV candles.

    Backed by a deque with `maxlen` so oldest bars are dropped automatically.
    All mutations and reads are guarded by an asyncio.Lock to prevent torn
    reads between the producer and consumer coroutines.
    """

    def __init__(self, maxlen: int = MAX_BUFFER_ROWS) -> None:
        self._buf: Deque[Candle] = collections.deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def push(self, candle: Candle) -> None:
        """Append a closed candle, deduplicating by timestamp."""
        async with self._lock:
            if self._buf and self._buf[-1].timestamp == candle.timestamp:
                # Replace in-place (re-delivered kline for the same bar)
                self._buf[-1] = candle
            else:
                self._buf.append(candle)

    async def fill_gap(self, candles: List[Candle]) -> None:
        """Bulk-push candles (e.g., after a reconnect gap fill)."""
        async with self._lock:
            for c in candles:
                if self._buf and self._buf[-1].timestamp == c.timestamp:
                    self._buf[-1] = c
                else:
                    self._buf.append(c)

    async def to_list(self) -> List[Candle]:
        """Return a snapshot list (oldest→newest) under the lock."""
        async with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Main bot
# ---------------------------------------------------------------------------

class SupertrendLiveBot:
    """
    Async Supertrend live trading bot for Binance Futures.

    Parameters
    ----------
    symbol : str
        Binance futures symbol, e.g. "BTCUSDT".
    timeframe : str
        Kline interval string, e.g. "15m".
    length : int
        Supertrend ATR period.
    multiplier : float
        Supertrend ATR multiplier.
    use_regime_adaptation : bool
        If True, the multiplier is scaled based on the detected market regime.
    max_buffer : int
        Maximum candles retained in the rolling window.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "15m",
        length: int = 10,
        multiplier: float = 3.0,
        use_regime_adaptation: bool = True,
        max_buffer: int = MAX_BUFFER_ROWS,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.length = length
        self.multiplier = multiplier
        self._max_buffer = max_buffer

        self.buffer = KlineBuffer(maxlen=max_buffer)
        self.signal_gen = build_signal_generator(
            length=length,
            multiplier=multiplier,
            adaptive_regime=use_regime_adaptation,
            window_size=max_buffer,
        )
        self.loader = CandleLoader(market_data_source="binance")

        self._signal_event = asyncio.Event()
        self._order_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._current_position = 0  # 1=long, -1=short, 0=flat

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ws_url(self) -> str:
        return f"{BINANCE_FAPI_WS_BASE}/{self.symbol.lower()}@kline_{self.timeframe}"

    def _tf_seconds(self) -> int:
        return CandleLoader.TIMEFRAME_SECONDS.get(self.timeframe, 60)

    async def _fill_history(self) -> None:
        """
        Warm up the rolling buffer with REST history before the stream starts.
        Runs the blocking CandleLoader in a thread-pool executor to avoid
        blocking the asyncio event loop.
        """
        warmup_candles = self._max_buffer
        tf_secs = self._tf_seconds()
        start = datetime.now() - timedelta(seconds=tf_secs * warmup_candles)
        loop = asyncio.get_event_loop()
        try:
            candles = await loop.run_in_executor(
                None,
                lambda: self.loader.fetch(self.symbol, self.timeframe, start, datetime.now()),
            )
            await self.buffer.fill_gap(candles)
            logger.info(f"History warm-up: {len(self.buffer)} candles loaded.")
        except Exception as exc:
            logger.warning(f"History warm-up failed: {exc}. Starting with empty buffer.")

    # ------------------------------------------------------------------
    # WebSocket producer
    # ------------------------------------------------------------------

    async def _stream_klines(self) -> None:
        """
        Connect to the Binance kline stream and push confirmed closed candles
        to the buffer.  Reconnects with exponential back-off on any disconnect.
        """
        url = self._ws_url()
        logger.info(f"Connecting to {url}")

        backoff = 2
        attempts = 0

        while self._running and attempts < MAX_RECONNECT_ATTEMPTS:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    attempts = 0
                    backoff = 2
                    logger.info("WebSocket connected.")

                    async for raw_msg in ws:
                        if not self._running:
                            return

                        try:
                            msg = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue

                        k = msg.get("k", {})

                        # Anti-repainting gate: only process fully closed candles
                        if not k.get("x", False):
                            continue

                        candle = Candle(
                            timestamp=int(k["t"]) // 1000,
                            open=float(k["o"]),
                            high=float(k["h"]),
                            low=float(k["l"]),
                            close=float(k["c"]),
                            volume=float(k["v"]),
                        )
                        await self.buffer.push(candle)
                        self._signal_event.set()
                        logger.debug(
                            f"Closed [{self.timeframe}] {candle.timestamp} "
                            f"O={candle.open} H={candle.high} L={candle.low} C={candle.close}"
                        )

            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException,
                OSError,
            ) as exc:
                attempts += 1
                logger.warning(
                    f"WebSocket error: {exc}. "
                    f"Attempt {attempts}/{MAX_RECONNECT_ATTEMPTS}. "
                    f"Reconnecting in {backoff}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 16)

                # Fill any candles missed during the outage
                logger.info("Gap-filling missed candles after reconnect...")
                await self._fill_history()

        logger.error("Max reconnect attempts reached. Stopping stream.")
        self._running = False
        self._signal_event.set()  # unblock consumer

    # ------------------------------------------------------------------
    # Signal consumer
    # ------------------------------------------------------------------

    async def _evaluate_signals(self) -> None:
        """
        Await the signal event after each closed candle, run the signal generator,
        and enqueue any detected trend flips for the order executor.
        """
        while self._running:
            await self._signal_event.wait()
            self._signal_event.clear()

            if not self._running:
                return

            candles = await self.buffer.to_list()
            if len(candles) < self.length + 2:
                continue

            try:
                signal = self.signal_gen(candles, runtime_mtf=None)
            except Exception as exc:
                logger.error(f"Signal generator error: {exc}")
                continue

            if signal is None:
                continue

            direction = signal["direction"]

            # Filter redundant signals (already holding the same side)
            if direction == "long" and self._current_position > 0:
                continue
            if direction == "short" and self._current_position < 0:
                continue

            logger.info(
                f"[SIGNAL] Supertrend flip → {direction.upper()}  "
                f"entry={signal['entry_price']:.4f}  stop_dist={signal['stop_distance']:.4f}"
            )
            await self._order_queue.put(signal)

    # ------------------------------------------------------------------
    # Order executor (decoupled from WS recv loop)
    # ------------------------------------------------------------------

    async def _execute_orders(self) -> None:
        """
        Consume signals from the order queue and execute trades.

        This coroutine is intentionally isolated from the WebSocket producer
        so that slow order placement (REST API latency) does not cause missed
        WebSocket frames or stream disconnections.

        In this reference implementation orders are logged only.
        Wire in your exchange client here (CoinDCX, ccxt.pro, etc.).
        """
        while self._running:
            try:
                signal = await asyncio.wait_for(self._order_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            direction = signal["direction"]
            entry = signal["entry_price"]
            stop_dist = signal["stop_distance"]
            stop = entry - stop_dist if direction == "long" else entry + stop_dist

            logger.info(
                f"[ORDER] {direction.upper()} {self.symbol}  "
                f"entry={entry:.4f}  stop={stop:.4f}  dist={stop_dist:.4f}"
            )

            # Update position tracking
            self._current_position = 1 if direction == "long" else -1

            # ----------------------------------------------------------------
            # TODO: replace with actual order placement
            # Example (ccxt.pro):
            #   order = await exchange.create_market_order(
            #       symbol=self.symbol, side=direction, amount=quantity
            #   )
            # Example (CoinDCX REST via asyncio executor):
            #   loop = asyncio.get_event_loop()
            #   order = await loop.run_in_executor(None, coindcx_client.place_order, ...)
            # ----------------------------------------------------------------

            self._order_queue.task_done()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all coroutines. Warm up the buffer first, then run concurrently."""
        self._running = True
        logger.info(
            f"SupertrendBot starting: {self.symbol} {self.timeframe}  "
            f"length={self.length}  multiplier={self.multiplier}"
        )
        await self._fill_history()

        await asyncio.gather(
            self._stream_klines(),
            self._evaluate_signals(),
            self._execute_orders(),
        )

    async def stop(self) -> None:
        """Gracefully stop all coroutines."""
        self._running = False
        self._signal_event.set()
        logger.info("Bot stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main() -> None:
    parser = argparse.ArgumentParser(description="Supertrend live trading bot")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance futures symbol")
    parser.add_argument("--timeframe", default="15m", help="Kline interval (e.g. 5m, 1h)")
    parser.add_argument("--length", type=int, default=10, help="Supertrend ATR period")
    parser.add_argument("--multiplier", type=float, default=3.0, help="Supertrend ATR multiplier")
    parser.add_argument(
        "--no-regime",
        action="store_true",
        default=False,
        help="Disable regime-adaptive multiplier scaling",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=MAX_BUFFER_ROWS,
        help="Max candles to retain in the rolling buffer",
    )
    args = parser.parse_args()

    bot = SupertrendLiveBot(
        symbol=args.symbol,
        timeframe=args.timeframe,
        length=args.length,
        multiplier=args.multiplier,
        use_regime_adaptation=not args.no_regime,
        max_buffer=args.buffer,
    )
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
