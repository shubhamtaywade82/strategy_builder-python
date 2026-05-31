"""
Binance USD-M Futures WebSocket Client
Drops into: src/strategy_builder/market_data/binance_websocket_client.py

Three stream classes:
    BinanceKlineStream      – single symbol + interval
    BinanceCombinedStream   – multiple symbols/intervals on one connection
    BinanceMiniTickerStream – low-latency price + 24h stats (~1s updates)
"""
import json
import threading
import time
from typing import Callable, Optional, List, Tuple

try:
    import websocket
except ImportError:
    raise ImportError("Install websocket-client: pip install websocket-client")

from ..domain import Candle

WS_BASE = "wss://fstream.binance.com"


class BinanceKlineStream:
    """WebSocket stream for a single symbol + interval kline.

    Emits closed Candle objects only. Auto-reconnects on disconnect.
    """

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        on_candle: Optional[Callable[[Candle], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        reconnect_delay: float = 5.0,
    ):
        self.symbol = symbol.upper()
        self.interval = interval
        self.on_candle = on_candle
        self.on_error = on_error
        self.on_connect = on_connect
        self.on_close = on_close
        self.reconnect_delay = reconnect_delay

        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.last_close_time = 0

    def _on_open(self, ws):
        if self.on_connect:
            self.on_connect()

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
            k = data.get("k", {})
            if not k or not k.get("x", False):
                return

            close_time = k.get("T", 0)
            if close_time <= self.last_close_time:
                return
            self.last_close_time = close_time

            candle = Candle(
                timestamp=int(k["t"]),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
            )
            if self.on_candle:
                self.on_candle(candle)
        except Exception as e:
            if self.on_error:
                self.on_error(f"Parse error: {e}")

    def _on_error(self, ws, error):
        if self.on_error:
            self.on_error(str(error))

    def _on_close(self, ws, close_status, close_msg):
        if self.on_close:
            self.on_close()
        if self.running:
            time.sleep(self.reconnect_delay)
            self._connect()

    def _connect(self):
        stream = f"{self.symbol.lower()}@kline_{self.interval}"
        url = f"{WS_BASE}/ws/{stream}"
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

    def start(self):
        self.running = True
        self._connect()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class BinanceCombinedStream:
    """Combined WebSocket stream for multiple symbols + intervals.

    Uses Binance's combined stream endpoint — all data on one connection.
    Callback: on_candle(symbol, interval, candle)
    """

    def __init__(
        self,
        streams: List[Tuple[str, str]],  # [("SOLUSDT", "1m"), ("BTCUSDT", "1h")]
        on_candle: Optional[Callable[[str, str, Candle], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        reconnect_delay: float = 5.0,
    ):
        self.streams = streams
        self.on_candle = on_candle
        self.on_error = on_error
        self.on_connect = on_connect
        self.reconnect_delay = reconnect_delay

        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.last_close_times: dict = {}

    def _on_open(self, ws):
        if self.on_connect:
            self.on_connect()

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
            payload = data.get("data", {})
            k = payload.get("k", {})
            if not k or not k.get("x", False):
                return

            symbol = k.get("s", "").upper()
            interval = k.get("i", "")
            key = (symbol, interval)
            close_time = k.get("T", 0)

            if close_time <= self.last_close_times.get(key, 0):
                return
            self.last_close_times[key] = close_time

            candle = Candle(
                timestamp=int(k["t"]),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
            )
            if self.on_candle:
                self.on_candle(symbol, interval, candle)
        except Exception as e:
            if self.on_error:
                self.on_error(f"Parse error: {e}")

    def _on_error(self, ws, error):
        if self.on_error:
            self.on_error(str(error))

    def _on_close(self, ws, close_status, close_msg):
        if self.running:
            time.sleep(self.reconnect_delay)
            self._connect()

    def _connect(self):
        stream_names = [f"{s.lower()}@kline_{i}" for s, i in self.streams]
        url = f"{WS_BASE}/stream?streams={'/'.join(stream_names)}"
        self.ws = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

    def start(self):
        self.running = True
        self._connect()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class BinanceMiniTickerStream:
    """Real-time price + 24h stats for a symbol (~1s updates).

    Use for live position tracking, stop/target monitoring.
    Callback: on_tick(data_dict)
    """

    def __init__(
        self,
        symbol: str,
        on_tick: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        reconnect_delay: float = 5.0,
    ):
        self.symbol = symbol.upper()
        self.on_tick = on_tick
        self.on_error = on_error
        self.reconnect_delay = reconnect_delay

        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.current_price = 0.0
        self.price_change_24h = 0.0

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
            self.current_price = float(data.get("c", 0))
            self.price_change_24h = float(data.get("P", 0))
            if self.on_tick:
                self.on_tick(data)
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))

    def _on_error(self, ws, error):
        if self.on_error:
            self.on_error(str(error))

    def _on_close(self, ws, close_status, close_msg):
        if self.running:
            time.sleep(self.reconnect_delay)
            self._connect()

    def _connect(self):
        url = f"{WS_BASE}/ws/{self.symbol.lower()}@miniTicker"
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

    def start(self):
        self.running = True
        self._connect()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=5)

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def get_price(self) -> float:
        return self.current_price

    def get_change_24h_pct(self) -> float:
        return self.price_change_24h
