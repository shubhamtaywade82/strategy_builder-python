import time
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from ..domain import Candle
from ..exceptions import DataError

class CandleLoader:
    TIMEFRAME_SECONDS = {
        '1m': 60, '3m': 180, '5m': 300, '15m': 900,
        '30m': 1800, '1h': 3600, '2h': 7200, '4h': 14400,
        '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400, 
        '3d': 259200, '1w': 604800, '1M': 2592000
    }

    RESOLUTION_MAP = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15',
        '30m': '30', '1h': '60', '2h': '120', '4h': '240',
        '6h': '360', '8h': '480', '12h': '720', '1d': '1D', 
        '3d': '3D', '1w': '1W', '1M': '1M'
    }

    MAX_CANDLES_PER_REQUEST = 1500
    MAX_UNIX_SECONDS_SANE = 10_000_000_000

    def __init__(self, client=None, market_data_source: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.client = client
        self.logger = logger or logging.getLogger(__name__)
        self.market_data_source = market_data_source or "auto"

    def fetch(self, instrument: str, timeframe: str, start_from: Union[int, datetime], to: Optional[Union[int, datetime]] = None) -> List[Candle]:
        to = to or datetime.now()
        source = self.market_data_source.lower()

        if source == "binance":
            return self.fetch_binance(instrument, timeframe, start_from, to)
        elif source == "coindcx":
            return self.fetch_coindcx(instrument, timeframe, start_from, to)
        else: # auto
            try:
                return self.fetch_binance(instrument, timeframe, start_from, to)
            except Exception as e:
                self.logger.warning(f"Binance candle fetch failed for {instrument} ({type(e).__name__}: {e}). Falling back to CoinDCX.")
                return self.fetch_coindcx(instrument, timeframe, start_from, to)

    def fetch_open_interest(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {
            "symbol": self.map_to_binance_symbol(symbol),
            "period": interval,
            "limit": limit
        }
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to fetch OI for {symbol}: {response.text}")
            return []

    def fetch_mtf(self, instrument: str, timeframes: List[str], start_from: Union[int, datetime], to: Optional[Union[int, datetime]] = None) -> Dict[str, List[Candle]]:
        to = to or datetime.now()
        result = {}
        for tf in timeframes:
            result[tf] = self.fetch(instrument, tf, start_from, to)
            self.logger.info(f"Loaded {len(result[tf])} candles for {instrument} {tf}")
        return result

    def fetch_coindcx(self, instrument: str, timeframe: str, start_from: Union[int, datetime], to: Union[int, datetime]) -> List[Candle]:
        resolution = self.RESOLUTION_MAP.get(timeframe)
        if not resolution:
            raise DataError(f"Unknown timeframe: {timeframe}")

        from_ts = int(start_from.timestamp()) if isinstance(start_from, datetime) else int(start_from)
        to_ts = int(to.timestamp()) if isinstance(to, datetime) else int(to)
        tf_secs = self.TIMEFRAME_SECONDS[timeframe]

        all_candles = []
        cursor = from_ts

        while cursor < to_ts:
            window_end = min(cursor + (self.MAX_CANDLES_PER_REQUEST * tf_secs), to_ts)
            self.logger.debug(f"Fetching CoinDCX {instrument} {timeframe} from {datetime.fromtimestamp(cursor)} to {datetime.fromtimestamp(window_end)}")

            raw = self.fetch_batch(instrument, resolution, cursor, window_end)
            if not raw:
                cursor = window_end
                continue

            normalized = [self.normalize_candle(c, timeframe) for c in raw]
            normalized.sort(key=lambda x: x.timestamp)
            all_candles.extend(normalized)

            if not normalized:
                cursor = window_end
            else:
                last_ts = normalized[-1].timestamp
                cursor = max(last_ts + tf_secs, window_end)

        return self.deduplicate_and_sort(all_candles)

    def fetch_binance(self, instrument: str, timeframe: str, start_from: Union[int, datetime], to: Union[int, datetime]) -> List[Candle]:
        binance_symbol = self.map_to_binance_symbol(instrument)
        from_ts = int(start_from.timestamp()) if isinstance(start_from, datetime) else int(start_from)
        to_ts = int(to.timestamp()) if isinstance(to, datetime) else int(to)
        tf_secs = self.TIMEFRAME_SECONDS[timeframe]
        tf_ms = tf_secs * 1000

        from_ms = from_ts * 1000
        to_ms = to_ts * 1000

        all_candles = []
        cursor_ms = from_ms

        while cursor_ms < to_ms:
            self.logger.debug(f"Fetching Binance {binance_symbol} {timeframe} from {datetime.fromtimestamp(cursor_ms / 1000)} to {datetime.fromtimestamp(to_ms / 1000)}")

            raw = self.fetch_binance_batch(binance_symbol, timeframe, cursor_ms, to_ms, self.MAX_CANDLES_PER_REQUEST)
            if not raw:
                break

            normalized = [self.normalize_binance_candle(c, timeframe) for c in raw]
            normalized.sort(key=lambda x: x.timestamp)
            all_candles.extend(normalized)

            last_ts = normalized[-1].timestamp
            last_ms = last_ts * 1000

            next_cursor_ms = last_ms + tf_ms
            if next_cursor_ms <= cursor_ms:
                next_cursor_ms = cursor_ms + tf_ms
            cursor_ms = next_cursor_ms

        return self.deduplicate_and_sort(all_candles)

    def fetch_binance_batch(self, symbol: str, interval: str, startTime: int, endTime: int, limit: int) -> List[List[Any]]:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": startTime,
            "endTime": endTime,
            "limit": limit
        }
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            raise DataError(f"Binance API returned HTTP {response.status_code}: {response.text}")

    def map_to_binance_symbol(self, instrument: str) -> str:
        symbol = instrument.upper()
        if symbol.startswith("B-"):
            symbol = symbol[2:]
        return symbol.replace("_", "")

    def normalize_binance_candle(self, raw: List[Any], timeframe: str) -> Candle:
        return Candle(
            timestamp=int(raw[0] / 1000),
            open=float(raw[1]),
            high=float(raw[2]),
            low=float(raw[3]),
            close=float(raw[4]),
            volume=float(raw[5])
        )

    def fetch_batch(self, instrument: str, resolution: str, start_from: int, to: int) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        
        try:
            # Assuming client has a similar interface or we adapt it
            # For now, this is a placeholder for the coindcx-client call
            raw = self.client.futures.market_data.list_candlesticks(
                pair=instrument,
                from_ts=start_from,
                to_ts=to,
                resolution=resolution
            )
            return self.coerce_candlestick_rows(raw)
        except Exception as e:
            self.logger.error(f"Error fetching candles from CoinDCX: {e}")
            return []

    def coerce_candlestick_rows(self, raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            for key in ['data', 'candles', 'candlesticks']:
                if key in raw:
                    return raw[key]
        return []

    def normalize_candle(self, raw: Dict[str, Any], timeframe: str) -> Candle:
        return Candle(
            timestamp=self.normalize_candle_timestamp(raw.get('time') or raw.get('t')),
            open=float(raw.get('open') or raw.get('o')),
            high=float(raw.get('high') or raw.get('h')),
            low=float(raw.get('low') or raw.get('l')),
            close=float(raw.get('close') or raw.get('c')),
            volume=float(raw.get('volume') or raw.get('v'))
        )

    def normalize_candle_timestamp(self, value: Any) -> int:
        if isinstance(value, datetime):
            return int(value.timestamp())
        
        try:
            i = int(value)
            for _ in range(3):
                if i <= self.MAX_UNIX_SECONDS_SANE:
                    break
                i //= 1000
            return i
        except (ValueError, TypeError):
            raise DataError(f"Invalid candle timestamp: {value}")

    def deduplicate_and_sort(self, candles: List[Candle]) -> List[Candle]:
        seen = set()
        unique = []
        for c in candles:
            if c.timestamp not in seen:
                unique.append(c)
                seen.add(c.timestamp)
        unique.sort(key=lambda x: x.timestamp)
        return unique
