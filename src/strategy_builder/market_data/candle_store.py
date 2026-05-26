from typing import List, Dict, Any, Optional
from ..domain import Candle

class CandleStore:
    def __init__(self):
        self._store = {}

    def _key(self, instrument: str, timeframe: str) -> str:
        return f"{instrument}:{timeframe}"

    def put(self, instrument: str, timeframe: str, candles: List[Candle]):
        k = self._key(instrument, timeframe)
        existing = self._store.get(k, [])
        
        # Merge, deduplicate by timestamp, and sort
        merged = existing + candles
        seen = set()
        unique = []
        for c in merged:
            ts = c.timestamp if isinstance(c.timestamp, int) else int(c.timestamp.timestamp())
            if ts not in seen:
                unique.append(c)
                seen.add(ts)
        
        unique.sort(key=lambda x: x.timestamp if isinstance(x.timestamp, int) else x.timestamp.timestamp())
        self._store[k] = unique

    def get(self, instrument: str, timeframe: str, start_from: Optional[int] = None, to: Optional[int] = None) -> List[Candle]:
        k = self._key(instrument, timeframe)
        candles = self._store.get(k, [])
        
        if start_from:
            candles = [c for c in candles if (c.timestamp if isinstance(c.timestamp, int) else int(c.timestamp.timestamp())) >= start_from]
        if to:
            candles = [c for c in candles if (c.timestamp if isinstance(c.timestamp, int) else int(c.timestamp.timestamp())) <= to]
            
        return candles

    def instruments(self) -> List[str]:
        return list(set(k.split(":")[0] for k in self._store.keys()))

    def size(self) -> int:
        return sum(len(c) for c in self._store.values())

    def clear(self):
        self._store.clear()
