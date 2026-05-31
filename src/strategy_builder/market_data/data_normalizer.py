import time
from typing import List, Dict, Any, Optional
from ..features.session_detector import SessionDetector

class DataNormalizer:
    @staticmethod
    def normalize(instrument: str, mtf_candles: Dict[str, List[Any]], sessions: Optional[List[str]] = None) -> Dict[str, Any]:
        if sessions is None:
            first_series = next(iter(mtf_candles.values())) if mtf_candles else []
            sessions = SessionDetector.detect_sessions(first_series)

        return {
            "instrument": instrument,
            "timeframes": list(mtf_candles.keys()),
            "candles": mtf_candles,
            "sessions": sessions,
            "fetched_at": int(time.time()),
            "candle_counts": {tf: len(c) for tf, c in mtf_candles.items()}
        }
