from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from ..domain import Candle

class SessionDetector:
    # Session windows in UTC.
    SESSIONS = {
        'asia': {'start_hour': 0, 'end_hour': 8},
        'london': {'start_hour': 7, 'end_hour': 16},
        'new_york': {'start_hour': 13, 'end_hour': 22},
        'asia_london': {'start_hour': 7, 'end_hour': 8}, # overlap
        'london_ny': {'start_hour': 13, 'end_hour': 16}, # overlap
        'off_hours': {'start_hour': 22, 'end_hour': 0}
    }

    @staticmethod
    def tag_candles(candles: List[Candle]) -> List[Dict[str, Any]]:
        results = []
        for candle in candles:
            ts = candle.timestamp
            if isinstance(ts, int):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            
            hour = dt.hour
            sessions = []
            for name, window in SessionDetector.SESSIONS.items():
                start = window['start_hour']
                end = window['end_hour']
                if start < end:
                    if start <= hour < end:
                        sessions.append(name)
                else: # Overnight sessions (e.g. 22 to 0)
                    if hour >= start or hour < end:
                        sessions.append(name)
            
            # Create a dict with candle data + sessions
            results.append({
                **candle.model_dump(),
                'sessions': sessions
            })
        return results

    @staticmethod
    def detect_sessions(candles: List[Candle]) -> List[str]:
        if not candles:
            return []
        tagged = SessionDetector.tag_candles(candles)
        sessions = set()
        for c in tagged:
            sessions.update(c['sessions'])
        return sorted(list(sessions))

    @staticmethod
    def group_by_session(candles: List[Candle]) -> Dict[str, List[Dict[str, Any]]]:
        tagged = SessionDetector.tag_candles(candles)
        result = {name: [] for name in SessionDetector.SESSIONS.keys()}
        for c in tagged:
            for s in c['sessions']:
                result[s].append(c)
        return result

    @staticmethod
    def session_range(candles: List[Candle], session: str, date: datetime) -> Optional[Dict[str, Any]]:
        tagged = SessionDetector.tag_candles(candles)
        day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc).timestamp()
        day_end = day_start + 86400

        session_candles = [
            c for c in tagged
            if day_start <= c['timestamp'] < day_end and session in c['sessions']
        ]

        if not session_candles:
            return None

        return {
            "session": session,
            "date": date,
            "high": max(c['high'] for c in session_candles),
            "low": min(c['low'] for c in session_candles),
            "open": session_candles[0]['open'],
            "close": session_candles[-1]['close'],
            "volume": sum(c['volume'] for c in session_candles),
            "candle_count": len(session_candles)
        }

    @staticmethod
    def utc_day_bounds(reference_candle: Candle) -> Tuple[int, int]:
        ts = reference_candle.timestamp
        if isinstance(ts, int):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        
        day_start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
        start_ts = int(day_start.timestamp())
        return start_ts, start_ts + 86400

    @staticmethod
    def candles_for_session_on_day(candles: List[Candle], session: str, reference_candle: Candle) -> List[Dict[str, Any]]:
        if not candles:
            return []
        start_i, end_i = SessionDetector.utc_day_bounds(reference_candle)
        tagged = SessionDetector.tag_candles(candles)
        return [
            c for c in tagged
            if start_i <= c['timestamp'] < end_i and session in c['sessions']
        ]

    @staticmethod
    def session_box(candles: List[Candle], session_name: str, completed_only: bool = False, min_candles: int = 1) -> Optional[Dict[str, Any]]:
        if not candles:
            return None

        tagged = SessionDetector.tag_candles(candles)
        session_candles = []
        in_session_block = False
        found_non_session = False

        # Work backwards
        for c in reversed(tagged):
            in_session = session_name in c['sessions']

            if completed_only and not found_non_session:
                if not in_session:
                    found_non_session = True
                continue

            if in_session:
                in_session_block = True
                session_candles.insert(0, c)
            elif in_session_block:
                break
        
        if len(session_candles) < min_candles:
            return None

        high = max(c['high'] for c in session_candles)
        low = min(c['low'] for c in session_candles)
        range_val = high - low
        if range_val <= 1e-12:
            return None

        return {
            "high": high,
            "low": low,
            "candle_count": len(session_candles),
            "range": range_val
        }
