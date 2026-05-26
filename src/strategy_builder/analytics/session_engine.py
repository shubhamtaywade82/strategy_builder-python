from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from ..domain import Candle, Session, Regime

class SessionAnalyticsEngine:
    """
    Institutional Session Analytics Engine.
    Analyzes RVOL, volatility, and trend efficiency across synthetic trading sessions.
    """
    
    SESSIONS = {
        Session.ASIA: {"start": 0, "end": 8},
        Session.LONDON: {"start": 7, "end": 13},
        Session.NY_OPEN: {"start": 13, "end": 17},
        Session.NY_AFTERNOON: {"start": 17, "end": 22},
        Session.LATE_NY: {"start": 22, "end": 24}
    }

    def __init__(self):
        pass

    def analyze(self, candles: List[Candle]) -> Dict[str, Any]:
        if not candles:
            return {}

        df = self._to_dataframe(candles)
        df = self._prepare_features(df)
        
        session_metrics = self._calculate_session_metrics(df)
        current_session = self._get_current_session(df.iloc[-1]["timestamp"])
        
        return {
            "current_session": current_session.value,
            "session_metrics": session_metrics,
            "rvol": float(df.iloc[-1]["rvol"]),
            "trend_efficiency": float(df.iloc[-1]["directional_efficiency"]),
            "true_range_pct": float(df.iloc[-1]["true_range_pct"])
        }

    def _to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        data = [
            {
                "timestamp": c.timestamp if isinstance(c.timestamp, datetime) else datetime.fromtimestamp(c.timestamp, tz=timezone.utc),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume
            }
            for c in candles
        ]
        df = pd.DataFrame(data)
        df["hour"] = df["timestamp"].dt.hour
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # RVOL Calculation (Relative to same hour mean)
        avg_volume_by_hour = df.groupby("hour")["volume"].transform("mean")
        df["rvol"] = df["volume"] / avg_volume_by_hour.replace(0, np.nan)
        df["rvol"] = df["rvol"].fillna(1.0)

        # True Range %
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - prev_close)
        tr3 = abs(df["low"] - prev_close)
        df["true_range"] = np.maximum.reduce([tr1, tr2, tr3])
        df["true_range_pct"] = (df["true_range"] / df["close"]) * 100

        # Directional Efficiency
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["directional_efficiency"] = abs(df["close"] - df["open"]) / candle_range
        df["directional_efficiency"] = df["directional_efficiency"].fillna(0)

        return df

    def _calculate_session_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        metrics = {}
        for session, bounds in self.SESSIONS.items():
            mask = (df["hour"] >= bounds["start"]) & (df["hour"] < bounds["end"])
            session_df = df[mask]
            
            if session_df.empty:
                continue
                
            metrics[session.value] = {
                "avg_rvol": float(session_df["rvol"].mean()),
                "avg_volatility": float(session_df["true_range_pct"].mean()),
                "avg_efficiency": float(session_df["directional_efficiency"].mean()),
                "volume_sum": float(session_df["volume"].sum())
            }
        return metrics

    def _get_current_session(self, ts: datetime) -> Session:
        hour = ts.hour
        
        # Check for overlaps first
        if 7 <= hour < 8:
            return Session.ASIA_LONDON_OVERLAP
        if 13 <= hour < 14:
            return Session.LONDON_NY_OVERLAP
            
        for session, bounds in self.SESSIONS.items():
            if bounds["start"] <= hour < bounds["end"]:
                return session
        
        return Session.UNKNOWN
