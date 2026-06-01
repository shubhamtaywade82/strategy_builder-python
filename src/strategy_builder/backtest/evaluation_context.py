from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from ..domain import Candle
from ..features.volatility_profile import VolatilityProfile
from ..features.momentum_engine import MomentumEngine
from ..features.structure_detector import StructureDetector
from ..features.volume_profile import VolumeProfile
from ..features.session_detector import SessionDetector
from ..analytics.session_engine import SessionAnalyticsEngine
from ..state.regime_classifier import RegimeClassifier

import logging

class EvaluationContext:
    def __init__(self, candles: List[Candle], strategy: Dict[str, Any], mtf_candles: Optional[Dict[str, List[Candle]]] = None):
        self.logger = logging.getLogger("EvaluationContext")
        self.candles = [Candle(**c) if isinstance(c, dict) else c for c in candles]
        self.strategy = strategy
        if mtf_candles:
            self.mtf_candles = {
                tf: [Candle(**c) if isinstance(c, dict) else c for c in tf_candles]
                for tf, tf_candles in mtf_candles.items()
            }
        else:
            self.mtf_candles = None
        self.index = len(candles) - 1
        self.current_candle = candles[-1] if candles else None
        self.previous_candle = candles[-2] if len(candles) >= 2 else None

        self.direction = None
        self.entry_price = None
        self.stop_distance = None
        self.size = 1.0

        self._memo = {}
        self._session_engine = SessionAnalyticsEngine()

    def atr(self, period: int = 14) -> float:
        key = f"atr_{period}"
        if key not in self._memo:
            vals = VolatilityProfile.atr(self.candles, period=period)
            compact = [v for v in vals if v is not None]
            self._memo[key] = compact[-1] if compact else 0.0
        return self._memo[key]

    def atr_percent(self, period: int = 14) -> float:
        key = f"atr_pct_{period}"
        if key not in self._memo:
            vals = VolatilityProfile.atr_percent(self.candles, period=period)
            compact = [v for v in vals if v is not None]
            self._memo[key] = compact[-1] if compact else 0.0
        return self._memo[key]

    def rsi(self, period: int = 14) -> Optional[float]:
        key = f"rsi_{period}"
        if key not in self._memo:
            vals = MomentumEngine.rsi(self.candles, period=period)
            compact = [v for v in vals if v is not None]
            self._memo[key] = compact[-1] if compact else None
        return self._memo[key]

    def vwap(self) -> List[float]:
        if "vwap" not in self._memo:
            self._memo["vwap"] = VolumeProfile.vwap(self.candles)
        return self._memo["vwap"]

    def swing_points(self, lookback: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        key = f"swing_points_{lookback}"
        if key not in self._memo:
            self._memo[key] = StructureDetector.swing_points(self.candles, lookback=lookback)
        return self._memo[key]

    def structure(self, lookback: int = 5) -> str:
        key = f"structure_{lookback}"
        if key not in self._memo:
            self._memo[key] = StructureDetector.structure(self.candles, lookback=lookback)
        return self._memo[key]

    def higher_tf_structure(self) -> str:
        if "higher_tf_structure" not in self._memo:
            self._memo["higher_tf_structure"] = self._compute_higher_tf_structure()
        return self._memo["higher_tf_structure"]

    def regime(self) -> str:
        if "regime" not in self._memo:
            analytics = self.session_analytics()
            features = {
                "rvol": analytics.get("rvol", 1.0),
                "volatility": {
                    "current_atr_percent": self.atr_percent(),
                    "regime": VolatilityProfile.regime(self.candles)
                },
                "trend_efficiency": self.trend_efficiency(),
                "structure": {"structure": self.structure()},
                "mtf_alignment": {"alignment": {"aligned_bullish": self.higher_tf_structure() == "bullish", "aligned_bearish": self.higher_tf_structure() == "bearish"}}
            }
            regime = RegimeClassifier.classify(features)
            self._memo["regime"] = regime.value
            if self.index % 1000 == 0:
                self.logger.debug(f"Index {self.index}: Regime={regime.value}, RVOL={features['rvol']:.2f}, Eff={features['trend_efficiency']:.2f}")
        return self._memo["regime"]

    def trend_efficiency(self) -> float:
        if "trend_efficiency" not in self._memo:
            analytics = self._session_engine.analyze(self.candles)
            self._memo["trend_efficiency"] = analytics.get("trend_efficiency", 0.5)
        return self._memo["trend_efficiency"]

    def session_analytics(self) -> Dict[str, Any]:
        if "session_analytics" not in self._memo:
            self._memo["session_analytics"] = self._session_engine.analyze(self.candles)
        return self._memo["session_analytics"]

    def volume_zscore(self, lookback: int = 20) -> float:
        key = f"volume_zscore_{lookback}"
        if key not in self._memo:
            vals = VolumeProfile.volume_zscore(self.candles, lookback=lookback)
            compact = [v for v in vals if v is not None]
            self._memo[key] = compact[-1] if compact else 0.0
        return self._memo[key]

    def current_sessions(self) -> List[str]:
        if "current_sessions" not in self._memo:
            if not self.current_candle:
                self._memo["current_sessions"] = []
            else:
                tagged = SessionDetector.tag_candles([self.current_candle])
                self._memo["current_sessions"] = tagged[0].get("sessions", [])
        return self._memo["current_sessions"]

    def asia_box(self) -> Optional[Dict[str, Any]]:
        if "asia_box" not in self._memo:
            self._memo["asia_box"] = SessionDetector.session_box(self.candles, "asia")
        return self._memo["asia_box"]

    def london_box(self) -> Optional[Dict[str, Any]]:
        if "london_box" not in self._memo:
            self._memo["london_box"] = SessionDetector.session_box(self.candles, "london")
        return self._memo["london_box"]

    def ny_box(self) -> Optional[Dict[str, Any]]:
        if "ny_box" not in self._memo:
            self._memo["ny_box"] = SessionDetector.session_box(self.candles, "new_york")
        return self._memo["ny_box"]

    def ema(self, period: int = 20) -> List[Optional[float]]:
        key = f"ema_{period}"
        if key not in self._memo:
            closes = [c.close for c in self.candles]
            self._memo[key] = MomentumEngine.ema(closes, period=period)
        return self._memo[key]

    def _compute_higher_tf_structure(self) -> str:
        tfs = self.strategy.get("timeframes")
        if not isinstance(tfs, list) or len(tfs) < 2 or not isinstance(self.mtf_candles, dict):
            return self.structure()

        htf = tfs[0]
        series = self._mtf_series_up_to(htf)
        if len(series) < 25:
            return "unknown"

        return StructureDetector.structure(series)

    def _mtf_series_up_to(self, timeframe: str) -> List[Candle]:
        if not self.mtf_candles:
            return []
        rows = self.mtf_candles.get(timeframe, [])
        if not rows:
            return []

        tmax = self._candle_ts(self.current_candle)
        
        # Convert HTF to milliseconds
        tf_ms = 60_000
        if timeframe.endswith("m"): tf_ms = int(timeframe[:-1]) * 60_000
        elif timeframe.endswith("h"): tf_ms = int(timeframe[:-1]) * 3_600_000
        elif timeframe.endswith("d"): tf_ms = int(timeframe[:-1]) * 86_400_000
        
        # Prevent future data leakage: only include the HTF candle if it is fully closed
        return [c for c in rows if self._candle_ts(c) + tf_ms <= tmax + 60_000]

    def _candle_ts(self, candle: Candle) -> int:
        if candle is None:
            return 0
        return candle.get_timestamp_int()

    def mtf_rsi(self, timeframe: str, period: int = 14) -> Optional[float]:
        key = f"mtf_rsi_{timeframe}_{period}"
        if key not in self._memo:
            series = self._mtf_series_up_to(timeframe)
            if not series:
                return None
            vals = MomentumEngine.rsi(series, period=period)
            compact = [v for v in vals if v is not None]
            self._memo[key] = compact[-1] if compact else None
        return self._memo[key]

    def mtf_ema(self, timeframe: str, period: int = 20) -> Optional[float]:
        key = f"mtf_ema_{timeframe}_{period}"
        if key not in self._memo:
            series = self._mtf_series_up_to(timeframe)
            if not series:
                return None
            closes = [c.close for c in series]
            vals = MomentumEngine.ema(closes, period=period)
            compact = [v for v in vals if v is not None]
            self._memo[key] = compact[-1] if compact else None
        return self._memo[key]
