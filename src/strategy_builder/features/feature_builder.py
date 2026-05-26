from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from .mtf_stack import MtfStack
from .session_detector import SessionDetector
from .volatility_profile import VolatilityProfile
from .structure_detector import StructureDetector
from .volume_profile import VolumeProfile
from .momentum_engine import MomentumEngine
from ..domain import Candle
from ..exceptions import DataError

class FeatureBuilder:
    @staticmethod
    def build(instrument: str, mtf_candles: Dict[str, List[Candle]]) -> Dict[str, Any]:
        if not mtf_candles:
            raise DataError(f"No candle data for {instrument}")

        primary_tf = FeatureBuilder._select_primary_timeframe(mtf_candles)
        primary_candles = mtf_candles[primary_tf]

        return {
            "instrument": instrument,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "primary_timeframe": primary_tf,
            "candle_counts": {tf: len(c) for tf, c in mtf_candles.items()},
            "mtf_alignment": MtfStack.profile(mtf_candles),
            "sessions": SessionDetector.detect_sessions(primary_candles),
            "session_ranges": FeatureBuilder._build_session_ranges(primary_candles),
            "volatility": VolatilityProfile.profile(primary_candles),
            "structure": StructureDetector.profile(primary_candles),
            "volume": VolumeProfile.profile(primary_candles),
            "momentum": MomentumEngine.profile(primary_candles),
            "per_timeframe_summary": FeatureBuilder._build_per_tf_summary(mtf_candles)
        }

    @staticmethod
    def build_at(instrument: str, mtf_candles: Dict[str, List[Candle]], primary_tf: str, up_to_index: int) -> Dict[str, Any]:
        primary_series = mtf_candles[primary_tf][:up_to_index + 1]
        if not primary_series:
            return {}
            
        last_ts = primary_series[-1].timestamp
        if hasattr(last_ts, "timestamp"):
            last_ts = int(last_ts.timestamp())

        truncated_mtf = {}
        for tf, tf_candles in mtf_candles.items():
            truncated_mtf[tf] = [c for c in tf_candles if (c.timestamp if isinstance(c.timestamp, int) else int(c.timestamp.timestamp())) <= last_ts]

        return FeatureBuilder.build(instrument=instrument, mtf_candles=truncated_mtf)

    @staticmethod
    def _select_primary_timeframe(mtf_candles: Dict[str, List[Candle]]) -> str:
        if "5m" in mtf_candles and len(mtf_candles["5m"]) > 100:
            return "5m"
        return max(mtf_candles.items(), key=lambda x: len(x[1]))[0]

    @staticmethod
    def _build_session_ranges(candles: List[Candle]) -> Dict[str, Any]:
        if not candles:
            return {}

        dates = []
        for c in candles:
            ts = c.timestamp
            if isinstance(ts, int):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            dates.append(dt.date())
        
        unique_dates = sorted(list(set(dates)))[-5:]
        ranges = {}
        for date_val in unique_dates:
            dt = datetime(date_val.year, date_val.month, date_val.day, tzinfo=timezone.utc)
            for session in ["asia", "london", "new_york"]:
                range_val = SessionDetector.session_range(candles, session=session, date=dt)
                if range_val:
                    ranges[f"{date_val}_{session}"] = range_val
        return ranges

    @staticmethod
    def _build_per_tf_summary(mtf_candles: Dict[str, List[Candle]]) -> Dict[str, Any]:
        summary = {}
        for tf, candles in mtf_candles.items():
            if len(candles) < 30:
                summary[tf] = {"insufficient_data": True}
                continue

            rsi_vals = MomentumEngine.rsi(candles)
            compact_rsi = [r for r in rsi_vals if r is not None]
            
            rvol_vals = VolumeProfile.relative_volume(candles)
            compact_rvol = [v for v in rvol_vals if v is not None]
            
            atr_pct_vals = VolatilityProfile.atr_percent(candles)
            compact_atr_pct = [v for v in atr_pct_vals if v is not None]

            summary[tf] = {
                "trend": MtfStack.trend_direction(candles),
                "structure": StructureDetector.structure(candles),
                "volatility_regime": VolatilityProfile.regime(candles),
                "rsi": round(compact_rsi[-1], 1) if compact_rsi else None,
                "relative_volume": round(compact_rvol[-1], 2) if compact_rvol else None,
                "atr_percent": round(compact_atr_pct[-1], 3) if compact_atr_pct else None,
                "candle_count": len(candles)
            }
        return summary
