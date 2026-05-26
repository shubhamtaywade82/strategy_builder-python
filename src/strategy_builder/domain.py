from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class Timeframe(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MN1 = "1M"

class Regime(str, Enum):
    DEAD_MARKET = "dead_market"
    LOW_VOL_CHOP = "low_vol_chop"
    RANGE = "range"
    ACCUMULATION = "accumulation"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    TREND_EXPANSION = "trend_expansion"
    LIQUIDATION_EVENT = "liquidation_event"
    BREAKOUT_ENVIRONMENT = "breakout_environment"
    MEAN_REVERSION = "mean_reversion"
    HIGH_RISK_CHAOS = "high_risk_chaos"
    CHOP = "chop"

class Session(str, Enum):
    ASIA = "asia"
    LONDON = "london"
    NY_OPEN = "ny_open"
    NY_AFTERNOON = "ny_afternoon"
    LATE_NY = "late_ny"
    ASIA_LONDON_OVERLAP = "asia_london_overlap"
    LONDON_NY_OVERLAP = "london_ny_overlap"
    CLOSED = "closed"
    UNKNOWN = "unknown"

class Bias(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class VolatilityState(str, Enum):
    CONTRACTING = "contracting"
    NORMAL = "normal"
    EXPANDING = "expanding"

class VolumeState(str, Enum):
    DECLINING = "declining"
    AVERAGE = "average"
    EXPANDING = "expanding"

class Candle(BaseModel):
    timestamp: Union[int, datetime]
    open: float
    high: float
    low: float
    close: float
    volume: float

    def get_timestamp_int(self) -> int:
        if isinstance(self.timestamp, datetime):
            return int(self.timestamp.timestamp())
        return self.timestamp

class LiquidityInfo(BaseModel):
    equal_highs: List[Dict[str, Any]] = Field(default_factory=list)
    equal_lows: List[Dict[str, Any]] = Field(default_factory=list)
    nearest_resist: Optional[Dict[str, Any]] = None
    nearest_support: Optional[Dict[str, Any]] = None

class MarketState(BaseModel):
    instrument: str
    snapshot_at: datetime
    primary_timeframe: str
    regime: Regime
    session: str
    higher_tf_bias: Optional[str] = None
    mid_tf_structure: Optional[str] = None
    lower_tf_state: Optional[str] = None
    volatility: VolatilityState
    volume: VolumeState
    liquidity: Optional[LiquidityInfo] = None
    bias: Bias
    raw_features: Optional[Dict[str, Any]] = None

    def to_llm_context(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "snapshot_at": self.snapshot_at.iso8601() if hasattr(self.snapshot_at, "iso8601") else self.snapshot_at.isoformat(),
            "primary_timeframe": self.primary_timeframe,
            "regime": self.regime.value,
            "session": self.session,
            "higher_tf_bias": self.higher_tf_bias,
            "mid_tf_structure": self.mid_tf_structure,
            "lower_tf_state": self.lower_tf_state,
            "volatility": self.volatility.value,
            "volume": self.volume.value,
            "bias": self.bias.value,
            "liquidity": {
                "equal_highs": self.liquidity.equal_highs[:3] if self.liquidity else [],
                "equal_lows": self.liquidity.equal_lows[:3] if self.liquidity else [],
                "nearest_resist": self.liquidity.nearest_resist if self.liquidity else None,
                "nearest_support": self.liquidity.nearest_support if self.liquidity else None
            } if self.liquidity else None
        }

    def is_valid(self) -> bool:
        if not self.instrument:
            return False
        return True
