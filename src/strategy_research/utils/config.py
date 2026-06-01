"""Configuration management for strategy research."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


@dataclass
class DataConfig:
    """Binance data fetching configuration."""
    base_url: str = "https://fapi.binance.com"
    max_limit: int = 1500
    request_delay: float = 0.12
    max_retries: int = 5
    backoff_base: float = 0.5
    timeout: float = 25.0


@dataclass
class TimeframeConfig:
    """Multi-timeframe configuration."""
    base_tf: str = "1m"
    mtf_tfs: List[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    all_tfs: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])

    INTERVAL_MS: Dict[str, int] = field(default_factory=lambda: {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
        "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
    })


@dataclass
class BarrierConfig:
    """Triple-barrier labeling configuration."""
    up_pct: float = 0.01
    dn_pct: float = 0.005
    max_horizon: int = 120
    round_trip_cost: float = 0.0009
    leverage: float = 10.0
    maint_margin_rate: float = 0.005
    liq_safety: float = 0.5
    funding_rate: float = 0.0001
    funding_interval_h: float = 8.0
    funding_anchor_utc_hour: int = 0
    both_sides: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    ema_fast: int = 9
    ema_slow: int = 21
    vol_lookback: int = 20
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    momentum_lookback: int = 10


@dataclass
class AIGeneratorConfig:
    """AI strategy generation configuration."""
    model_type: str = "xgboost"
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_child_weight: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class WalkForwardConfig:
    """Walk-forward validation configuration."""
    n_folds: int = 5
    embargo: int = 120
    min_fold_size: int = 1000


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    fee_per_side: float = 0.0005
    slippage: float = 0.0002
    position_size: float = 1.0


@dataclass
class StrategyConfig:
    """Strategy output configuration."""
    min_trades: int = 30
    min_profit_factor: float = 1.1
    min_expectancy: float = 0.0
    min_win_rate: float = 0.30
    max_drawdown: float = 0.20


@dataclass
class ResearchConfig:
    """Master research configuration."""
    symbol: str = "SOLUSDT"
    days: int = 60
    data: DataConfig = field(default_factory=DataConfig)
    timeframes: TimeframeConfig = field(default_factory=TimeframeConfig)
    barrier: BarrierConfig = field(default_factory=BarrierConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ai: AIGeneratorConfig = field(default_factory=AIGeneratorConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    # Convenience property for fee-adjusted barriers
    @property
    def gross_up_barrier(self) -> float:
        """Gross price move needed to net the target after costs."""
        return self.barrier.up_pct + self.barrier.round_trip_cost

    @property
    def target_margin_return(self) -> float:
        """Target return on margin (leverage-adjusted)."""
        return self.barrier.up_pct * self.barrier.leverage


def load_config(
    symbol: str = "SOLUSDT",
    days: int = 60,
    leverage: float = 10.0,
    up_pct: float = 0.01,
    dn_pct: float = 0.005,
    **kwargs
) -> ResearchConfig:
    """Create a research config with sensible defaults.

    Args:
        symbol: Trading pair (e.g., SOLUSDT, BTCUSDT, ETHUSDT)
        days: Days of historical data to fetch
        leverage: Leverage multiplier
        up_pct: Target favorable move (price space)
        dn_pct: Stop loss (price space)
        **kwargs: Additional overrides
    """
    config = ResearchConfig(
        symbol=symbol.upper(),
        days=days,
    )
    config.barrier.leverage = leverage
    config.barrier.up_pct = up_pct
    config.barrier.dn_pct = dn_pct

    # Override with any additional kwargs
    for key, val in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, val)

    return config
