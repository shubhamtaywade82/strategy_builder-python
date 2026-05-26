import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from .fee_model import FeeModel
from .slippage_model import SlippageModel
from .fill_model import FillModel
from .trailing_model import TrailingModel
from .partial_exit_model import PartialExitModel
from .position_state import PositionState
from .metrics import Metrics
from ..configuration import Configuration
from ..domain import Candle
from ..features.volume_profile import VolumeProfile
from ..features.volatility_profile import VolatilityProfile
from .evaluation_context import EvaluationContext

@dataclass
class Position:
    id: int
    direction: str
    entry_price: float
    entry_time: Union[int, Any]
    entry_index: int
    size: float
    stop_price: float
    targets: List[float] = field(default_factory=list)
    partial_exits: List[float] = field(default_factory=list)
    trail_config: Optional[str] = None
    state: str = PositionState.ACTIVE
    fills: List[Dict[str, Any]] = field(default_factory=list)
    pnl: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[Union[int, Any]] = None
    exit_reason: Optional[str] = None
    remaining_size: float = 0.0
    be_shifted: bool = False
    current_trail_stop: Optional[float] = None

@dataclass
class Trade:
    position_id: int
    direction: str
    entry_price: float
    exit_price: float
    entry_time: Union[int, Any]
    exit_time: Union[int, Any]
    size: float
    pnl: float
    pnl_r: float
    fees: float
    slippage: float
    exit_reason: str
    hold_candles: int

class BacktestEngine:
    def __init__(self, 
                 fee_model: Optional[FeeModel] = None,
                 slippage_model: Optional[SlippageModel] = None,
                 fill_model: Optional[FillModel] = None,
                 trailing_model: Optional[TrailingModel] = None,
                 partial_exit_model: Optional[PartialExitModel] = None):
        self.fee_model = fee_model or FeeModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.fill_model = fill_model or FillModel()
        self.trailing_model = trailing_model or TrailingModel()
        self.partial_exit_model = partial_exit_model or PartialExitModel()
        self.logger = logging.getLogger(__name__)

    def run(self, strategy: Dict[str, Any], candles: List[Candle], signal_generator: Callable, mtf_candles: Optional[Dict[str, List[Candle]]] = None) -> Dict[str, Any]:
        trades = []
        position = None
        position_counter = 0
        warmup = Configuration().backtest_indicator_warmup
        candles_prefix = []

        for i, candle in enumerate(candles):
            candles_prefix.append(candle)
            if i < warmup:
                continue

            if position and PositionState.is_active(position):
                exit_result = self._check_exits(position, candle, i, strategy, candles_prefix)
                if exit_result:
                    trade = self._close_position(position, exit_result, candle, i, candles_prefix)
                    trades.append(trade)
                    position = None
                elif position.trail_config:
                    position = self.trailing_model.update(position, candle)

            if position:
                continue

            signal = signal_generator(candles_prefix, runtime_mtf=mtf_candles)
            if not signal:
                continue

            if not self._passes_filters(signal, candles_prefix, strategy, mtf_candles=mtf_candles):
                continue

            position_counter += 1
            position = self._open_position(signal, candle, i, position_counter, strategy, candles_prefix)

        if position and PositionState.is_active(position):
            trade = self._close_position(
                position,
                {"reason": "end_of_data", "price": candles[-1].close},
                candles[-1],
                len(candles) - 1,
                candles_prefix
            )
            trades.append(trade)

        metrics = Metrics.compute(trades)
        return {
            "trades": trades,
            "metrics": metrics,
            "strategy_name": strategy.get("name")
        }

    def _open_position(self, signal: Dict[str, Any], candle: Candle, index: int, counter: int, strategy: Dict[str, Any], candles_prefix: List[Candle]) -> Position:
        direction = signal.get("direction", "long")
        raw_entry = signal.get("entry_price", candle.close)
        entry_price = self.slippage_model.apply(raw_entry, direction, candle, candles_so_far=candles_prefix)
        
        size = signal.get("size", 1.0)
        entry_fee = self.fee_model.calculate(entry_price, size)

        stop_distance = self._compute_stop_distance(strategy, candle, signal)
        stop_price = entry_price - stop_distance if direction == "long" else entry_price + stop_distance

        exit_cfg = strategy.get("exit", {})
        target_multiples = exit_cfg.get("targets", [1.0, 2.0])
        targets = []
        for r in target_multiples:
            t = entry_price + (stop_distance * r) if direction == "long" else entry_price - (stop_distance * r)
            targets.append(t)

        return Position(
            id=counter,
            direction=direction,
            entry_price=entry_price,
            entry_time=candle.timestamp,
            entry_index=index,
            size=size,
            remaining_size=size,
            stop_price=stop_price,
            targets=targets,
            partial_exits=exit_cfg.get("partial_exits", [1.0]),
            trail_config=exit_cfg.get("trail"),
            state=PositionState.ACTIVE,
            fills=[{"price": entry_price, "fee": entry_fee, "time": candle.timestamp}],
            pnl=-entry_fee,
            be_shifted=False,
            current_trail_stop=stop_price
        )

    def _check_exits(self, position: Position, candle: Candle, index: int, strategy: Dict[str, Any], candles_prefix: List[Candle]) -> Optional[Dict[str, Any]]:
        PositionState.ensure_active_for_exits(position)

        # 1. Stop loss
        if self._stop_hit(position, candle):
            return {"reason": "stop_loss", "price": position.current_trail_stop or position.stop_price}

        # 2. Targets
        target_result = self.partial_exit_model.check(position, candle)
        if target_result:
            if target_result.get("full_exit"):
                return {"reason": "target_hit", "price": target_result["price"]}
            else:
                self._apply_partial_exit(position, target_result, candle, candles_prefix)
                return None

        # 3. Time stop
        time_stop = strategy.get("exit", {}).get("time_stop_candles")
        if time_stop is not None:
            if (index - position.entry_index) >= time_stop:
                return {"reason": "time_stop", "price": candle.close}

        return None

    def _stop_hit(self, position: Position, candle: Candle) -> bool:
        stop = position.current_trail_stop or position.stop_price
        if position.direction == "long":
            return candle.low <= stop
        else:
            return candle.high >= stop

    def _apply_partial_exit(self, position: Position, result: Dict[str, Any], candle: Candle, candles_prefix: List[Candle]):
        fraction = result["fraction"]
        exit_size = position.size * fraction
        exit_size = min(exit_size, position.remaining_size)
        
        exit_price = self.slippage_model.apply(
            result["price"],
            self._reverse_direction(position.direction),
            candle,
            candles_so_far=candles_prefix
        )
        fee = self.fee_model.calculate(exit_price, exit_size)

        if position.direction == "long":
            pnl = (exit_price - position.entry_price) * exit_size - fee
        else:
            pnl = (position.entry_price - exit_price) * exit_size - fee

        position.remaining_size -= exit_size
        position.pnl += pnl
        position.fills.append({"price": exit_price, "fee": fee, "size": exit_size, "type": "partial_exit"})

    def _close_position(self, position: Position, exit_result: Dict[str, Any], candle: Candle, index: int, candles_prefix: List[Candle]) -> Trade:
        raw_exit = exit_result["price"]
        exit_price = self.slippage_model.apply(
            raw_exit,
            self._reverse_direction(position.direction),
            candle,
            candles_so_far=candles_prefix
        )
        exit_fee = self.fee_model.calculate(exit_price, position.remaining_size)

        if position.direction == "long":
            final_pnl = (exit_price - position.entry_price) * position.remaining_size - exit_fee
        else:
            final_pnl = (position.entry_price - exit_price) * position.remaining_size - exit_fee

        total_pnl = position.pnl + final_pnl
        stop_distance = abs(position.entry_price - position.stop_price)
        pnl_r = total_pnl / (stop_distance * position.size) if stop_distance > 0 else 0.0

        hold_bars = index - position.entry_index

        return Trade(
            position_id=position.id,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=candle.timestamp,
            size=position.size,
            pnl=total_pnl,
            pnl_r=pnl_r,
            fees=sum(f.get("fee", 0) for f in position.fills) + exit_fee,
            slippage=0.0,
            exit_reason=exit_result["reason"],
            hold_candles=hold_bars
        )

    def _compute_stop_distance(self, strategy: Dict[str, Any], candle: Candle, signal: Dict[str, Any]) -> float:
        default_frac = Configuration().backtest_default_stop_price_fraction
        return signal.get("stop_distance") or (candle.close * default_frac)

    def _passes_filters(self, signal: Dict[str, Any], candles: List[Candle], strategy: Dict[str, Any], mtf_candles: Optional[Dict[str, List[Candle]]] = None) -> bool:
        ctx = EvaluationContext(candles, strategy, mtf_candles=mtf_candles)
        filters = strategy.get("filters", {})

        min_vol_z = filters.get("min_volume_zscore")
        if min_vol_z is not None:
            if ctx.volume_zscore() < min_vol_z:
                return False

        min_atr_pct = filters.get("min_atr_percent")
        if min_atr_pct is not None:
            if ctx.atr_percent() < min_atr_pct:
                return False

        required_regime = filters.get("required_regime")
        if required_regime:
            if ctx.regime() not in required_regime:
                return False

        return True

    def _reverse_direction(self, direction: str) -> str:
        return "short" if direction == "long" else "long"
