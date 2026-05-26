import math
import logging
from typing import List, Dict, Any, Optional, Callable
from .engine import BacktestEngine
from ..configuration import Configuration
from ..exceptions import BacktestError
from ..features.session_detector import SessionDetector
from ..features.volatility_profile import VolatilityProfile

class WalkForward:
    DEFAULT_FOLDS = 5

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()
        self.logger = logging.getLogger(__name__)
        self.in_sample_ratio = Configuration().walk_forward_in_sample_ratio

    def run(self, strategy: Dict[str, Any], candles: List[Any], signal_generator: Callable, folds: int = DEFAULT_FOLDS, mtf_candles: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        fold_size = len(candles) // folds
        if fold_size < 100:
            raise BacktestError(f"Insufficient data for {folds} folds ({len(candles)} candles)")

        fold_results = []

        for i in range(folds):
            fold_start = i * fold_size
            fold_end = min((i + 1) * fold_size, len(candles))

            fold_candles = candles[fold_start:fold_end]
            split_idx = int(len(fold_candles) * self.in_sample_ratio)

            in_sample = fold_candles[:split_idx]
            out_of_sample = fold_candles[split_idx:]

            self.logger.info(f"Fold {i + 1}/{folds}: IS={len(in_sample)} OOS={len(out_of_sample)} candles")

            is_result = self.engine.run(
                strategy=strategy,
                candles=in_sample,
                mtf_candles=mtf_candles,
                signal_generator=signal_generator
            )

            oos_result = self.engine.run(
                strategy=strategy,
                candles=out_of_sample,
                mtf_candles=mtf_candles,
                signal_generator=signal_generator
            )

            fold_results.append({
                "fold": i + 1,
                "in_sample": is_result["metrics"],
                "out_of_sample": oos_result["metrics"],
                "is_trade_count": len(is_result["trades"]),
                "oos_trade_count": len(oos_result["trades"]),
                "degradation": self._compute_degradation(is_result["metrics"], oos_result["metrics"])
            })

        aggregate = self._aggregate_folds(fold_results)
        stability = self._compute_stability(fold_results)

        return {
            "folds": fold_results,
            "aggregate": aggregate,
            "stability_score": stability,
            "passes_walk_forward": stability > 0.5 and aggregate["oos_expectancy"] > 0
        }

    def session_analysis(self, strategy: Dict[str, Any], candles: List[Any], signal_generator: Callable, mtf_candles: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        session_groups = SessionDetector.group_by_session(candles)
        results = {}

        for session, session_candles in session_groups.items():
            if len(session_candles) < 100:
                continue

            bt = self.engine.run(
                strategy=strategy,
                candles=session_candles,
                mtf_candles=mtf_candles,
                signal_generator=signal_generator
            )
            results[session] = bt["metrics"]
        
        return results

    def anchored_holdout(self, strategy: Dict[str, Any], candles: List[Any], signal_generator: Callable, train_ratio: float = 0.65, mtf_candles: Optional[Dict[str, List[Any]]] = None) -> Optional[Dict[str, Any]]:
        warmup = Configuration().backtest_indicator_warmup
        if len(candles) < warmup + 120:
            return None

        split = min(int(len(candles) * train_ratio), len(candles) - 80)
        if split <= warmup or split >= len(candles) - 20:
            return None

        train = candles[:split]
        test = candles[split:]

        is_bt = self.engine.run(strategy=strategy, candles=train, mtf_candles=mtf_candles, signal_generator=signal_generator)
        oos_bt = self.engine.run(strategy=strategy, candles=test, mtf_candles=mtf_candles, signal_generator=signal_generator)

        return {
            "train_ratio": train_ratio,
            "train_candles": len(train),
            "test_candles": len(test),
            "in_sample": is_bt["metrics"],
            "out_of_sample": oos_bt["metrics"]
        }

    def volatility_regime_slices(self, strategy: Dict[str, Any], candles: List[Any], signal_generator: Callable, chunk_size: int = 400, mtf_candles: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        warmup = Configuration().backtest_indicator_warmup
        if len(candles) < chunk_size + warmup:
            return {"segments": [], "fraction_positive_expectancy": 0.5}

        segments = []
        step = max(chunk_size // 2, 100)
        i = 0
        while i + chunk_size <= len(candles):
            slice_data = candles[i:i + chunk_size]
            reg = VolatilityProfile.regime(slice_data)
            res = self.engine.run(
                strategy=strategy,
                candles=slice_data,
                mtf_candles=mtf_candles,
                signal_generator=signal_generator
            )
            exp = float(res["metrics"]["expectancy"])
            segments.append({
                "start_index": i,
                "regime": reg,
                "expectancy": exp,
                "trade_count": res["metrics"]["trade_count"]
            })
            i += step

        positive = sum(1 for s in segments if s["expectancy"] > 0)
        frac = round(positive / len(segments), 4) if segments else 0.5

        return {"segments": segments, "fraction_positive_expectancy": frac}

    def _compute_degradation(self, is_metrics: Dict[str, Any], oos_metrics: Dict[str, Any]) -> float:
        is_exp = is_metrics.get("expectancy", 0.0)
        if is_exp == 0:
            return 1.0
        
        oos_exp = oos_metrics.get("expectancy", 0.0)
        return 1.0 - (oos_exp / is_exp)

    def _aggregate_folds(self, folds: List[Dict[str, Any]]) -> Dict[str, Any]:
        oos_metrics = [f["out_of_sample"] for f in folds]
        is_metrics = [f["in_sample"] for f in folds]

        return {
            "oos_expectancy": self._safe_mean([m["expectancy"] for m in oos_metrics]),
            "oos_win_rate": self._safe_mean([m["win_rate"] for m in oos_metrics]),
            "oos_profit_factor": self._safe_mean([m["profit_factor"] for m in oos_metrics]),
            "oos_max_drawdown": max([m["max_drawdown"] for m in oos_metrics]) if oos_metrics else 0.0,
            "oos_avg_r": self._safe_mean([m["avg_r"] for m in oos_metrics]),
            "oos_trade_count": sum([m["trade_count"] for m in oos_metrics]),
            "is_expectancy": self._safe_mean([m["expectancy"] for m in is_metrics]),
            "is_profit_factor": self._safe_mean([m["profit_factor"] for m in is_metrics]),
            "avg_degradation": self._safe_mean([f["degradation"] for f in folds])
        }

    def _compute_stability(self, folds: List[Dict[str, Any]]) -> float:
        oos_expectancies = [f["out_of_sample"]["expectancy"] for f in folds]
        if not oos_expectancies:
            return 0.0

        positive_folds = sum(1 for e in oos_expectancies if e > 0)
        base_stability = positive_folds / len(oos_expectancies)

        mean = sum(oos_expectancies) / len(oos_expectancies)
        variance = sum((e - mean) ** 2 for e in oos_expectancies) / len(oos_expectancies)
        
        if mean == 0:
            cv = float('inf')
        else:
            cv = math.sqrt(variance) / abs(mean)

        stability_penalty = min(cv / 2.0, 0.5)
        return round(max(base_stability - stability_penalty, 0.0), 4)

    def _safe_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)
