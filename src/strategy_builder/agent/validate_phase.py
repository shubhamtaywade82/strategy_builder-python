import logging
import itertools
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Optional
from .parallel_instrument_runner import ParallelInstrumentRunner
from ..backtest.signal_evaluator import SignalEvaluator
from ..ranking.robustness import Robustness
from ..configuration import Configuration

class ValidatePhase:
    def __init__(self,
                 logger: logging.Logger,
                 candle_loader_factory: Callable,
                 backtest_engine_factory: Callable,
                 walk_forward_factory: Callable,
                 parallel_max: int):
        self.logger = logger
        self.candle_loader_factory = candle_loader_factory
        self.backtest_engine_factory = backtest_engine_factory
        self.walk_forward_factory = walk_forward_factory
        self.parallel_max = parallel_max

    def execute(self, catalog: Any, instruments: List[str], days_back: int, memory: List[Dict[str, Any]]):
        proposed = catalog.by_status("proposed")
        self.logger.info(f"Validating {len(proposed)} proposed strategies...")
        if not proposed:
            return

        start_from = datetime.now() - timedelta(days=days_back)
        jobs = list(itertools.product(proposed, instruments))

        def process_job(pair: tuple) -> Optional[Dict[str, Any]]:
            entry, instrument = pair
            return self._run_job(entry=entry, instrument=instrument, start_from=start_from)

        rows = ParallelInstrumentRunner.map_parallel(
            items=jobs,
            max_parallel=self.parallel_max,
            func=process_job
        )

        for row in rows:
            if row:
                catalog.attach_backtest(row["id"], row["payload"])
                memory.append(row["memory"])

    def _run_job(self, entry: Dict[str, Any], instrument: str, start_from: datetime) -> Optional[Dict[str, Any]]:
        strategy = entry["strategy"]
        self.logger.info(f"Backtesting {strategy['name']} on {instrument}...")

        loader = self.candle_loader_factory()
        tfs = strategy.get("timeframes")
        timeframes = tfs if isinstance(tfs, list) and tfs else Configuration().default_timeframes
        primary_tf = timeframes[-1] if timeframes else "5m"
        
        mtf = loader.fetch_mtf(instrument=instrument, timeframes=timeframes, start_from=start_from)
        candles = mtf.get(primary_tf, [])
        if len(candles) < 200:
            return None

        signal_gen = SignalEvaluator.build(strategy, mtf_candles=mtf)

        engine = self.backtest_engine_factory()
        walk_forward = self.walk_forward_factory(engine)
        
        wf_result = walk_forward.run(
            strategy=strategy,
            candles=candles,
            signal_generator=signal_gen,
            mtf_candles=mtf
        )

        wf_result["regime_slices"] = walk_forward.volatility_regime_slices(
            strategy=strategy,
            candles=candles,
            signal_generator=signal_gen,
            mtf_candles=mtf
        )
        
        wf_result["anchored_holdout"] = walk_forward.anchored_holdout(
            strategy=strategy,
            candles=candles,
            signal_generator=signal_gen,
            mtf_candles=mtf
        )

        session_results = walk_forward.session_analysis(
            strategy=strategy,
            candles=candles,
            signal_generator=signal_gen,
            mtf_candles=mtf
        )

        robustness_result = Robustness.analyze(
            strategy=strategy,
            candles=candles,
            engine=engine,
            mtf_candles=mtf,
            signal_generator_factory=lambda mutated: SignalEvaluator.build(mutated, mtf_candles=mtf)
        )

        return {
            "id": entry["id"],
            "payload": {
                "metrics": wf_result["aggregate"],
                "walk_forward": wf_result,
                "session_results": session_results,
                "robustness_result": robustness_result,
                "instrument": instrument,
                "candle_count": len(candles)
            },
            "memory": {"phase": "validate", "strategy_id": entry["id"], "instrument": instrument, "result": wf_result["aggregate"]}
        }
