import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable
from .parallel_instrument_runner import ParallelInstrumentRunner
from ..features.feature_builder import FeatureBuilder

class DiscoverPhase:
    def __init__(self, logger: logging.Logger, candle_loader_factory: Callable, parallel_max: int):
        self.logger = logger
        self.candle_loader_factory = candle_loader_factory
        self.parallel_max = parallel_max

    def execute(self, instruments: List[str], timeframes: List[str], days_back: int, memory: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_from = datetime.now() - timedelta(days=days_back)
        
        def process_instrument(instrument: str) -> tuple:
            self.logger.info(f"Discovering features for {instrument}...")
            loader = self.candle_loader_factory()
            mtf = loader.fetch_mtf(instrument=instrument, timeframes=timeframes, start_from=start_from)
            features = FeatureBuilder.build(instrument=instrument, mtf_candles=mtf)
            return (instrument, features)

        rows = ParallelInstrumentRunner.map_parallel(
            items=instruments,
            max_parallel=self.parallel_max,
            func=process_instrument
        )

        results = dict(rows)
        for instrument, features in results.items():
            memory.append({"phase": "discover", "instrument": instrument, "features": features})
        
        return results
