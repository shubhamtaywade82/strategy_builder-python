import logging
from typing import List, Any, Callable
from concurrent.futures import ThreadPoolExecutor

class ParallelInstrumentRunner:
    @staticmethod
    def map_parallel(items: List[Any], max_parallel: int, func: Callable[[Any], Any]) -> List[Any]:
        if not items:
            return []
        if max_parallel <= 1:
            return [func(item) for item in items]

        max_workers = min(max_parallel, len(items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(func, items))
                return results
            except Exception as e:
                logging.getLogger(__name__).error(f"Parallel execution error: {e}")
                raise e
