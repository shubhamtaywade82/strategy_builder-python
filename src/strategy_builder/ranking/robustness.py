import copy
from typing import Dict, Any, List, Optional, Callable

class Robustness:
    PERTURBATION_STEPS = 5

    @staticmethod
    def analyze(strategy: Dict[str, Any], candles: List[Any], engine: Any, signal_generator_factory: Callable, mtf_candles: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        ranges = strategy.get("parameter_ranges", {})
        if not ranges:
            return {"robustness_score": 0.5, "tested_params": 0}

        results_per_param = {}

        for param_name, range_spec in ranges.items():
            if not isinstance(range_spec, list) or len(range_spec) != 2:
                continue

            low, high = range_spec
            if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
                continue

            step = (high - low) / float(Robustness.PERTURBATION_STEPS)
            param_results = []

            for i in range(Robustness.PERTURBATION_STEPS + 1):
                value = low + (step * i)
                mutated = Robustness._deep_merge_param(strategy, param_name, value)
                sg = signal_generator_factory(mutated)

                bt = engine.run(
                    strategy=mutated,
                    candles=candles,
                    mtf_candles=mtf_candles,
                    signal_generator=sg
                )
                param_results.append({
                    "value": value,
                    "expectancy": bt.get("metrics", {}).get("expectancy", 0.0)
                })

            results_per_param[param_name] = param_results

        robustness_score = Robustness._compute_robustness_score(results_per_param)

        return {
            "robustness_score": robustness_score,
            "tested_params": len(results_per_param),
            "param_sensitivity": {name: Robustness._summarize_sensitivity(res) for name, res in results_per_param.items()}
        }

    @staticmethod
    def _compute_robustness_score(results: Dict[str, List[Dict[str, Any]]]) -> float:
        if not results:
            return 0.5

        scores = []
        for param_results in results.values():
            expectancies = [r["expectancy"] for r in param_results]
            positive_count = sum(1 for e in expectancies if e > 0)
            scores.append(positive_count / len(expectancies))

        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def _summarize_sensitivity(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        expectancies = [r["expectancy"] for r in results]
        max_exp = max(expectancies)
        min_exp = min(expectancies)
        
        abs_max = max(abs(e) for e in expectancies) if expectancies else 0
        
        return {
            "min": min_exp,
            "max": max_exp,
            "mean": round(sum(expectancies) / len(expectancies), 4),
            "all_positive": all(e > 0 for e in expectancies),
            "stable": abs(max_exp - min_exp) < abs_max * 0.5 if abs_max > 0 else True
        }

    @staticmethod
    def _deep_merge_param(strategy: Dict[str, Any], param_name: str, value: Any) -> Dict[str, Any]:
        mutated = copy.deepcopy(strategy)
        
        found = False
        for section in ["filters", "risk", "exit", "entry"]:
            if section in mutated and isinstance(mutated[section], dict):
                if param_name in mutated[section]:
                    mutated[section][param_name] = value
                    found = True
                    break
        
        if not found:
            if "parameter_ranges" not in mutated:
                mutated["parameter_ranges"] = {}
            mutated["parameter_ranges"][param_name] = value
            
        return mutated
