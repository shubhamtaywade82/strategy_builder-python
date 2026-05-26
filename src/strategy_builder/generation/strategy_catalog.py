import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from .candidate_parser import CandidateParser
from ..exceptions import ValidationError
from ..configuration import Configuration

class StrategyCatalog:
    STATUSES = ["proposed", "validated", "backtested", "ranked", "pass", "watchlist", "reject"]

    def __init__(self, storage_dir: Optional[str] = None):
        cfg = Configuration()
        self.storage_dir = storage_dir or os.path.join(cfg.output_dir, "strategies")
        os.makedirs(self.storage_dir, exist_ok=True)
        self.catalog = self._load_catalog()

    def add(self, candidate: Dict[str, Any], status: str = "proposed") -> str:
        id_val = self._generate_id(candidate)
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": id_val,
            "strategy": candidate,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "backtest_results": None,
            "ranking": None,
            "documentation": None
        }

        self.catalog[id_val] = entry
        self._persist()
        return id_val

    def update_status(self, id_val: str, status: str):
        if status not in self.STATUSES:
            raise ValidationError(f"Unknown status: {status}")
        if id_val not in self.catalog:
            raise ValidationError(f"Strategy {id_val} not found")

        self.catalog[id_val]["status"] = status
        self.catalog[id_val]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def attach_backtest(self, id_val: str, results: Dict[str, Any]):
        if id_val not in self.catalog:
            raise ValidationError(f"Strategy {id_val} not found")

        prev = self.catalog[id_val].get("backtest_results")
        self.catalog[id_val]["backtest_results"] = self._merge_backtest_results(prev, results)
        self.catalog[id_val]["status"] = "backtested"
        self.catalog[id_val]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def attach_ranking(self, id_val: str, ranking: Dict[str, Any]):
        if id_val not in self.catalog:
            raise ValidationError(f"Strategy {id_val} not found")

        self.catalog[id_val]["ranking"] = ranking
        self.catalog[id_val]["status"] = ranking.get("status", "ranked")
        self.catalog[id_val]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def attach_documentation(self, id_val: str, doc: Dict[str, Any]):
        if id_val not in self.catalog:
            raise ValidationError(f"Strategy {id_val} not found")

        self.catalog[id_val]["documentation"] = doc
        self.catalog[id_val]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._persist()

    def get(self, id_val: str) -> Optional[Dict[str, Any]]:
        return self.catalog.get(id_val)

    def all(self) -> List[Dict[str, Any]]:
        return list(self.catalog.values())

    def by_status(self, status: str) -> List[Dict[str, Any]]:
        return [e for e in self.catalog.values() if e["status"] == status]

    def by_family(self, family: str) -> List[Dict[str, Any]]:
        return [e for e in self.catalog.values() if e["strategy"]["family"] == family]

    def ranked(self, limit: int = 20) -> List[Dict[str, Any]]:
        ranked_list = [e for e in self.catalog.values() if e.get("ranking")]
        ranked_list.sort(key=lambda x: x["ranking"].get("final_score", 0.0), reverse=True)
        return ranked_list[:limit]

    def passing(self) -> List[Dict[str, Any]]:
        return self.by_status("pass")

    def size(self) -> int:
        return len(self.catalog)

    def clear(self):
        self.catalog = {}
        self._persist()

    def _generate_id(self, candidate: Dict[str, Any]) -> str:
        name = candidate.get("name", "unnamed").lower()
        base = "".join(c if c.isalnum() else "_" for c in name)
        base = "_".join(filter(None, base.split("_")))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{base}_{timestamp}"

    def _persist(self):
        with open(self._catalog_path(), 'w') as f:
            json.dump(self.catalog, f, indent=2)

    def _load_catalog(self) -> Dict[str, Any]:
        path = self._catalog_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"Corrupt catalog file: {e}")
            return {}

    def _catalog_path(self) -> str:
        return os.path.join(self.storage_dir, "catalog.json")

    def _merge_backtest_results(self, previous: Optional[Dict[str, Any]], latest: Dict[str, Any]) -> Dict[str, Any]:
        instrument = latest["instrument"]
        per = self._extract_instruments_map(previous)
        per[instrument] = {
            "metrics": latest["metrics"],
            "walk_forward": latest["walk_forward"],
            "candle_count": latest["candle_count"],
            "session_results": latest.get("session_results"),
            "robustness_result": latest.get("robustness_result")
        }

        canonical = self._build_ranking_walk_forward(per)
        return {
            "instruments": per,
            "instrument": instrument,
            "metrics": canonical["aggregate"],
            "walk_forward": canonical,
            "session_results": self._merge_session_results(per),
            "robustness_result": self._merge_robustness_results(per),
            "candle_count": sum(int(v.get("candle_count", 0)) for v in per.values())
        }

    def _extract_instruments_map(self, previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not previous or not isinstance(previous, dict):
            return {}
        if "instruments" in previous and isinstance(previous["instruments"], dict):
            return previous["instruments"].copy()
        
        if "instrument" in previous and "walk_forward" in previous:
            return {
                previous["instrument"]: {
                    "metrics": previous["metrics"],
                    "walk_forward": previous["walk_forward"],
                    "candle_count": previous.get("candle_count", 0)
                }
            }
        return {}

    def _build_ranking_walk_forward(self, per_instrument: Dict[str, Any]) -> Dict[str, Any]:
        wfs = [v["walk_forward"] for v in per_instrument.values() if v.get("walk_forward")]
        if not wfs:
            return {}
        if len(wfs) == 1:
            return wfs[0]

        aggregates = [wf["aggregate"] for wf in wfs if wf.get("aggregate")]
        regime_fracs = [wf.get("regime_slices", {}).get("fraction_positive_expectancy") for wf in wfs if wf.get("regime_slices")]
        regime_fracs = [f for f in regime_fracs if f is not None]
        
        regime_slices = None
        if regime_fracs:
            regime_slices = {
                "fraction_positive_expectancy": self._safe_mean(regime_fracs),
                "merged": True
            }

        return {
            "aggregate": self._merge_aggregate_rows(aggregates),
            "stability_score": self._safe_mean([wf.get("stability_score", 0.0) for wf in wfs]),
            "passes_walk_forward": all(wf.get("passes_walk_forward", False) for wf in wfs),
            "folds": [f for wf in wfs for f in wf.get("folds", [])],
            "regime_slices": regime_slices
        }

    def _merge_session_results(self, per_instrument: Dict[str, Any]) -> Dict[str, Any]:
        acc = {}
        for inst, data in per_instrument.items():
            sess_res = data.get("session_results")
            if isinstance(sess_res, dict):
                for session, metrics in sess_res.items():
                    acc[f"{inst}:{session}"] = metrics
        return acc

    def _merge_robustness_results(self, per_instrument: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        scores = [v.get("robustness_result", {}).get("robustness_score") for v in per_instrument.values() if v.get("robustness_result")]
        scores = [s for s in scores if s is not None]
        if not scores:
            return None

        tested_params = sum(v.get("robustness_result", {}).get("tested_params", 0) for v in per_instrument.values() if v.get("robustness_result"))
        return {
            "robustness_score": self._safe_mean(scores),
            "merged": True,
            "tested_params": tested_params
        }

    def _merge_aggregate_rows(self, aggs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not aggs:
            return {}

        return {
            "oos_expectancy": self._safe_mean([a.get("oos_expectancy", 0.0) for a in aggs]),
            "oos_win_rate": self._safe_mean([a.get("oos_win_rate", 0.0) for a in aggs]),
            "oos_profit_factor": self._safe_mean([a.get("oos_profit_factor", 0.0) for a in aggs]),
            "oos_max_drawdown": max([float(a.get("oos_max_drawdown", 0.0)) for a in aggs]) if aggs else 0.0,
            "oos_avg_r": self._safe_mean([a.get("oos_avg_r", 0.0) for a in aggs]),
            "oos_trade_count": sum(int(a.get("oos_trade_count", 0)) for a in aggs),
            "is_expectancy": self._safe_mean([a.get("is_expectancy", 0.0) for a in aggs]),
            "is_profit_factor": self._safe_mean([a.get("is_profit_factor", 0.0) for a in aggs]),
            "avg_degradation": self._safe_mean([a.get("avg_degradation", 0.0) for a in aggs])
        }

    def _safe_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)
