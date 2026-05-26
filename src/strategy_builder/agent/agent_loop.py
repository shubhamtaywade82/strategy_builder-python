import logging
from typing import List, Dict, Any, Optional, Callable
from .discover_phase import DiscoverPhase
from .validate_phase import ValidatePhase
from .desk_pipeline import DeskPipeline
from ..generation.strategy_catalog import StrategyCatalog
from ..generation.strategy_generator import StrategyGenerator
from ..generation.strategy_templates import StrategyTemplates
from ..market_data.candle_loader import CandleLoader
from ..backtest.engine import BacktestEngine
from ..backtest.walk_forward import WalkForward
from ..backtest.condition_registry import ConditionRegistry
from ..ranking.gatekeeper import Gatekeeper
from ..ranking.scorer import Scorer
from ..documentation.strategy_card import StrategyCard
from ..documentation.markdown_exporter import MarkdownExporter
from ..documentation.json_exporter import JsonExporter
from ..configuration import Configuration

class AgentLoop:
    def __init__(self,
                 client: Any,
                 max_steps: Optional[int] = None,
                 catalog_factory: Optional[Callable] = None,
                 candle_loader_factory: Optional[Callable] = None,
                 backtest_engine_factory: Optional[Callable] = None,
                 walk_forward_factory: Optional[Callable] = None,
                 strategy_generator_factory: Optional[Callable] = None,
                 desk_pipeline_factory: Optional[Callable] = None,
                 discover_phase: Optional[DiscoverPhase] = None,
                 validate_phase: Optional[ValidatePhase] = None,
                 parallel_instrument_max: Optional[int] = None):
        
        cfg = Configuration()
        self.client = client
        self.max_steps = max_steps or cfg.max_agent_iterations
        self.logger = logging.getLogger(__name__)
        self.memory = []
        self.step_count = 0
        
        self.catalog_factory = catalog_factory or (lambda: StrategyCatalog())
        self.candle_loader_factory = candle_loader_factory or (lambda: CandleLoader())
        self.backtest_engine_factory = backtest_engine_factory or (lambda: BacktestEngine())
        self.walk_forward_factory = walk_forward_factory or (lambda eng: WalkForward(engine=eng))
        self.strategy_generator_factory = strategy_generator_factory or (lambda: StrategyGenerator(client=client))
        self.desk_pipeline_factory = desk_pipeline_factory or (lambda: DeskPipeline(client=client))

        pmax = parallel_instrument_max or cfg.parallel_instrument_max
        self.discover_phase = discover_phase or DiscoverPhase(
            logger=self.logger,
            candle_loader_factory=self.candle_loader_factory,
            parallel_max=pmax
        )
        self.validate_phase = validate_phase or ValidatePhase(
            logger=self.logger,
            candle_loader_factory=self.candle_loader_factory,
            backtest_engine_factory=self.backtest_engine_factory,
            walk_forward_factory=self.walk_forward_factory,
            parallel_max=pmax
        )

    def run(self, query: str) -> Dict[str, Any]:
        self.logger.info(f"Agent starting (deterministic pipeline): {query}")
        catalog = self.catalog_factory()
        steps = self._run_manual_pipeline(query)
        
        return {
            "steps": steps,
            "final_result": None,
            "strategies_found": catalog.size(),
            "passing_strategies": len(catalog.passing())
        }

    def discover(self, instruments: Optional[List[str]] = None, timeframes: Optional[List[str]] = None, days_back: int = 30) -> Dict[str, Any]:
        cfg = Configuration()
        instruments = instruments or cfg.default_instruments
        timeframes = timeframes or cfg.default_timeframes

        return self.discover_phase.execute(
            instruments=instruments,
            timeframes=timeframes,
            days_back=days_back,
            memory=self.memory
        )

    def propose(self, features_by_instrument: Dict[str, Any], mode: str = "generate") -> List[Dict[str, Any]]:
        catalog = self.catalog_factory()
        all_candidates = []
        seen_keys = set()

        for instrument, features in features_by_instrument.items():
            self.logger.info(f"Proposing strategies for {instrument} via DeskPipeline (mode: {mode})...")

            if mode == "mutate":
                candidates = self.strategy_generator_factory().mutate(features=features)
            else:
                candidates = self.desk_pipeline_factory().run(instrument=instrument, features=features)

            for candidate in candidates:
                key = self._proposal_dedupe_key(candidate)
                if key in seen_keys:
                    self.logger.info(f"Skipping duplicate proposal {candidate.get('name')} ({key})")
                    continue

                seen_keys.add(key)
                strat_id = catalog.add(candidate)
                all_candidates.append({"id": strat_id, "name": candidate.get("name"), "instrument": instrument})
                self.logger.info(f"Added candidate: {candidate.get('name')} ({strat_id})")

        self.memory.append({"phase": "propose", "candidates": all_candidates})
        return all_candidates

    def validate(self, catalog: Optional[StrategyCatalog] = None, instruments: Optional[List[str]] = None, days_back: int = 90):
        catalog = catalog or self.catalog_factory()
        instruments = instruments or Configuration().default_instruments

        self.validate_phase.execute(
            catalog=catalog,
            instruments=instruments,
            days_back=days_back,
            memory=self.memory
        )

    def rank(self, catalog: Optional[StrategyCatalog] = None) -> List[Dict[str, Any]]:
        catalog = catalog or self.catalog_factory()
        backtested = catalog.by_status("backtested")
        self.logger.info(f"Ranking {len(backtested)} backtested strategies...")

        for entry in backtested:
            wf_result = entry.get("backtest_results", {}).get("walk_forward")
            if not wf_result:
                continue

            gate_result = Gatekeeper.evaluate(walk_forward_result=wf_result)
            score_result = Scorer.score(
                walk_forward_result=wf_result,
                session_results=entry.get("backtest_results", {}).get("session_results"),
                robustness_result=entry.get("backtest_results", {}).get("robustness_result")
            )

            ranking = {**score_result, **gate_result}
            catalog.attach_ranking(entry["id"], ranking)

            self.logger.info(f"Ranked {entry['strategy']['name']}: score={score_result['final_score']} status={gate_result['status']}")
            self.memory.append({"phase": "rank", "strategy_id": entry["id"], "ranking": ranking})

        return catalog.ranked()

    def document(self, catalog: Optional[StrategyCatalog] = None):
        catalog = catalog or self.catalog_factory()
        passing = catalog.passing()
        self.logger.info(f"Documenting {len(passing)} passing strategies...")

        for entry in passing:
            card = StrategyCard.build(entry)
            MarkdownExporter.export(card)
            JsonExporter.export(card)

            try:
                generator = self.strategy_generator_factory()
                llm_doc = generator.document(
                    strategy=entry["strategy"],
                    backtest_results=entry["backtest_results"]
                )
                if llm_doc:
                    catalog.attach_documentation(entry["id"], llm_doc)
            except Exception as e:
                self.logger.warning(f"LLM documentation failed for {entry['id']}: {e}")

    def _proposal_dedupe_key(self, candidate: Dict[str, Any]) -> str:
        name = str(candidate.get("name", "")).lower()
        import re
        name = re.sub(r"\s*\((?:offline|mutated) template\)\s*$", "", name).strip()
        family = str(candidate.get("family", "")).lower()
        return f"{family}:{name}"

    def _run_manual_pipeline(self, query: str) -> List[Dict[str, Any]]:
        steps = []
        
        # Phase 1: Discover
        features = self.discover()
        steps.append({"type": "discover", "instruments": list(features.keys())})
        
        # Phase 2: Propose
        candidates = self.propose(features_by_instrument=features)
        steps.append({"type": "propose", "candidates_count": len(candidates)})
        
        # Phase 3: Validate
        self.validate()
        steps.append({"type": "validate", "status": "complete"})
        
        # Phase 4: Rank
        ranked = self.rank()
        steps.append({"type": "rank", "ranked_count": len(ranked)})
        
        # Phase 5: Document
        self.document()
        steps.append({"type": "document", "status": "complete"})
        
        return steps
