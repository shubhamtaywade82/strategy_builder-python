import logging
from typing import List, Dict, Any
from .roles.observer import Observer
from .roles.pattern_analyst import PatternAnalyst
from .roles.trade_designer import TradeDesigner
from .roles.skeptic import Skeptic
from ..state.snapshot_builder import SnapshotBuilder
from ..patterns.pattern_miner import PatternMiner

class DeskPipeline:
    def __init__(self, client: Any):
        self.observer = Observer(client)
        self.pattern_analyst = PatternAnalyst(client)
        self.trade_designer = TradeDesigner(client)
        self.skeptic = Skeptic(client)
        self.logger = logging.getLogger(__name__)

    def run(self, instrument: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        snapshot = SnapshotBuilder.build(instrument=instrument, features=features)
        self.logger.info(f"DeskPipeline: {instrument} — regime={snapshot.regime} bias={snapshot.bias} session={snapshot.session}")

        observer_result = self.observer.classify(snapshot)
        self.logger.info(f"Observer: {observer_result.get('narrative', '')[:120]}")

        mined = PatternMiner.mine(snapshot)
        self.logger.info(f"PatternMiner: {len(mined)} candidates ({', '.join(str(p.get('name')) for p in mined)})")

        confirmed = self.pattern_analyst.analyze(
            market_state=snapshot,
            mined_patterns=mined,
            observer_result=observer_result
        )
        self.logger.info(f"PatternAnalyst: {len(confirmed)} confirmed patterns")

        candidates = self.trade_designer.synthesize(
            market_state=snapshot,
            confirmed_patterns=confirmed,
            observer_result=observer_result
        )
        self.logger.info(f"TradeDesigner: {len(candidates)} candidates generated")

        accepted = []
        for c in candidates:
            reviewed = self.skeptic.review(c, snapshot)
            if reviewed:
                accepted.append(reviewed)
        
        self.logger.info(f"Skeptic: {len(accepted)}/{len(candidates)} accepted")

        return accepted
