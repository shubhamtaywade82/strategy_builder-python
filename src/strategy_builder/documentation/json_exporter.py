import os
import json
import logging
from typing import Dict, Any, List, Optional
from ..configuration import Configuration
from .strategy_card import StrategyCard

class JsonExporter:
    @staticmethod
    def export(strategy_card: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        cfg = Configuration()
        output_dir = output_dir or os.path.join(cfg.output_dir, "strategies")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{strategy_card.get('id', 'unnamed')}.json"
        path = os.path.join(output_dir, filename)

        with open(path, 'w') as f:
            json.dump(strategy_card, f, indent=2)
        
        logging.getLogger(__name__).info(f"Exported strategy JSON: {path}")
        return path

    @staticmethod
    def export_ranking_table(catalog: Any) -> List[Dict[str, Any]]:
        ranked = catalog.ranked()
        table = []
        for i, entry in enumerate(ranked):
            card = StrategyCard.build(entry)
            perf = card.get("performance", {})
            table.append({
                "rank": i + 1,
                "id": card["id"],
                "name": card["name"],
                "family": card["family"],
                "score": card["ranking_score"],
                "expectancy": perf.get("expectancy"),
                "profit_factor": perf.get("profit_factor"),
                "win_rate": perf.get("win_rate"),
                "trades": perf.get("trade_count"),
                "status": card["status"]
            })
        return table
