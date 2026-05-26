import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from ..configuration import Configuration

class MarkdownExporter:
    @staticmethod
    def export(strategy_card: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        cfg = Configuration()
        output_dir = output_dir or os.path.join(cfg.output_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{strategy_card.get('id', 'unnamed')}.md"
        path = os.path.join(output_dir, filename)

        content = MarkdownExporter.render(strategy_card)
        with open(path, 'w') as f:
            f.write(content)
        
        logging.getLogger(__name__).info(f"Exported strategy card: {path}")
        return path

    @staticmethod
    def render(card: Dict[str, Any]) -> str:
        score = card.get("ranking_score")
        score_s = f"{round(score, 3)}" if score is not None else "N/A"
        
        checklist = "\n".join([f"{i+1}. {c}" for i, c in enumerate(card.get("entry_checklist", []))])
        
        perf = card.get("performance", {})
        wr = perf.get("win_rate")
        wr_s = f"{round(wr * 100, 1)}%" if wr is not None else "N/A"

        timeframes = ", ".join(card.get('best_conditions', {}).get('timeframes', []))
        sessions = ", ".join(card.get('best_conditions', {}).get('sessions', []))
        regimes = ", ".join(card.get('best_conditions', {}).get('regimes', []))
        
        targets_r = ", ".join(map(str, card.get('exit_plan', {}).get('targets_r', [])))
        partial_exits = ", ".join(map(str, card.get('exit_plan', {}).get('partial_exits', [])))
        
        invalidation_rules = "\n".join([f"- {r}" for r in card.get('invalidation', [])])
        failure_modes = "\n".join([f"- {f}" for f in card.get('failure_modes', [])])
        parameter_bounds = "\n".join([f"- **{k}:** {v}" for k, v in card.get('parameter_bounds', {}).items()])

        exp = round(perf.get('expectancy', 0), 4) if perf.get('expectancy') is not None else 'N/A'
        pf = round(perf.get('profit_factor', 0), 2) if perf.get('profit_factor') is not None else 'N/A'
        max_dd = round(perf.get('max_drawdown', 0), 4) if perf.get('max_drawdown') is not None else 'N/A'
        avg_r = round(perf.get('avg_r', 0), 2) if perf.get('avg_r') is not None else 'N/A'
        sharpe = round(perf.get('sharpe', 0), 2) if perf.get('sharpe') is not None else 'N/A'

        return f"""# Strategy Card: {card.get('name')}

**Family:** {card.get('family')}
**Status:** {card.get('status')}
**Score:** {score_s}

## Summary

{card.get('summary')}

## Edge Explanation

{card.get('edge_explanation')}

## Best Conditions

- **Timeframes:** {timeframes}
- **Sessions:** {sessions}
- **Regimes:** {regimes}

## Entry Checklist

{checklist}

## Exit Plan

- **Targets (R):** {targets_r}
- **Partial Exits:** {partial_exits}
- **Trail:** {card.get('exit_plan', {}).get('trail')}
- **Time Stop:** {card.get('exit_plan', {}).get('time_stop') or 'None'}

## Risk Model

- **Stop:** {card.get('risk_model', {}).get('stop')}
- **Sizing:** {card.get('risk_model', {}).get('sizing')}
- **Max Risk:** {card.get('risk_model', {}).get('max_risk')}%

## Performance

| Metric | Value |
|--------|-------|
| Expectancy | {exp} |
| Win Rate | {wr_s} |
| Profit Factor | {pf} |
| Max Drawdown | {max_dd} |
| Avg R | {avg_r} |
| Trade Count | {perf.get('trade_count', 0)} |
| Sharpe | {sharpe} |

## Invalidation Rules

{invalidation_rules}

## Failure Modes

{failure_modes}

## Parameter Bounds

{parameter_bounds}

---
*Generated: {card.get('updated_at') or datetime.now(timezone.utc).isoformat()}*
"""
