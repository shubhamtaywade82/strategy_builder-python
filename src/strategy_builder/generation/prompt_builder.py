import os
import json
import random
from typing import List, Dict, Any, Optional
from ..backtest.condition_registry import ConditionRegistry
from .strategy_templates import StrategyTemplates

class PromptBuilder:
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "agent", "schemas", "strategy_candidate.json")

    SYSTEM_PROMPT_STATIC = (
        "You are a quantitative strategy researcher for cryptocurrency futures markets.\n"
        "Your role is to propose trading strategy candidates based on computed market features.\n\n"
        "RULES:\n"
        "1. You MUST output valid JSON matching the provided schema. No prose, no markdown, no explanation outside the JSON.\n"
        "2. Every strategy MUST have concrete, testable entry conditions — not vague descriptions.\n"
        "3. Every strategy MUST have explicit stop-loss logic, position sizing, and exit targets.\n"
        "4. You MUST NOT invent indicators or features not present in the provided feature data.\n"
        "5. You MUST specify parameter_ranges for every tunable parameter so the backtester can optimize.\n"
        "6. Strategies must be realistic for crypto futures: consider 24/7 markets, session-based liquidity, and fee impact.\n"
        "7. Prefer strategies with clear invalidation conditions — a strategy that cannot be invalidated is not a strategy.\n"
        "8. When mutating templates, change at most 3 parameters or conditions at a time. Do not reinvent the template.\n"
        "9. exit.partial_exits MUST have the same length as exit.targets, each value in (0,1], and the values MUST sum to exactly 1.0.\n"
        "10. entry.conditions MUST be a JSON array of strings where each string is EXACTLY one identifier from the ALLOWED_ENTRY_CONDITION_IDS list appended below.\n\n"
        "STRATEGY FAMILIES YOU MAY USE:\n"
        "- session_breakout: Trade breakouts of session ranges (Asia, London, NY)\n"
        "- session_mean_reversion: Fade moves at session extremes\n"
        "- mtf_pullback: Enter on lower-TF pullback within higher-TF trend\n"
        "- compression_breakout: Enter on volatility expansion after compression\n"
        "- failed_breakout: Fade false breakouts with structure confirmation\n"
        "- vwap_reclaim: Enter on VWAP/MA reclaim with structure shift\n"
        "- volume_continuation: Enter on volume burst continuation\n"
        "- atr_expansion: Enter on ATR expansion follow-through\n"
        "- structure_shift: Enter on market structure shift (MSS)\n"
        "- custom: Novel combination"
    )

    @staticmethod
    def system_prompt() -> str:
        return f"{PromptBuilder.SYSTEM_PROMPT_STATIC}\n\n{PromptBuilder.entry_conditions_contract_appendix()}"

    @staticmethod
    def entry_conditions_contract_appendix() -> str:
        ids = ConditionRegistry.condition_ids()
        return (
            f"\nALLOWED_ENTRY_CONDITION_IDS (comma-separated; each entry.conditions[] value must be one of these, verbatim):\n"
            f"{', '.join(ids)}\n\n"
            f"EXAMPLE_VALID_BUNDLES_BY_FAMILY (copy identifiers only from the list above):\n"
            f"{PromptBuilder.example_condition_bundles_by_family()}"
        )

    @staticmethod
    def example_condition_bundles_by_family() -> str:
        seen = {}
        lines = []
        for t in StrategyTemplates.all():
            fam = str(t["family"])
            if fam in seen:
                continue

            conds = t.get("entry", {}).get("conditions")
            if not conds:
                continue

            seen[fam] = True
            lines.append(f"- {fam}: {', '.join(conds)}")
        return "\n".join(lines)

    @staticmethod
    def entry_conditions_snippet_for_task_prompt() -> str:
        return (
            f"\nALLOWED_ENTRY_CONDITION_IDS:\n"
            f"{', '.join(ConditionRegistry.condition_ids())}\n\n"
            f"EXAMPLE_VALID_BUNDLES_BY_FAMILY:\n"
            f"{PromptBuilder.example_condition_bundles_by_family()}"
        )

    @staticmethod
    def strategy_candidates_generate_schema() -> Dict[str, Any]:
        ids = ConditionRegistry.condition_ids()
        return {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "required": ["name", "family", "timeframes", "entry", "exit", "risk"],
                "additionalProperties": True,
                "properties": {
                    "entry": {
                        "type": "object",
                        "required": ["conditions"],
                        "additionalProperties": True,
                        "properties": {
                            "conditions": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 8,
                                "items": {"type": "string", "enum": ids}
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["long", "short", "both"]
                            }
                        }
                    }
                }
            }
        }

    @staticmethod
    def generation_prompt(features: Dict[str, Any], templates: List[Dict[str, Any]], mode: str = "generate") -> str:
        if mode == "generate":
            return PromptBuilder.new_strategy_prompt(features, templates)
        elif mode == "mutate":
            return PromptBuilder.mutation_prompt(features, templates)
        elif mode == "critique":
            return PromptBuilder.critique_prompt(features, templates)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def new_strategy_prompt(features: Dict[str, Any], templates: List[Dict[str, Any]]) -> str:
        t_list = "\n".join([f"- {t['name']} ({t['family']})" for t in templates])
        
        with open(PromptBuilder.SCHEMA_PATH, 'r') as f:
            schema_text = f.read()

        return (
            f"TASK: Analyze the following market features and propose 3 new strategy candidates.\n\n"
            f"MARKET FEATURES:\n{json.dumps(features, indent=2)}\n\n"
            f"EXISTING TEMPLATE FAMILIES:\n{t_list}\n"
            f"{PromptBuilder.entry_conditions_snippet_for_task_prompt()}\n"
            "INSTRUCTIONS:\n"
            "1. Study the feature data.\n"
            "2. Identify edges.\n"
            "3. Construct 3 distinct strategy candidates.\n"
            "4. Return exactly 3 strategy objects in a JSON array.\n\n"
            f"OUTPUT FORMAT:\nReturn ONLY a JSON array matching this schema:\n{schema_text}"
        )

    @staticmethod
    def mutation_prompt(features: Dict[str, Any], templates: List[Dict[str, Any]]) -> str:
        template = random.choice(templates) if templates else {}
        
        return (
            f"TASK: Mutate this existing strategy template to improve it for the current market conditions.\n\n"
            f"TEMPLATE TO MUTATE:\n{json.dumps(template, indent=2)}\n\n"
            f"CURRENT MARKET FEATURES:\n{json.dumps(features, indent=2)}\n"
            f"{PromptBuilder.entry_conditions_snippet_for_task_prompt()}\n"
            "INSTRUCTIONS:\n"
            "1. Analyze alignment/divergence.\n"
            "2. Propose exactly 2 mutations (Conservative and Aggressive).\n"
            "3. Return ONLY a JSON array of exactly 2 strategy objects."
        )

    @staticmethod
    def critique_prompt(features: Dict[str, Any], templates: List[Dict[str, Any]]) -> str:
        return (
            f"TASK: Critique these strategy candidates against the current market features.\n\n"
            f"STRATEGIES TO CRITIQUE:\n{json.dumps(templates, indent=2)}\n\n"
            f"CURRENT MARKET FEATURES:\n{json.dumps(features, indent=2)}\n\n"
            "INSTRUCTIONS:\nEvaluate entry logic, stop realism, target achievability, and filters.\n\n"
            "OUTPUT FORMAT:\nReturn ONLY a JSON array of objects: {strategy_name, score, issues, improvements, viable}."
        )

    @staticmethod
    def documentation_prompt(strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> str:
        return (
            f"TASK: Write a strategy card document for this strategy based on its backtest results.\n\n"
            f"STRATEGY:\n{json.dumps(strategy, indent=2)}\n\n"
            f"BACKTEST RESULTS:\n{json.dumps(backtest_results, indent=2)}\n\n"
            "INSTRUCTIONS:\nWrite a concise strategy card covering edge, best conditions, checklists, rules, risk, performance, failure modes.\n\n"
            "OUTPUT FORMAT:\nReturn ONLY a JSON object: {title, summary, edge_explanation, best_conditions, entry_checklist, invalidation_rules, risk_summary, performance_summary, failure_modes, tuning_notes}."
        )
