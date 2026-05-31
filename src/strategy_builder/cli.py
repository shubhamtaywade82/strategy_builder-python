import argparse
import sys
import os
import logging
from dotenv import load_dotenv
from .agent.agent_loop import AgentLoop
from .generation.strategy_catalog import StrategyCatalog
from .generation.strategy_templates import StrategyTemplates
from .configuration import Configuration

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Strategy Builder CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # research
    research_p = subparsers.add_parser("research", help="Run full strategy research pipeline")
    research_p.add_argument("query", help="Research query")
    research_p.add_argument("--instruments", nargs="+", help="Instruments to research")
    research_p.add_argument("--timeframes", nargs="+", help="Timeframes")
    research_p.add_argument("--days", type=int, default=30, help="Days of data")
    research_p.add_argument("--max-steps", type=int, default=20, help="Max iterations")

    # discover
    discover_p = subparsers.add_parser("discover", help="Discover market features")
    discover_p.add_argument("--instruments", nargs="+")
    discover_p.add_argument("--timeframes", nargs="+")
    discover_p.add_argument("--days", type=int, default=30)

    # propose
    propose_p = subparsers.add_parser("propose", help="Generate strategy candidates")
    propose_p.add_argument("--instruments", nargs="+")
    propose_p.add_argument("--timeframes", nargs="+")
    propose_p.add_argument("--days", type=int, default=30)
    propose_p.add_argument("--mode", choices=["generate", "mutate"], default="generate")

    # backtest
    backtest_p = subparsers.add_parser("backtest", help="Run backtests")
    backtest_p.add_argument("--days", type=int, default=90)

    # rank
    rank_p = subparsers.add_parser("rank", help="Rank strategies")

    # document
    document_p = subparsers.add_parser("document", help="Generate documentation")

    # catalog
    catalog_p = subparsers.add_parser("catalog", help="List catalog entries")
    catalog_p.add_argument("--status")
    catalog_p.add_argument("--format", choices=["text", "markdown"], default="text")

    # templates
    templates_p = subparsers.add_parser("templates", help="List templates")

    # pipeline
    pipeline_p = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_p.add_argument("query")
    pipeline_p.add_argument("--instruments", nargs="+")
    pipeline_p.add_argument("--timeframes", nargs="+")
    pipeline_p.add_argument("--days", type=int, default=30)
    pipeline_p.add_argument("--fresh", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cfg = Configuration()

    # Setup client (mock for now as in original if not configured)
    from .ollama.ssl_bearer_client import OllamaSslBearerClient
    client = OllamaSslBearerClient(cfg, bearer_token=cfg.ollama_api_key)

    agent = AgentLoop(client=client)

    if args.command == "research":
        if args.instruments: cfg.default_instruments = args.instruments
        if args.timeframes: cfg.default_timeframes = args.timeframes
        res = agent.run(query=args.query)
        print("\n=== Research Complete ===")
        print(f"Steps: {len(res['steps'])}")
        print(f"Strategies found: {res['strategies_found']}")
        print(f"Passing: {res['passing_strategies']}")

    elif args.command == "discover":
        features = agent.discover(instruments=args.instruments, timeframes=args.timeframes, days_back=args.days)
        for inst, feat in features.items():
            print(f"\n=== {inst} ===")
            print(f"  MTF Alignment: {feat.get('mtf_alignment', {}).get('alignment', {}).get('regime')}")
            print(f"  Volatility: {feat.get('volatility', {}).get('regime')}")
            print(f"  Structure: {feat.get('structure', {}).get('structure')}")
            print(f"  RSI: {feat.get('momentum', {}).get('rsi_current')}")

    elif args.command == "propose":
        features = agent.discover(instruments=args.instruments, timeframes=args.timeframes, days_back=args.days)
        candidates = agent.propose(features_by_instrument=features, mode=args.mode)
        print("\n=== Proposed Strategies ===")
        for c in candidates:
            print(f"  [{c['id']}] {c['name']} for {c['instrument']}")

    elif args.command == "backtest":
        agent.validate(days_back=args.days)
        print("Backtest phase complete.")

    elif args.command == "rank":
        ranked = agent.rank()
        print("\n=== Strategy Rankings ===")
        for i, entry in enumerate(ranked):
            score = entry.get("ranking", {}).get("final_score")
            print(f"  {i+1}. {entry['strategy']['name']} — score: {score} — {entry['status']}")

    elif args.command == "document":
        agent.document()
        print("Documentation exported.")

    elif args.command == "catalog":
        cat = StrategyCatalog()
        entries = cat.by_status(args.status) if args.status else cat.all()
        
        if args.format == "markdown":
            print("# Strategy Catalog\n")
            print("| ID | Name | Instrument | Status | Score |")
            print("| :--- | :--- | :--- | :--- | :--- |")
            for e in entries:
                if not e: continue
                ranking = e.get("ranking") or {}
                score = ranking.get("final_score", "N/A")
                if isinstance(score, float): score = round(score, 2)
                
                strategy = e.get("strategy") or {}
                name = strategy.get("name", "Unknown")
                inst = strategy.get("instrument", "All")
                print(f"| {e['id']} | {name} | {inst} | {e['status']} | {score} |")
        else:
            print(f"\n=== Strategy Catalog ({len(entries)} entries) ===")
            for e in entries:
                if not e: continue
                ranking = e.get("ranking") or {}
                score = ranking.get("final_score", "N/A")
                strategy = e.get("strategy") or {}
                name = strategy.get("name", "Unknown")
                print(f"  [{e['id']}] {name} — {e['status']} — score: {score}")

    elif args.command == "templates":
        print("\n=== Strategy Templates ===")
        for t in StrategyTemplates.all():
            print(f"  {t['name']} ({t['family']}) — TFs: {'/'.join(t['timeframes'])}")

    elif args.command == "pipeline":
        if args.fresh:
            StrategyCatalog().clear()
            print("Cleared strategy catalog (--fresh).")

        print("Phase 1: Discovering features...")
        features = agent.discover(instruments=args.instruments, timeframes=args.timeframes, days_back=args.days)
        
        print("Phase 2: Proposing strategies...")
        candidates = agent.propose(features_by_instrument=features)
        
        print("Phase 3: Backtesting...")
        agent.validate(days_back=args.days * 3)
        
        print("Phase 4: Ranking...")
        ranked = agent.rank()
        
        print("Phase 5: Documenting...")
        agent.document
        
        print("\n=== Pipeline Complete ===")
        cat = StrategyCatalog()
        print(f"Total strategies: {cat.size()}")
        print(f"Passing: {len(cat.passing())}")

if __name__ == "__main__":
    main()
