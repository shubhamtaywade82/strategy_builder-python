#!/usr/bin/env python3
"""
Strategy Research Runner
========================
Main entry point for AI-powered strategy discovery.

Usage:
    # Research SOLUSDT for 1% moves
    python run_research.py --symbol SOLUSDT --days 60

    # Research BTCUSDT with custom target
    python run_research.py --symbol BTCUSDT --days 90 --target 0.01 --stop 0.005

    # Research ETHUSDT, save results
    python run_research.py --symbol ETHUSDT --days 60 --output results/eth.json

    # Verbose mode for debugging
    python run_research.py --symbol SOLUSDT --days 30 --verbose
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from strategy_research.orchestrator import ResearchOrchestrator
from strategy_research.utils.config import ResearchConfig
from strategy_research.labeling.triple_barrier import LevBarrierConfig


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Silence noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Strategy Research for Binance USD-M Futures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --symbol SOLUSDT --days 60
  %(prog)s --symbol BTCUSDT --days 90 --target 0.01 --leverage 10
  %(prog)s --symbol ETHUSDT --days 60 --output results/eth.json --verbose
        """
    )

    parser.add_argument(
        "--symbol", "-s",
        default="SOLUSDT",
        help="Trading pair symbol (default: SOLUSDT)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Days of historical data to fetch (default: 60)"
    )
    parser.add_argument(
        "--target", "-t",
        type=float,
        default=0.01,
        help="Target price move, e.g. 0.01 = 1%% (default: 0.01)"
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=0.005,
        help="Stop loss, e.g. 0.005 = 0.5%% (default: 0.005)"
    )
    parser.add_argument(
        "--leverage", "-l",
        type=float,
        default=10.0,
        help="Leverage multiplier (default: 10)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="Max holding horizon in 1m bars (default: 120 = 2h)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Walk-forward folds (default: 5)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Build config
    barrier_cfg = LevBarrierConfig(
        up_pct=args.target,
        dn_pct=args.stop,
        leverage=args.leverage,
        max_horizon=args.horizon,
    )

    config = ResearchConfig(
        symbol=args.symbol,
        days=args.days,
        barrier=barrier_cfg,
        walkforward={"n_folds": args.folds, "embargo": args.horizon},
    )

    # Run
    orchestrator = ResearchOrchestrator(config)
    result = orchestrator.run()

    # Save if requested
    if args.output:
        result.to_json(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
