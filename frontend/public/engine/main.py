#!/usr/bin/env python3
"""
Strategy Research Runner - Multi-RR Grid Search
Usage: python main.py --symbol SOLUSDT --days 60 --output result.json
"""
import argparse
import os
import sys
# Insert the engine directory so sibling modules resolve regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_symbol_mtf
from grid_search import run_grid_search


def main():
    parser = argparse.ArgumentParser(description="Multi-RR Strategy Research")
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--cost", type=float, default=0.0009)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--output", default="research_result.json")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} data ({args.days} days)...")
    data = fetch_symbol_mtf(args.symbol, days=args.days)

    if "1m" not in data or len(data["1m"]) < 1000:
        print("ERROR: Insufficient 1m data")
        return 1

    run_grid_search(data, args.symbol, args.leverage, args.cost, args.horizon, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
