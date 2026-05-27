"""
Supertrend Multi-Symbol, Multi-Timeframe Parameter Researcher
=============================================================

Exhaustively searches the (symbol × timeframe × length × multiplier) grid,
validates each combination with in-sample backtesting and an OOS anchored-holdout
split, scores results, and reports the best fine-tuned setup.

Usage (CLI):
    python -m strategy_builder.strategies.supertrend_researcher \\
        --symbols BTCUSDT ETHUSDT \\
        --timeframes 15m 1h 4h \\
        --days 90 \\
        --lengths 7 10 14 \\
        --multipliers 2.0 3.0 3.5 4.0
"""

import argparse
import itertools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..backtest.engine import BacktestEngine
from ..backtest.walk_forward import WalkForward
from ..market_data.candle_loader import CandleLoader
from .supertrend import DEFAULT_LENGTHS, DEFAULT_MULTIPLIERS, build_signal_generator

logger = logging.getLogger("SupertrendResearcher")

# Strategy exit template used for all backtest runs
_EXIT_TEMPLATE = {
    "targets": [2.0, 4.0],
    "trail": "atr_2.0",
}


@dataclass
class SupertrendResult:
    symbol: str
    timeframe: str
    length: int
    multiplier: float
    is_profit_factor: float
    is_win_rate: float
    is_max_drawdown: float
    is_trade_count: int
    oos_expectancy: float
    oos_profit_factor: float
    oos_win_rate: float
    oos_trade_count: int
    score: float
    passes_holdout: bool


def _score(is_metrics: dict, oos_metrics: Optional[dict], base_price: float = 1.0) -> Tuple[float, bool]:
    """
    Composite score: profit_factor × win_rate × (1 - max_drawdown).

    Penalises configs with too few trades or negative OOS expectancy so that
    over-fit, lucky results sort below robust ones.
    """
    pf = is_metrics.get("profit_factor", 0.0)
    wr = is_metrics.get("win_rate", 0.0)
    dd = min(is_metrics.get("max_drawdown", 0.0) / base_price, 1.0)
    tc = is_metrics.get("trade_count", 0)

    base = pf * wr * (1.0 - dd)

    if tc < 10:
        base *= 0.5

    passes = False
    if oos_metrics:
        oos_exp = oos_metrics.get("expectancy", -1.0)
        if oos_exp < 0:
            base *= 0.5
        passes = oos_exp > 0 and oos_metrics.get("profit_factor", 0.0) > 1.0

    return round(base, 6), passes


class SupertrendResearcher:
    """
    Multi-symbol × multi-timeframe × parameter grid search for the Supertrend strategy.

    For every (symbol, timeframe, length, multiplier) combination:
      1. Fetches historical candles via CandleLoader (Binance FAPI).
      2. Runs a full in-sample backtest with BacktestEngine.
      3. Runs an anchored holdout (70/30 IS/OOS) with WalkForward.
      4. Scores the combination and stores the result.
    Reports the ranked table and the top-3 globally recommended setups.

    Parameters
    ----------
    symbols : List[str]
        Binance futures symbols, e.g. ["BTCUSDT", "ETHUSDT"].
    timeframes : List[str]
        Timeframe strings, e.g. ["15m", "1h", "4h"].
    lengths : List[int]
        ATR periods to search. Default: [7, 10, 14].
    multipliers : List[float]
        ATR multipliers to search. Default: [2.0, 3.0, 3.5, 4.0].
    lookback_days : int
        Days of history to fetch. Default: 90.
    adaptive_regime : bool
        Whether to enable regime-adaptive multiplier scaling. Default: False
        for the researcher (keeps param comparison fair). Set True in live trading.
    market_data_source : str
        "binance" or "coindcx". Default: "binance".
    """

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        lengths: Optional[List[int]] = None,
        multipliers: Optional[List[float]] = None,
        lookback_days: int = 90,
        adaptive_regime: bool = False,
        market_data_source: str = "binance",
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.timeframes = timeframes
        self.lengths = lengths or DEFAULT_LENGTHS
        self.multipliers = multipliers or DEFAULT_MULTIPLIERS
        self.lookback_days = lookback_days
        self.adaptive_regime = adaptive_regime

        self.loader = CandleLoader(market_data_source=market_data_source)
        self.engine = BacktestEngine()
        self.wf = WalkForward(engine=self.engine)

        self._results: List[SupertrendResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[SupertrendResult]:
        """
        Execute the full grid search. Returns all results sorted by score (desc).
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        total_combos = (
            len(self.symbols)
            * len(self.timeframes)
            * len(self.lengths)
            * len(self.multipliers)
        )
        logger.info(
            f"Starting Supertrend research: {len(self.symbols)} symbols × "
            f"{len(self.timeframes)} timeframes × "
            f"{len(self.lengths) * len(self.multipliers)} param combos = "
            f"{total_combos} total backtests"
        )

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                logger.info(f"Fetching {symbol} {timeframe} ({self.lookback_days}d)...")
                try:
                    candles = self.loader.fetch(symbol, timeframe, start_date, end_date)
                    time.sleep(0.3)  # Binance rate-limit courtesy delay
                except Exception as exc:
                    logger.error(f"Failed to fetch {symbol} {timeframe}: {exc}")
                    continue

                if len(candles) < 200:
                    logger.warning(f"Too few candles ({len(candles)}) for {symbol} {timeframe}. Skipping.")
                    continue

                logger.info(f"  Loaded {len(candles)} candles. Running grid search...")
                self._grid_search(symbol, timeframe, candles)

        self._results.sort(key=lambda r: r.score, reverse=True)
        self._report()
        return self._results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grid_search(self, symbol: str, timeframe: str, candles: list) -> None:
        base_price = float(candles[0].close) if candles and len(candles) > 0 and float(candles[0].close) > 0 else 1.0
        for length, multiplier in itertools.product(self.lengths, self.multipliers):
            strategy_dict = {
                "name": f"ST_{symbol}_{timeframe}_L{length}_M{multiplier}",
                "exit": _EXIT_TEMPLATE,
            }
            sig_gen = build_signal_generator(
                length=length,
                multiplier=multiplier,
                adaptive_regime=self.adaptive_regime,
            )

            # In-sample backtest
            try:
                is_result = self.engine.run(strategy_dict, candles, sig_gen)
                is_metrics = is_result["metrics"]
            except Exception as exc:
                logger.debug(f"IS backtest failed {symbol} {timeframe} L{length} M{multiplier}: {exc}")
                continue

            # OOS anchored holdout
            oos_metrics = None
            try:
                holdout = self.wf.anchored_holdout(strategy_dict, candles, sig_gen)
                if holdout:
                    oos_metrics = holdout["out_of_sample"]
            except Exception as exc:
                logger.debug(f"OOS holdout failed {symbol} {timeframe} L{length} M{multiplier}: {exc}")

            score, passes = _score(is_metrics, oos_metrics, base_price)

            result = SupertrendResult(
                symbol=symbol,
                timeframe=timeframe,
                length=length,
                multiplier=multiplier,
                is_profit_factor=round(is_metrics.get("profit_factor", 0.0), 3),
                is_win_rate=round(is_metrics.get("win_rate", 0.0), 3),
                is_max_drawdown=round(is_metrics.get("max_drawdown", 0.0) / base_price, 3),
                is_trade_count=is_metrics.get("trade_count", 0),
                oos_expectancy=round((oos_metrics or {}).get("expectancy", 0.0), 4),
                oos_profit_factor=round((oos_metrics or {}).get("profit_factor", 0.0), 3),
                oos_win_rate=round((oos_metrics or {}).get("win_rate", 0.0), 3),
                oos_trade_count=(oos_metrics or {}).get("trade_count", 0),
                score=score,
                passes_holdout=passes,
            )
            self._results.append(result)

            logger.debug(
                f"  {symbol} {timeframe} L{length} M{multiplier}: "
                f"PF={result.is_profit_factor:.2f} WR={result.is_win_rate:.1%} "
                f"OOS_exp={result.oos_expectancy:.4f} score={score:.4f}"
            )

    def _report(self) -> None:
        sep = "=" * 100
        print(f"\n{sep}")
        print("SUPERTREND PARAMETER RESEARCH REPORT")
        print(sep)

        header = (
            f"{'Symbol':<10} {'TF':<5} {'L':>3} {'M':>5} "
            f"{'Score':>8} {'WR%':>6} {'PF':>6} {'MaxDD':>7} "
            f"{'Trades':>6} {'OOS_Exp':>8} {'OOS_PF':>7} {'Pass':>5}"
        )
        print(header)
        print("-" * 100)

        for r in self._results[:30]:  # top 30 rows
            flag = "YES" if r.passes_holdout else "no"
            print(
                f"{r.symbol:<10} {r.timeframe:<5} {r.length:>3} {r.multiplier:>5.1f} "
                f"{r.score:>8.4f} {r.is_win_rate:>5.1%} {r.is_profit_factor:>6.2f} "
                f"{r.is_max_drawdown:>6.1%} {r.is_trade_count:>6} "
                f"{r.oos_expectancy:>8.4f} {r.oos_profit_factor:>7.2f} {flag:>5}"
            )

        # Best per symbol × timeframe
        best_map: Dict[str, SupertrendResult] = {}
        for r in self._results:
            key = f"{r.symbol}_{r.timeframe}"
            if key not in best_map or r.score > best_map[key].score:
                best_map[key] = r

        print(f"\n{'*' * 100}")
        print("RECOMMENDED SETUP PER SYMBOL × TIMEFRAME")
        print("*" * 100)
        for key, r in sorted(best_map.items()):
            print(
                f"  {r.symbol} / {r.timeframe}  →  "
                f"length={r.length}  multiplier={r.multiplier}  "
                f"score={r.score:.4f}  OOS_exp={r.oos_expectancy:.4f}  "
                f"passes_holdout={r.passes_holdout}"
            )

        # Global top-3
        print(f"\n{'=' * 100}")
        print("TOP-3 GLOBAL CONFIGURATIONS")
        print("=" * 100)
        top3 = [r for r in self._results if r.passes_holdout][:3] or self._results[:3]
        for rank, r in enumerate(top3, 1):
            print(
                f"  #{rank}: {r.symbol} {r.timeframe}  length={r.length}  "
                f"multiplier={r.multiplier}  score={r.score:.4f}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Supertrend multi-symbol/timeframe parameter researcher"
    )
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Binance symbols")
    p.add_argument("--timeframes", nargs="+", default=["15m", "1h"], help="Timeframe strings")
    p.add_argument("--days", type=int, default=90, help="Lookback days")
    p.add_argument(
        "--lengths", nargs="+", type=int, default=[7, 10, 14], help="ATR periods"
    )
    p.add_argument(
        "--multipliers",
        nargs="+",
        type=float,
        default=[2.0, 3.0, 3.5, 4.0],
        help="ATR multipliers",
    )
    p.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Enable regime-adaptive multiplier scaling",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = _parse_args()
    researcher = SupertrendResearcher(
        symbols=args.symbols,
        timeframes=args.timeframes,
        lengths=args.lengths,
        multipliers=args.multipliers,
        lookback_days=args.days,
        adaptive_regime=args.adaptive,
    )
    researcher.run()
