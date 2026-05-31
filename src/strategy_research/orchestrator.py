"""
Strategy Research Orchestrator
==============================
End-to-end pipeline: symbol -> data -> features -> labels -> AI discovery ->
strategy -> backtest -> validation.

Usage:
    orchestrator = ResearchOrchestrator(config)
    result = orchestrator.run()
    
    # Access results
    result.strategies  # List of validated Strategy objects
    result.metrics     # Performance summary
    result.to_json()   # Serialize
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from strategy_research.utils.config import ResearchConfig, AIGeneratorConfig
from strategy_research.data.binance_loader import (
    BinanceDataManager, fetch_and_align_mtf
)
from strategy_research.features.mtf_features import (
    build_mtf_features, add_regime_features
)
from strategy_research.labeling.triple_barrier import (
    LevBarrierConfig, triple_barrier_both_sides, create_target_vector
)
from strategy_research.ai_research.discovery import (
    StrategyDiscoveryAgent, discover_strategy
)
from strategy_research.strategies.generator import (
    StrategyGenerator, Strategy
)
from strategy_research.backtest.engine import (
    backtest_1m_strategy, evaluate_expectancy
)
from strategy_research.validation.walkforward import WalkForwardValidator

log = logging.getLogger("orchestrator")


@dataclass
class ResearchResult:
    """Complete result of a research run."""
    symbol: str
    config: ResearchConfig
    strategies: List[Strategy]
    discovery_long: Optional[Dict] = None
    discovery_short: Optional[Dict] = None
    walkforward_summary: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def best_strategy(self) -> Optional[Strategy]:
        """Return the best performing strategy."""
        if not self.strategies:
            return None
        return max(
            self.strategies,
            key=lambda s: s.metrics.get("expectancy", 0) * s.metrics.get("trade_count", 0)
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            f"STRATEGY RESEARCH RESULT: {self.symbol}",
            "=" * 70,
            f"Target: +{self.config.barrier.up_pct*100:.1f}% price "
            f"(={self.config.target_margin_return*100:.0f}% on "
            f"{self.config.barrier.leverage}x margin)",
            f"Data: {self.metadata.get('n_bars', 0)} 1m bars "
            f"({self.metadata.get('days', 0)} days)",
            f"Features: {self.metadata.get('n_features', 0)} MTF features",
            "",
            f"DISCOVERED STRATEGIES: {len(self.strategies)}",
            "-" * 70,
        ]

        for i, s in enumerate(self.strategies[:5], 1):
            m = s.metrics
            viable = m.get("expectancy", 0) > 0 and m.get("trade_count", 0) >= 30
            status = "TRADEABLE" if viable else "MARGINAL"
            lines.append(
                f"  {i}. {s.name} [{status}]"
            )
            lines.append(
                f"     PF={m.get('profit_factor', 0):.2f} | "
                f"WR={m.get('win_rate', 0)*100:.1f}% | "
                f"E={m.get('expectancy', 0):.4f} | "
                f"Trades={m.get('trade_count', 0)} | "
                f"MaxDD={m.get('max_drawdown', 0)*100:.1f}%"
            )
            lines.append(f"     Rule: {s.description[:80]}")
            lines.append("")

        # Walk-forward validation
        if self.walkforward_summary:
            wf = self.walkforward_summary
            lines.extend([
                "-" * 70,
                "WALK-FORWARD VALIDATION:",
                f"  Folds: {wf.get('folds', 0)}",
                f"  Avg Test AUC: {wf.get('avg_test_auc', 0):.3f}",
                f"  Stability: {wf.get('stability', 0):.1%}",
                f"  Degradation: {wf.get('degradation', 0):.3f}",
            ])

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize results to JSON."""
        data = {
            "symbol": self.symbol,
            "config": {
                "up_pct": self.config.barrier.up_pct,
                "dn_pct": self.config.barrier.dn_pct,
                "leverage": self.config.barrier.leverage,
                "target_margin_return": self.config.target_margin_return,
            },
            "strategies": [s.to_dict() for s in self.strategies],
            "walkforward": self.walkforward_summary,
            "metadata": self.metadata,
        }

        json_str = json.dumps(data, indent=2, default=str)

        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json_str)
            log.info(f"Results saved to {path}")

        return json_str


class ResearchOrchestrator:
    """Main orchestrator for the strategy research pipeline."""

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        self.result: Optional[ResearchResult] = None

    def run(self) -> ResearchResult:
        """Run the complete research pipeline."""
        cfg = self.config
        start_time = time.time()

        log.info("=" * 70)
        log.info(f"STRATEGY RESEARCH: {cfg.symbol}")
        log.info(f"Target: +{cfg.barrier.up_pct*100:.1f}% price move")
        log.info(f"Leverage: {cfg.barrier.leverage}x margin")
        log.info("=" * 70)

        # Step 1: Fetch data
        log.info("STEP 1: Fetching multi-timeframe data...")
        data = self._fetch_data()
        if not data:
            raise RuntimeError("Failed to fetch data")

        # Step 2: Build features
        log.info("STEP 2: Building leakage-safe MTF features...")
        features = self._build_features(data)

        # Step 3: Create labels
        log.info("STEP 3: Triple-barrier labeling...")
        labels = self._create_labels(data)

        # Step 4: AI Discovery
        log.info("STEP 4: AI strategy discovery...")
        discovery_long, discovery_short = self._discover(features, labels)

        # Step 5: Walk-forward validation
        log.info("STEP 5: Walk-forward validation...")
        wf_summary = self._validate_walkforward(features, labels)

        # Step 6: Generate and backtest strategies
        log.info("STEP 6: Generating strategies...")
        strategies = self._generate_strategies(
            discovery_long, discovery_short, features, labels, data
        )

        elapsed = time.time() - start_time
        log.info(f"Research complete in {elapsed:.0f}s")

        # Build result
        self.result = ResearchResult(
            symbol=cfg.symbol,
            config=cfg,
            strategies=strategies,
            discovery_long=discovery_long if discovery_long else None,
            discovery_short=discovery_short if discovery_short else None,
            walkforward_summary=wf_summary,
            metadata={
                "n_bars": len(data.get("1m", pd.DataFrame())),
                "n_features": len(features.columns),
                "days": cfg.days,
                "elapsed_seconds": elapsed,
            },
        )

        log.info("\n" + self.result.summary())

        return self.result

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch multi-timeframe OHLCV data."""
        cfg = self.config

        manager = BinanceDataManager(
            data_config=cfg.data,
            tf_config=cfg.timeframes,
        )

        try:
            data = manager.fetch_symbol(
                cfg.symbol,
                days=cfg.days,
                timeframes=cfg.timeframes.all_tfs,
            )
        except Exception as e:
            log.error(f"Data fetch failed: {e}")
            # Try with smaller request
            log.info("Retrying with 30 days...")
            data = manager.fetch_symbol(
                cfg.symbol,
                days=30,
                timeframes=["1m", "15m", "1h", "4h"],
            )

        # Fetch funding config
        try:
            avg_rate, interval = manager.get_funding_config(cfg.symbol)
            self.config.barrier.funding_rate = avg_rate
            self.config.barrier.funding_interval_h = interval
        except Exception as e:
            log.warning(f"Could not fetch funding config: {e}")

        return data

    def _build_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build MTF features."""
        features = build_mtf_features(
            data,
            cfg=self.config.features,
            base_tf="1m",
        )
        features = add_regime_features(features)
        return features

    def _create_labels(self, data: Dict[str, pd.DataFrame]):
        """Create triple-barrier labels."""
        if "1m" not in data:
            raise ValueError("No 1m data for labeling")

        barrier_cfg = LevBarrierConfig(
            up_pct=self.config.barrier.up_pct,
            dn_pct=self.config.barrier.dn_pct,
            max_horizon=self.config.barrier.max_horizon,
            round_trip_cost=self.config.barrier.round_trip_cost,
            leverage=self.config.barrier.leverage,
            maint_margin_rate=self.config.barrier.maint_margin_rate,
            funding_rate=self.config.barrier.funding_rate,
            funding_interval_h=self.config.barrier.funding_interval_h,
        )

        labels = triple_barrier_both_sides(data["1m"], barrier_cfg)
        return labels

    def _discover(self, features, labels):
        """Run AI discovery for both long and short."""
        # Long discovery
        try:
            long_labels = create_target_vector(features, labels, "best_side")
            discovery_long = discover_strategy(
                features, long_labels,
                config=self.config.ai,
                target_name="long",
            )
        except Exception as e:
            log.error(f"Long discovery failed: {e}")
            discovery_long = None

        # Short discovery
        try:
            short_labels = create_target_vector(features, labels, "short_label")
            discovery_short = discover_strategy(
                features, short_labels,
                config=self.config.ai,
                target_name="short",
            )
        except Exception as e:
            log.error(f"Short discovery failed: {e}")
            discovery_short = None

        return discovery_long, discovery_short

    def _validate_walkforward(self, features, labels):
        """Run walk-forward validation."""
        try:
            X = features.select_dtypes(include=[np.number]).fillna(0)
            y = (labels.set_index("entry_idx")["best_side"] > 0).astype(int)

            # Align
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]

            validator = WalkForwardValidator(
                n_folds=self.config.walkforward.n_folds,
                embargo=self.config.walkforward.embargo,
            )

            def model_factory():
                return xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                    random_state=42,
                )

            summary = validator.validate(X, y, model_factory)
            return summary

        except Exception as e:
            log.error(f"Walk-forward validation failed: {e}")
            return None

    def _generate_strategies(
        self,
        discovery_long,
        discovery_short,
        features,
        labels,
        data,
    ) -> List[Strategy]:
        """Generate and backtest strategies."""
        generator = StrategyGenerator(
            up_pct=self.config.barrier.up_pct,
            dn_pct=self.config.barrier.dn_pct,
            leverage=self.config.barrier.leverage,
        )

        all_strategies = []

        # Long strategies
        if discovery_long and discovery_long.get("rules"):
            long_strats = generator.from_discovery_result(discovery_long, side="long")
            for strat in long_strats:
                probs = discovery_long["agent"].predict_proba(features)
                signals = generator.generate_signals(strat, features, probs)
                if signals:
                    result = backtest_1m_strategy(
                        data["1m"], signals,
                        up_pct=self.config.barrier.up_pct,
                        dn_pct=self.config.barrier.dn_pct,
                        leverage=self.config.barrier.leverage,
                    )
                    viable, reason = evaluate_expectancy(
                        result,
                        min_trades=self.config.strategy.min_trades,
                        min_pf=self.config.strategy.min_profit_factor,
                    )
                    strat.metrics = result.metrics
                    strat.metadata["viable"] = viable
                    strat.metadata["eval_reason"] = reason
                    all_strategies.append(strat)

        # Short strategies
        if discovery_short and discovery_short.get("rules"):
            short_strats = generator.from_discovery_result(discovery_short, side="short")
            for strat in short_strats:
                probs = discovery_short["agent"].predict_proba(features)
                signals = generator.generate_signals(strat, features, probs, confidence_threshold=0.5)
                if signals:
                    # Flip side for short
                    for sig in signals:
                        sig.side = -1

                    result = backtest_1m_strategy(
                        data["1m"], signals,
                        up_pct=self.config.barrier.up_pct,
                        dn_pct=self.config.barrier.dn_pct,
                        leverage=self.config.barrier.leverage,
                    )
                    viable, reason = evaluate_expectancy(
                        result,
                        min_trades=self.config.strategy.min_trades,
                        min_pf=self.config.strategy.min_profit_factor,
                    )
                    strat.metrics = result.metrics
                    strat.metadata["viable"] = viable
                    strat.metadata["eval_reason"] = reason
                    all_strategies.append(strat)

        # Sort by expectancy * trade_count
        all_strategies.sort(
            key=lambda s: s.metrics.get("expectancy", 0) * s.metrics.get("trade_count", 0),
            reverse=True,
        )

        return all_strategies


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered Strategy Research for Binance Futures"
    )
    parser.add_argument("--symbol", default="SOLUSDT", help="Trading pair")
    parser.add_argument("--days", type=int, default=60, help="Days of history")
    parser.add_argument("--leverage", type=float, default=10.0, help="Leverage")
    parser.add_argument("--target", type=float, default=0.01, help="Target price move")
    parser.add_argument("--stop", type=float, default=0.005, help="Stop loss")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = ResearchConfig(
        symbol=args.symbol,
        days=args.days,
        barrier=LevBarrierConfig(
            up_pct=args.target,
            dn_pct=args.stop,
            leverage=args.leverage,
        ),
    )

    orchestrator = ResearchOrchestrator(config)
    result = orchestrator.run()

    if args.output:
        result.to_json(args.output)

    return result


if __name__ == "__main__":
    main()
