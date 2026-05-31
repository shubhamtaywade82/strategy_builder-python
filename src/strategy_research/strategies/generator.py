"""
Strategy Generator
=================
Converts AI-discovered rules into executable strategy objects.

A strategy here is a collection of entry conditions with:
- Signal generation (when to enter)
- Side determination (long/short)
- Risk parameters (stop, target, position size)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import numpy as np
import pandas as pd

from strategy_research.ai_research.discovery import StrategyRule, FeatureInsight

log = logging.getLogger("strategy_generator")


@dataclass
class EntrySignal:
    """A single entry signal at a specific bar."""
    timestamp: pd.Timestamp
    bar_idx: int
    side: int  # +1 long, -1 short
    confidence: float  # model probability
    price: float
    rule_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Strategy:
    """An executable trading strategy."""
    name: str
    description: str
    side: str  # "long", "short", or "both"
    entry_conditions: List[Dict[str, Any]]
    exit_config: Dict[str, Any]
    insights: List[FeatureInsight]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Generated signals (populated after backtest)
    signals: List[EntrySignal] = field(default_factory=list)

    # Performance metrics (populated after evaluation)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strategy to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "side": self.side,
            "entry_conditions": self.entry_conditions,
            "exit_config": self.exit_config,
            "metrics": self.metrics,
            "n_signals": len(self.signals),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        m = self.metrics
        return (f"Strategy({self.name}, side={self.side}, "
                f"PF={m.get('profit_factor', 0):.2f}, "
                f"WR={m.get('win_rate', 0)*100:.1f}%, "
                f"trades={m.get('trade_count', 0)})")


class StrategyGenerator:
    """Generates executable strategies from AI-discovered rules."""

    def __init__(
        self,
        up_pct: float = 0.01,
        dn_pct: float = 0.005,
        leverage: float = 10.0,
    ):
        self.up_pct = up_pct
        self.dn_pct = dn_pct
        self.leverage = leverage

    def from_discovery_result(
        self,
        discovery_result: Dict[str, Any],
        side: str = "long",
    ) -> List[Strategy]:
        """Generate strategies from AI discovery output.

        Args:
            discovery_result: Output from discover_strategy()
            side: 'long', 'short', or 'both'

        Returns:
            List of Strategy objects
        """
        rules = discovery_result.get("rules", [])
        if not rules:
            log.warning("No rules in discovery result")
            return []

        strategies = []
        for rule in rules:
            strat = self._rule_to_strategy(rule, side)
            if strat:
                strategies.append(strat)

        log.info(f"Generated {len(strategies)} strategies from {len(rules)} rules")
        return strategies

    def _rule_to_strategy(self, rule: StrategyRule, side: str) -> Optional[Strategy]:
        """Convert a StrategyRule to a Strategy."""
        if not rule.conditions:
            return None

        # Exit configuration based on target
        exit_config = {
            "target_pct": self.up_pct,
            "stop_pct": self.dn_pct,
            "leverage": self.leverage,
            "time_stop_bars": 120,  # 2 hours on 1m
        }

        return Strategy(
            name=rule.name,
            description=rule.description,
            side=side,
            entry_conditions=rule.conditions,
            exit_config=exit_config,
            insights=rule.feature_insights,
            metadata={
                "expected_win_rate": rule.expected_win_rate,
                "expected_trades_pct": rule.expected_trades_pct,
            },
        )

    def generate_signals(
        self,
        strategy: Strategy,
        features: pd.DataFrame,
        probabilities: Optional[np.ndarray] = None,
        confidence_threshold: float = 0.55,
    ) -> List[EntrySignal]:
        """Generate entry signals from a strategy on feature data.

        Args:
            strategy: Strategy with entry conditions
            features: Feature DataFrame
            probabilities: Model probabilities (optional)
            confidence_threshold: Minimum confidence to trigger

        Returns:
            List of EntrySignal objects
        """
        signals = []
        mask = pd.Series(True, index=features.index)

        # Apply all conditions (AND logic)
        for cond in strategy.entry_conditions:
            feat = cond["feature"]
            op = cond["operator"]
            thresh = cond["threshold"]

            if feat not in features.columns:
                log.warning(f"Feature {feat} not in data")
                return []

            if op == ">=":
                mask &= features[feat] >= thresh
            elif op == ">":
                mask &= features[feat] > thresh
            elif op == "<=":
                mask &= features[feat] <= thresh
            elif op == "<":
                mask &= features[feat] < thresh
            elif op == "==":
                mask &= features[feat] == thresh

        if mask.sum() == 0:
            return []

        # Determine side
        side_map = {"long": 1, "short": -1, "both": 1}
        side_val = side_map.get(strategy.side, 1)

        for idx in features[mask].index:
            conf = probabilities[idx] if probabilities is not None else 0.6
            if conf < confidence_threshold:
                continue

            price = features.loc[idx, "close"] if "close" in features.columns else 0

            sig = EntrySignal(
                timestamp=features.loc[idx, "open_time"] if "open_time" in features.columns else pd.Timestamp.now(),
                bar_idx=int(idx),
                side=side_val,
                confidence=float(conf),
                price=float(price),
                rule_name=strategy.name,
            )
            signals.append(sig)

        strategy.signals = signals
        log.info(f"Generated {len(signals)} signals for {strategy.name}")
        return signals

    def create_meta_strategy(
        self,
        long_strategies: List[Strategy],
        short_strategies: List[Strategy],
        features: pd.DataFrame,
        long_probs: np.ndarray,
        short_probs: np.ndarray,
        threshold: float = 0.55,
    ) -> Strategy:
        """Create a meta-strategy that picks best side per bar.

        Args:
            long_strategies: Long strategies
            short_strategies: Short strategies
            features: Feature data
            long_probs: Long win probabilities
            short_probs: Short win probabilities
            threshold: Min confidence to trade

        Returns:
            Meta Strategy
        """
        signals = []

        for i in range(len(features)):
            lp = long_probs[i] if i < len(long_probs) else 0
            sp = short_probs[i] if i < len(short_probs) else 0

            if max(lp, sp) < threshold:
                continue

            side = 1 if lp > sp else -1
            conf = max(lp, sp)

            sig = EntrySignal(
                timestamp=features.index[i] if isinstance(features.index[i], pd.Timestamp)
                       else features.loc[i, "open_time"],
                bar_idx=i,
                side=side,
                confidence=conf,
                price=features.loc[i, "close"] if "close" in features.columns else 0,
                rule_name="meta_best_side",
            )
            signals.append(sig)

        meta = Strategy(
            name="Meta_BestSide",
            description="Selects best side (long/short) based on model confidence",
            side="both",
            entry_conditions=[],
            exit_config={
                "target_pct": self.up_pct,
                "stop_pct": self.dn_pct,
                "leverage": self.leverage,
                "time_stop_bars": 120,
            },
            insights=[],
            signals=signals,
            metadata={"threshold": threshold},
        )

        log.info(f"Meta strategy: {len(signals)} signals")
        return meta
