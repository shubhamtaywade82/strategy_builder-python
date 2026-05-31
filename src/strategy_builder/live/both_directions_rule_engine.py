"""
Both-Direction Rule Engine
==========================
Evaluates LONG and SHORT rules simultaneously on every 1m bar.
When both sides fire, picks the higher-confidence signal.
Tracks PnL independently for long and short.

Usage:
    from strategy_builder.live.both_directions_rule_engine import BothDirectionsEngine

    engine = BothDirectionsEngine(
        symbol="SOLUSDT",
        long_rules=[{"name": "...", "conditions": [...], "target_pct": 0.01, "stop_pct": 0.005}],
        short_rules=[{"name": "...", "conditions": [...], "target_pct": 0.01, "stop_pct": 0.005}],
        account_size=10_000,
        paper=True,
    )
    engine.start()  # blocks
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional

from ..market_data.binance_rest_client import BinanceRestClient
from ..market_data.binance_websocket_client import BinanceCombinedStream, BinanceMiniTickerStream
from ..features import FeatureBuilder
from ..analytics.position_sizing import PositionSizer
from ..domain import Candle


@dataclass
class DirectionSignal:
    symbol: str
    side: str  # LONG or SHORT
    rule_name: str
    entry_price: float
    target_price: float
    stop_price: float
    position_size: float
    margin_required: float
    risk_amount: float
    confidence: float
    features_snapshot: Dict
    timestamp: float


@dataclass
class ActivePosition:
    signal: DirectionSignal
    entry_bar: int
    current_bar: int
    status: str = "open"  # open, target, stop, timeout
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    pnl_dollar: float = 0.0


@dataclass
class SessionPnL:
    long_trades: int = 0
    long_wins: int = 0
    long_pnl: float = 0.0
    short_trades: int = 0
    short_wins: int = 0
    short_pnl: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0


class BothDirectionsEngine:
    """Executes both long AND short strategies on the same live data stream.

    Design:
    - Independent rule evaluation: long rules and short rules checked separately
    - No hedging: if both fire on the same bar, only the higher-confidence signal is taken
    - Separate PnL tracking per direction
    - Combined risk cap: total risk across open positions limited by max_correlated_risk
    """

    def __init__(
        self,
        symbol: str,
        long_rules: List[Dict],
        short_rules: List[Dict],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        account_size: float = 10_000.0,
        base_risk_pct: float = 1.56,
        leverage: int = 10,
        max_correlated_risk: float = 0.05,
        confidence_threshold: float = 0.55,
        on_signal: Optional[Callable[[DirectionSignal], None]] = None,
        on_position_close: Optional[Callable[[ActivePosition], None]] = None,
        paper: bool = True,
    ):
        self.symbol = symbol.upper()
        self.long_rules = long_rules
        self.short_rules = short_rules
        self.rest = BinanceRestClient(api_key, api_secret)
        self.account_size = account_size
        self.base_risk_pct = base_risk_pct
        self.leverage = leverage
        self.max_correlated_risk = max_correlated_risk
        self.confidence_threshold = confidence_threshold
        self.on_signal = on_signal
        self.on_position_close = on_position_close
        self.paper = paper

        self.candles: Dict[str, List[Candle]] = {"1m": [], "15m": [], "1h": [], "4h": []}
        self.bar_count = 0
        self.last_price = 0.0

        self.long_positions: List[ActivePosition] = []
        self.short_positions: List[ActivePosition] = []
        self.signals_history: List[DirectionSignal] = []
        self.session_pnl = SessionPnL()

        self.long_sizer = PositionSizer(win_rate=0.576, avg_win=0.0095, avg_loss=0.0059)
        self.short_sizer = PositionSizer(win_rate=0.560, avg_win=0.0092, avg_loss=0.0061)

        self.ws_combined: Optional[BinanceCombinedStream] = None
        self.ws_ticker: Optional[BinanceMiniTickerStream] = None
        self.running = False

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        print(f"[BothDirections] {self.symbol} | long:{len(self.long_rules)} short:{len(self.short_rules)} | paper={self.paper}")
        self.running = True
        self._seed_candles()

        self.ws_combined = BinanceCombinedStream(
            [(self.symbol, tf) for tf in ["1m", "15m", "1h", "4h"]],
            on_candle=self._on_candle,
            on_error=lambda e: print(f"[WS Error] {e}"),
            on_connect=lambda: print("[BothDirections] WebSocket connected"),
        )
        self.ws_combined.start()

        self.ws_ticker = BinanceMiniTickerStream(
            self.symbol,
            on_tick=lambda t: setattr(self, "last_price", float(t.get("c", 0))),
        )
        self.ws_ticker.start()

        try:
            while self.running:
                self._monitor_positions()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[BothDirections] Stopping...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.ws_combined:
            self.ws_combined.stop()
        if self.ws_ticker:
            self.ws_ticker.stop()
        self._print_session_report()

    # ── Data ──────────────────────────────────────────────────────

    def _seed_candles(self):
        print("[BothDirections] Seeding candles...")
        for tf in ["1m", "15m", "1h", "4h"]:
            days = {"1m": 2, "15m": 7, "1h": 30, "4h": 90}[tf]
            try:
                self.candles[tf] = self.rest.get_candles_days(self.symbol, tf, days)[-500:]
                print(f"  {tf}: {len(self.candles[tf])} candles")
            except Exception as e:
                print(f"  {tf}: FAILED: {e}")

    def _on_candle(self, symbol: str, interval: str, candle: Candle):
        if symbol != self.symbol:
            return
        self.candles[interval].append(candle)
        if len(self.candles[interval]) > 1000:
            self.candles[interval] = self.candles[interval][-500:]
        if interval == "1m":
            self.bar_count += 1
            self._evaluate_both_directions()

    # ── Signal Evaluation ─────────────────────────────────────────

    def _evaluate_both_directions(self):
        if len(self.candles["1m"]) < 200:
            return
        try:
            features = FeatureBuilder.build(self.symbol, {tf: self.candles[tf] for tf in ["1m", "15m", "1h", "4h"]})
        except Exception:
            return

        long_signals = self._check_rules(self.long_rules, "LONG", features)
        short_signals = self._check_rules(self.short_rules, "SHORT", features)

        if long_signals and short_signals:
            best_long = max(long_signals, key=lambda s: s.confidence)
            best_short = max(short_signals, key=lambda s: s.confidence)
            self._emit_signal(best_long if best_long.confidence >= best_short.confidence else best_short)
        elif long_signals:
            self._emit_signal(max(long_signals, key=lambda s: s.confidence))
        elif short_signals:
            self._emit_signal(max(short_signals, key=lambda s: s.confidence))

    def _check_rules(self, rules: List[Dict], side: str, features: Dict) -> List[DirectionSignal]:
        signals = []
        for rule in rules:
            if not self._check_conditions(rule.get("conditions", []), features):
                continue
            sig = self._build_signal(rule, side, features)
            if sig and sig.confidence >= self.confidence_threshold:
                signals.append(sig)
        return signals

    def _check_conditions(self, conditions: List[Dict], features: Dict) -> bool:
        for cond in conditions:
            val = self._get_nested(features, cond["feature"].split("_"))
            if val is None:
                return False
            op, thresh = cond["operator"], cond["value"]
            if op == ">" and not (val > thresh): return False
            if op == ">=" and not (val >= thresh): return False
            if op == "<" and not (val < thresh): return False
            if op == "<=" and not (val <= thresh): return False
            if op == "==" and not (val == thresh): return False
        return True

    def _get_nested(self, d: Dict, path: List[str]):
        try:
            for key in path:
                d = d[key] if isinstance(d, dict) else None
                if d is None:
                    return None
            return d
        except (KeyError, TypeError):
            return None

    # ── Signal Building ───────────────────────────────────────────

    def _build_signal(self, rule: Dict, side: str, features: Dict) -> Optional[DirectionSignal]:
        entry = self.last_price or self.candles["1m"][-1].close
        target_pct = rule.get("target_pct", 0.01)
        stop_pct = rule.get("stop_pct", 0.005)
        sizer = self.long_sizer if side == "LONG" else self.short_sizer

        if side == "LONG":
            target = entry * (1 + target_pct)
            stop = entry * (1 - stop_pct)
        else:
            target = entry * (1 - target_pct)
            stop = entry * (1 + stop_pct)

        kelly = sizer.kelly()
        risk_pct = min(self.base_risk_pct, kelly.half_kelly * 100)
        size_info = sizer.fixed_fractional(self.account_size, risk_pct, entry, stop, self.leverage)

        if self._get_current_risk() + (risk_pct / 100) > self.max_correlated_risk:
            return None

        confidence = self._calculate_confidence(rule, features)
        return DirectionSignal(
            symbol=self.symbol, side=side, rule_name=rule["name"],
            entry_price=entry, target_price=target, stop_price=stop,
            position_size=size_info.get("notional", 0),
            margin_required=size_info.get("notional", 0) / self.leverage,
            risk_amount=size_info.get("risk_amount", 0),
            confidence=confidence,
            features_snapshot={k: v for k, v in features.items() if isinstance(v, (int, float, str))},
            timestamp=time.time(),
        )

    def _calculate_confidence(self, rule: Dict, features: Dict) -> float:
        conditions = rule.get("conditions", [])
        if not conditions:
            return 0.5
        score = 0
        for cond in conditions:
            val = self._get_nested(features, cond["feature"].split("_"))
            if val is None:
                continue
            thresh = cond["value"]
            if cond["operator"] in (">", ">="):
                score += min(1.0, val / max(thresh, 0.001))
            elif cond["operator"] in ("<", "<="):
                score += min(1.0, thresh / max(val, 0.001))
            else:
                score += 1.0 if val == thresh else 0.0
        return min(1.0, score / len(conditions))

    def _get_current_risk(self) -> float:
        return sum(
            p.signal.risk_amount / self.account_size
            for p in self.long_positions + self.short_positions
            if p.status == "open"
        )

    # ── Execution ─────────────────────────────────────────────────

    def _emit_signal(self, signal: DirectionSignal):
        self.signals_history.append(signal)
        color = "\033[32m" if signal.side == "LONG" else "\033[31m"
        reset = "\033[0m"
        print(f"\n{'='*60}")
        print(f"  {color}{signal.side} SIGNAL{reset}: {signal.rule_name}")
        print(f"  Entry: ${signal.entry_price:.2f} | Target: ${signal.target_price:.2f} | Stop: ${signal.stop_price:.2f}")
        print(f"  Size: ${signal.position_size:,.0f} | Confidence: {signal.confidence:.1%}")
        print(f"{'='*60}\n")

        if not self.paper and self.rest.api_key:
            self._place_real_order(signal)

        pos = ActivePosition(signal=signal, entry_bar=self.bar_count, current_bar=self.bar_count)
        if signal.side == "LONG":
            self.long_positions.append(pos)
        else:
            self.short_positions.append(pos)
        if self.on_signal:
            self.on_signal(signal)

    def _place_real_order(self, signal: DirectionSignal):
        try:
            side = "BUY" if signal.side == "LONG" else "SELL"
            contracts = round(signal.position_size / signal.entry_price, 3)
            self.rest.place_order(symbol=signal.symbol, side=side, order_type="MARKET", quantity=contracts)
            self.rest.place_order(
                symbol=signal.symbol,
                side="SELL" if signal.side == "LONG" else "BUY",
                order_type="STOP_MARKET", quantity=contracts,
                stop_price=round(signal.stop_price, 2), reduce_only=True,
            )
        except Exception as e:
            print(f"  Order error: {e}")

    # ── Position Monitoring ───────────────────────────────────────

    def _monitor_positions(self):
        price = self.last_price or (self.candles["1m"][-1].close if self.candles["1m"] else 0)
        if price == 0:
            return

        for pos in self.long_positions + self.short_positions:
            if pos.status != "open":
                continue
            sig = pos.signal
            pos.current_bar = self.bar_count
            bars_held = self.bar_count - pos.entry_bar

            if sig.side == "LONG":
                hit_target = price >= sig.target_price
                hit_stop = price <= sig.stop_price
            else:
                hit_target = price <= sig.target_price
                hit_stop = price >= sig.stop_price

            hit_timeout = bars_held >= 120

            if hit_target:
                pos.status = "target"
                pos.exit_price = sig.target_price
                raw = abs(sig.target_price - sig.entry_price) / sig.entry_price
                pos.pnl_pct, pos.pnl_dollar = raw, raw * sig.position_size
                self._close_position(pos)
            elif hit_stop:
                pos.status = "stop"
                pos.exit_price = sig.stop_price
                raw = -abs(sig.stop_price - sig.entry_price) / sig.entry_price
                pos.pnl_pct, pos.pnl_dollar = raw, raw * sig.position_size
                self._close_position(pos)
            elif hit_timeout:
                pos.status = "timeout"
                pos.exit_price = price
                raw = (price - sig.entry_price) / sig.entry_price
                pos.pnl_pct = raw if sig.side == "LONG" else -raw
                pos.pnl_dollar = pos.pnl_pct * sig.position_size
                self._close_position(pos)

    def _close_position(self, pos: ActivePosition):
        sig = pos.signal
        is_win = pos.pnl_pct > 0
        self.session_pnl.total_trades += 1
        self.session_pnl.total_pnl += pos.pnl_dollar

        if sig.side == "LONG":
            self.session_pnl.long_trades += 1
            self.session_pnl.long_pnl += pos.pnl_dollar
            if is_win:
                self.session_pnl.long_wins += 1
        else:
            self.session_pnl.short_trades += 1
            self.session_pnl.short_pnl += pos.pnl_dollar
            if is_win:
                self.session_pnl.short_wins += 1

        if self.session_pnl.total_pnl > self.session_pnl.peak_pnl:
            self.session_pnl.peak_pnl = self.session_pnl.total_pnl
        dd = self.session_pnl.peak_pnl - self.session_pnl.total_pnl
        if dd > self.session_pnl.max_drawdown:
            self.session_pnl.max_drawdown = dd

        color = "\033[32m" if is_win else "\033[31m"
        reset = "\033[0m"
        print(f"\n  [{color}{pos.status.upper()}{reset}] {sig.side} | {sig.rule_name}")
        print(f"    PnL: {color}${pos.pnl_dollar:+.2f} ({pos.pnl_pct*100:+.2f}%){reset}")
        if self.on_position_close:
            self.on_position_close(pos)

    # ── Reporting ─────────────────────────────────────────────────

    def _print_session_report(self):
        pnl = self.session_pnl
        print(f"\n{'='*60}\nSESSION REPORT")
        print(f"  Long:  {pnl.long_trades} trades, {pnl.long_wins} wins, PnL: ${pnl.long_pnl:+.2f}")
        print(f"  Short: {pnl.short_trades} trades, {pnl.short_wins} wins, PnL: ${pnl.short_pnl:+.2f}")
        print(f"  Total: ${pnl.total_pnl:+.2f} | MaxDD: ${pnl.max_drawdown:.2f}")
        print(f"{'='*60}")

    def get_stats(self) -> Dict:
        pnl = self.session_pnl
        return {
            "long_signals": sum(1 for s in self.signals_history if s.side == "LONG"),
            "short_signals": sum(1 for s in self.signals_history if s.side == "SHORT"),
            "long_trades": pnl.long_trades,
            "long_wins": pnl.long_wins,
            "long_win_rate": pnl.long_wins / max(pnl.long_trades, 1),
            "long_pnl": pnl.long_pnl,
            "short_trades": pnl.short_trades,
            "short_wins": pnl.short_wins,
            "short_win_rate": pnl.short_wins / max(pnl.short_trades, 1),
            "short_pnl": pnl.short_pnl,
            "total_pnl": pnl.total_pnl,
            "max_drawdown": pnl.max_drawdown,
            "open_long": sum(1 for p in self.long_positions if p.status == "open"),
            "open_short": sum(1 for p in self.short_positions if p.status == "open"),
            "current_price": self.last_price,
            "bar_count": self.bar_count,
        }
