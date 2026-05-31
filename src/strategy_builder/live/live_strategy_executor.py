"""
Live Strategy Executor
======================
Executes research-validated rules using real Binance WebSocket + REST data.

Usage:
    from strategy_builder.live.live_strategy_executor import LiveStrategyExecutor

    executor = LiveStrategyExecutor(
        symbol="SOLUSDT",
        rules=[{
            "name": "SOL Long 2:1",
            "side": "LONG",
            "conditions": [
                {"feature": "4h_trend", "operator": ">", "value": 0},
                {"feature": "1h_bos", "operator": "==", "value": 1},
            ],
            "target_pct": 0.01,
            "stop_pct": 0.005,
            "max_hold_bars": 120,
        }],
        api_key="YOUR_KEY",    # optional — omit for paper mode
        api_secret="YOUR_SECRET",
        paper=True,
    )
    executor.start()  # blocks; Ctrl-C to stop
"""
import time
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional

from ..market_data.binance_rest_client import BinanceRestClient
from ..market_data.binance_websocket_client import BinanceCombinedStream, BinanceMiniTickerStream
from ..features import FeatureBuilder
from ..analytics.position_sizing import PositionSizer
from ..domain import Candle


@dataclass
class Signal:
    symbol: str
    side: str  # LONG or SHORT
    rule_name: str
    entry_price: float
    target_price: float
    stop_price: float
    max_hold_bars: int
    position_size: float
    timestamp: float


@dataclass
class ActiveTrade:
    signal: Signal
    entry_bar: int
    current_bar: int
    status: str = "open"  # open, target_hit, stop_hit, timeout
    exit_price: float = 0.0
    pnl: float = 0.0


class LiveStrategyExecutor:
    """Live executor that:
    1. Seeds historical candles via REST
    2. Streams 1m/15m/1h/4h klines via WebSocket
    3. Builds full MTF feature set on each 1m close
    4. Evaluates research-validated rules
    5. Sizes position with Kelly Criterion
    6. Places orders via REST API (when paper=False and keys provided)
    7. Monitors open trades for target/stop/timeout
    """

    def __init__(
        self,
        symbol: str,
        rules: List[Dict],
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        account_size: float = 10_000.0,
        base_risk_pct: float = 1.56,  # half-Kelly default
        leverage: int = 10,
        on_signal: Optional[Callable[[Signal], None]] = None,
        on_trade_close: Optional[Callable[[ActiveTrade], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        paper: bool = True,
    ):
        self.symbol = symbol.upper()
        self.rules = rules
        self.rest = BinanceRestClient(api_key, api_secret)
        self.account_size = account_size
        self.base_risk_pct = base_risk_pct
        self.leverage = leverage
        self.on_signal = on_signal
        self.on_trade_close = on_trade_close
        self.on_error = on_error
        self.paper = paper

        self.candles: Dict[str, List[Candle]] = {"1m": [], "15m": [], "1h": [], "4h": []}
        self.bar_counts: Dict[str, int] = {"1m": 0, "15m": 0, "1h": 0, "4h": 0}
        self.active_trades: List[ActiveTrade] = []
        self.signals_generated: List[Signal] = []
        self.running = False
        self.last_price = 0.0

        self.sizer = PositionSizer(win_rate=0.576, avg_win=0.0095, avg_loss=0.0059)
        self.ws_combined: Optional[BinanceCombinedStream] = None
        self.ws_ticker: Optional[BinanceMiniTickerStream] = None

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        print(f"[LiveExecutor] {self.symbol} | {len(self.rules)} rules | paper={self.paper}")
        self.running = True
        self._seed_candles()

        self.ws_combined = BinanceCombinedStream(
            [(self.symbol, tf) for tf in ["1m", "15m", "1h", "4h"]],
            on_candle=self._on_candle,
            on_error=self._on_ws_error,
            on_connect=lambda: print("[LiveExecutor] WebSocket connected"),
        )
        self.ws_combined.start()

        self.ws_ticker = BinanceMiniTickerStream(
            self.symbol,
            on_tick=lambda t: setattr(self, "last_price", float(t.get("c", 0))),
            on_error=self._on_ws_error,
        )
        self.ws_ticker.start()

        try:
            while self.running:
                self._monitor_trades()
                time.sleep(1)
        except KeyboardInterrupt:
            print("[LiveExecutor] Stopping...")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.ws_combined:
            self.ws_combined.stop()
        if self.ws_ticker:
            self.ws_ticker.stop()
        print("[LiveExecutor] Stopped")

    # ── Data ──────────────────────────────────────────────────────

    def _seed_candles(self):
        print("[LiveExecutor] Seeding historical candles...")
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
        self.bar_counts[interval] += 1
        if interval == "1m":
            self._evaluate_rules()

    # ── Signal Evaluation ─────────────────────────────────────────

    def _evaluate_rules(self):
        if len(self.candles["1m"]) < 200 or len(self.candles["15m"]) < 50:
            return
        try:
            features = FeatureBuilder.build(self.symbol, {tf: self.candles[tf] for tf in ["1m", "15m", "1h", "4h"]})
        except Exception as e:
            if self.on_error:
                self.on_error(f"Feature build error: {e}")
            return

        for rule in self.rules:
            if self._check_rule(rule, features):
                self._execute_signal(rule, features)

    def _check_rule(self, rule: Dict, features: Dict) -> bool:
        for cond in rule.get("conditions", []):
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

    # ── Signal Execution ──────────────────────────────────────────

    def _execute_signal(self, rule: Dict, features: Dict):
        entry = self.last_price or self.candles["1m"][-1].close
        side = rule["side"]
        target_pct = rule.get("target_pct", 0.01)
        stop_pct = rule.get("stop_pct", 0.005)

        target = entry * (1 + target_pct) if side == "LONG" else entry * (1 - target_pct)
        stop = entry * (1 - stop_pct) if side == "LONG" else entry * (1 + stop_pct)

        size_info = self.sizer.fixed_fractional(self.account_size, self.base_risk_pct, entry, stop, self.leverage)
        signal = Signal(
            symbol=self.symbol, side=side, rule_name=rule["name"],
            entry_price=entry, target_price=target, stop_price=stop,
            max_hold_bars=rule.get("max_hold_bars", 120),
            position_size=size_info.get("notional", 0),
            timestamp=time.time(),
        )
        self.signals_generated.append(signal)

        print(f"\n{'='*60}")
        print(f"  SIGNAL: {signal.rule_name} ({signal.side})")
        print(f"  Entry: ${signal.entry_price:.2f}")
        print(f"  Target: ${signal.target_price:.2f}")
        print(f"  Stop: ${signal.stop_price:.2f}")
        print(f"  Size: ${signal.position_size:,.0f} | Paper: {self.paper}")
        print(f"{'='*60}\n")

        if not self.paper and self.rest.api_key:
            self._place_order(signal)

        trade = ActiveTrade(signal=signal, entry_bar=self.bar_counts["1m"], current_bar=self.bar_counts["1m"])
        self.active_trades.append(trade)
        if self.on_signal:
            self.on_signal(signal)

    def _place_order(self, signal: Signal):
        try:
            side = "BUY" if signal.side == "LONG" else "SELL"
            contracts = round(signal.position_size / signal.entry_price, 3)
            self.rest.place_order(symbol=signal.symbol, side=side, order_type="MARKET", quantity=contracts)
            self.rest.place_order(
                symbol=signal.symbol,
                side="SELL" if signal.side == "LONG" else "BUY",
                order_type="STOP_MARKET",
                quantity=contracts,
                stop_price=round(signal.stop_price, 2),
                reduce_only=True,
            )
        except Exception as e:
            if self.on_error:
                self.on_error(f"Order error: {e}")

    # ── Position Monitoring ───────────────────────────────────────

    def _monitor_trades(self):
        price = self.last_price or (self.candles["1m"][-1].close if self.candles["1m"] else 0)
        if price == 0:
            return

        for trade in self.active_trades:
            if trade.status != "open":
                continue
            sig = trade.signal
            trade.current_bar = self.bar_counts["1m"]
            bars_held = self.bar_counts["1m"] - trade.entry_bar

            if sig.side == "LONG":
                hit_target = price >= sig.target_price
                hit_stop = price <= sig.stop_price
            else:
                hit_target = price <= sig.target_price
                hit_stop = price >= sig.stop_price

            hit_timeout = bars_held >= sig.max_hold_bars

            if hit_target:
                trade.status = "target_hit"
                trade.exit_price = sig.target_price
                trade.pnl = abs(sig.target_price - sig.entry_price) / sig.entry_price
                self._close_trade(trade)
            elif hit_stop:
                trade.status = "stop_hit"
                trade.exit_price = sig.stop_price
                trade.pnl = -abs(sig.stop_price - sig.entry_price) / sig.entry_price
                self._close_trade(trade)
            elif hit_timeout:
                trade.status = "timeout"
                trade.exit_price = price
                raw = (price - sig.entry_price) / sig.entry_price
                trade.pnl = raw if sig.side == "LONG" else -raw
                self._close_trade(trade)

    def _close_trade(self, trade: ActiveTrade):
        print(f"\n  [CLOSED] {trade.signal.rule_name} | {trade.status}")
        print(f"    Exit: ${trade.exit_price:.2f} | PnL: {trade.pnl*100:+.2f}%")
        if self.on_trade_close:
            self.on_trade_close(trade)

    def _on_ws_error(self, error: str):
        if self.on_error:
            self.on_error(f"WebSocket: {error}")
        else:
            print(f"[LiveExecutor] WebSocket error: {error}")

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        closed = [t for t in self.active_trades if t.status != "open"]
        wins = [t for t in closed if t.pnl > 0]
        return {
            "signals_generated": len(self.signals_generated),
            "trades_total": len(closed),
            "trades_open": len(self.active_trades) - len(closed),
            "wins": len(wins),
            "losses": len(closed) - len(wins),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl_pct": sum(t.pnl for t in closed) * 100,
            "current_price": self.last_price,
        }
