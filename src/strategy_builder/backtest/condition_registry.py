import logging
from typing import Dict, List, Any, Optional, Callable
from .evaluation_context import EvaluationContext
from ..exceptions import ValidationError
from ..features.momentum_engine import MomentumEngine
from ..features.structure_detector import StructureDetector

class ConditionRegistry:
    _registry = {}
    _logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, name: str, func: Optional[Callable] = None):
        if func is None:
            def decorator(f: Callable):
                cls._registry[name] = f
                return f
            return decorator
        
        cls._registry[name] = func

    @classmethod
    def evaluate(cls, condition_name: str, context: EvaluationContext) -> bool:
        evaluator = cls._registry.get(condition_name)
        if evaluator:
            try:
                return evaluator(context)
            except Exception as e:
                cls._logger.error(f"Error evaluating condition {condition_name}: {e}")
                return False
        else:
            cls._logger.warning(f"Unregistered condition evaluated: {condition_name}")
            return False

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def condition_ids(cls) -> List[str]:
        return sorted(list(cls._registry.keys()))

    @classmethod
    def validate_condition_names(cls, names: List[str]):
        unknown = [n for n in names if not cls.is_registered(n)]
        if unknown:
            raise ValidationError(f"Unknown entry conditions (not in ConditionRegistry): {', '.join(unknown)}")

    @classmethod
    def load_defaults(cls):
        # Institutional Regime Conditions
        @cls.register('regime_trend_expansion')
        def regime_trend_expansion(ctx):
            return ctx.regime() == "trend_expansion"

        @cls.register('regime_breakout_environment')
        def regime_breakout_environment(ctx):
            return ctx.regime() == "breakout_environment"

        @cls.register('regime_mean_reversion')
        def regime_mean_reversion(ctx):
            return ctx.regime() == "mean_reversion"

        @cls.register('regime_compression')
        def regime_compression(ctx):
            return ctx.regime() == "compression"

        @cls.register('kill_zone_london_ny')
        def kill_zone_london_ny(ctx):
            sessions = ctx.session_analytics().get("current_session")
            return sessions == "london_ny_overlap"

        @cls.register('kill_zone_asia_london')
        def kill_zone_asia_london(ctx):
            sessions = ctx.session_analytics().get("current_session")
            return sessions == "asia_london_overlap"

        @cls.register('high_rvol_confirmation')
        def high_rvol_confirmation(ctx):
            # Session engine rvol
            analytics = ctx.session_analytics()
            return analytics.get("rvol", 0) > 1.4

        # 1. Session Breakout
        @cls.register('asia_range_defined')
        def asia_range_defined(ctx):
            box = ctx.asia_box()
            if not box: return False
            return box['high'] > box['low']

        @cls.register('session_high_break')
        def session_high_break(ctx):
            sessions = ctx.current_sessions()
            if 'new_york' in sessions:
                box = ctx.london_box() or ctx.asia_box()
            elif 'london' in sessions or 'london_ny' in sessions:
                box = ctx.asia_box()
            else:
                box = ctx.asia_box()

            if not box or not ctx.current_candle or not ctx.previous_candle:
                return False

            box_high = box['high']
            box_low = box['low']
            cur = ctx.current_candle.close
            prev = ctx.previous_candle.close

            if cur > box_high and prev <= box_high:
                ctx.direction = "long"
                ctx.entry_price = cur
                ctx.stop_distance = min(ctx.atr() * 1.5, cur - box_low)
                return True
            elif cur < box_low and prev >= box_low:
                ctx.direction = "short"
                ctx.entry_price = cur
                ctx.stop_distance = min(ctx.atr() * 1.5, box_high - cur)
                return True
            return False

        @cls.register('volume_confirmation')
        def volume_confirmation(ctx):
            return ctx.volume_zscore() >= 1.0

        # 2. Mean Reversion
        @cls.register('price_near_session_extreme')
        def price_near_session_extreme(ctx):
            sessions = ctx.current_sessions()
            if 'new_york' in sessions:
                box = ctx.london_box()
            elif 'london' in sessions or 'london_ny' in sessions:
                box = ctx.asia_box()
            else:
                box = ctx.asia_box()

            if box:
                last_high_price = box['high']
                last_low_price = box['low']
            else:
                sp = ctx.swing_points()
                if not sp['highs'] or not sp['lows']: return False
                last_high_price = sp['highs'][-1]['price']
                last_low_price = sp['lows'][-1]['price']
            
            if not ctx.current_candle: return False
            cur = ctx.current_candle.close
            atr = ctx.atr()

            if abs(cur - last_low_price) < atr * 0.5:
                ctx.direction = "long"
                ctx.entry_price = cur
                ctx.stop_distance = atr
                return True
            elif abs(cur - last_high_price) < atr * 0.5:
                ctx.direction = "short"
                ctx.entry_price = cur
                ctx.stop_distance = atr
                return True
            return False

        @cls.register('rsi_divergence')
        def rsi_divergence(ctx):
            rsi = ctx.rsi()
            if rsi is None: return False
            if ctx.direction == "long" and rsi < 35: return True
            if ctx.direction == "short" and rsi > 65: return True
            return False

        @cls.register('volume_decline')
        def volume_decline(ctx):
            return ctx.volume_zscore() < 1.0

        # 3. MTF Pullback Conditions
        @cls.register('higher_tf_trend_bullish')
        def higher_tf_trend_bullish(ctx):
            return ctx.higher_tf_structure() == "bullish"

        @cls.register('lower_tf_pullback_to_ema')
        def lower_tf_pullback_to_ema(ctx):
            ema_20 = ctx.ema(period=20)
            if not ema_20: return False
            
            ema_current = ema_20[-1]
            atr = ctx.atr()
            distance = abs(ctx.current_candle.close - ema_current) / (atr + 0.0001)
            return distance < 0.5

        @cls.register('structure_hold')
        def structure_hold(ctx):
            sp = ctx.swing_points()
            if not sp['lows'] or not ctx.current_candle: return False
            last_low_price = sp['lows'][-1]['price']
            return ctx.current_candle.close > (last_low_price - ctx.atr() * 0.25)

        @cls.register('trigger_candle')
        def trigger_candle(ctx):
            if not ctx.current_candle or not ctx.previous_candle: return False
            struct = ctx.structure()
            if struct == "bullish" and ctx.current_candle.close > ctx.previous_candle.close:
                ctx.direction = "long"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.5
                return True
            elif struct == "bearish" and ctx.current_candle.close < ctx.previous_candle.close:
                ctx.direction = "short"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.5
                return True
            return False

        # 4. Compression Breakout
        @cls.register('compression_detected')
        def compression_detected(ctx):
            # Simplified: check current regime if lookback is hard to implement here
            return ctx.regime() == "compression"

        @cls.register('range_break')
        def range_break(ctx):
            atr = ctx.atr()
            current_range = ctx.current_candle.high - ctx.current_candle.low
            expansion = current_range / atr if atr > 0 else 0

            if expansion > 1.3:
                ctx.direction = "long" if ctx.current_candle.close > ctx.current_candle.open else "short"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = atr * 1.0
                return True
            return False

        @cls.register('volume_surge')
        def volume_surge(ctx):
            return ctx.volume_zscore() >= 1.5

        @cls.register('direction_bias')
        def direction_bias(ctx):
            return ctx.direction is not None

        # 5. Failed Breakout
        @cls.register('breakout_attempt')
        def breakout_attempt(ctx):
            sp = ctx.swing_points()
            if not sp['highs'] or not sp['lows'] or len(ctx.candles) < 5: return False
            
            last_high = sp['highs'][-1]['price']
            last_low = sp['lows'][-1]['price']
            recent = ctx.candles[-5:]
            
            if any(c.high > last_high for c in recent):
                ctx.direction = "short"
                return True
            elif any(c.low < last_low for c in recent):
                ctx.direction = "long"
                return True
            return False

        @cls.register('retest_below_level')
        def retest_below_level(ctx):
            sp = ctx.swing_points()
            last_high = sp['highs'][-1]['price'] if sp['highs'] else None
            last_low = sp['lows'][-1]['price'] if sp['lows'] else None

            if ctx.direction == "short" and last_high and ctx.current_candle.close < last_high:
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.0
                return True
            elif ctx.direction == "long" and last_low and ctx.current_candle.close > last_low:
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.0
                return True
            return False

        @cls.register('rejection_candle')
        def rejection_candle(ctx):
            c = ctx.current_candle
            if not c: return False
            body_range = abs(c.high - c.low) + 0.0001
            if ctx.direction == "short":
                wick_ratio = (c.high - c.close) / body_range
                return wick_ratio >= 0.5
            elif ctx.direction == "long":
                wick_ratio = (c.close - c.low) / body_range
                return wick_ratio >= 0.5
            return False

        @cls.register('volume_divergence')
        def volume_divergence(ctx):
            # This requires access to VolumeProfile or similar. 
            # Simplified for now.
            return ctx.volume_zscore() < 0.5

        # 6. VWAP Reclaim
        @cls.register('price_reclaims_vwap')
        def price_reclaims_vwap(ctx):
            vwap_vals = ctx.vwap()
            if len(vwap_vals) < 2 or not ctx.previous_candle: return False
            vwap_current = vwap_vals[-1]
            vwap_prev = vwap_vals[-2]

            if ctx.previous_candle.close < vwap_prev and ctx.current_candle.close > vwap_current:
                ctx.direction = "long"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.5
                return True
            elif ctx.previous_candle.close > vwap_prev and ctx.current_candle.close < vwap_current:
                ctx.direction = "short"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.5
                return True
            return False

        @cls.register('structure_bullish_shift')
        def structure_bullish_shift(ctx):
            # Simplified
            return ctx.higher_tf_structure() == "bullish"

        @cls.register('volume_on_reclaim')
        def volume_on_reclaim(ctx):
            return ctx.volume_zscore() >= 1.0

        @cls.register('momentum_confirmation')
        def momentum_confirmation(ctx):
            rsi = ctx.rsi()
            if rsi is None: return False
            if ctx.direction == "long": return rsi >= 45
            if ctx.direction == "short": return rsi <= 55
            return False

        @cls.register('volatility_exhaustion_entry')
        def volatility_exhaustion_entry(ctx):
            if len(ctx.candles) < 21: return False
            closes = [c.close for c in ctx.candles[-21:-1]]
            mean = sum(closes) / 20
            variance = sum((x - mean) ** 2 for x in closes) / 20
            std_dev = variance ** 0.5
            upper_bb = mean + 3.0 * std_dev
            lower_bb = mean - 3.0 * std_dev

            vols = [c.volume for c in ctx.candles[-21:-1]]
            vol_sma = sum(vols) / 20

            prev = ctx.previous_candle
            cur = ctx.current_candle
            if not prev or not cur: return False

            rsi = ctx.rsi()
            if rsi is None: return False

            atr = ctx.atr()

            if prev.close < lower_bb and prev.volume > 2.0 * vol_sma and rsi < 30:
                if cur.close > cur.open and cur.close > prev.open:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = 1.5 * atr
                    return True
            
            if prev.close > upper_bb and prev.volume > 2.0 * vol_sma and rsi > 70:
                if cur.close < cur.open and cur.close < prev.open:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = 1.5 * atr
                    return True

            return False

        @cls.register('liquidity_sweep_mss_entry')
        def liquidity_sweep_mss_entry(ctx):
            if len(ctx.candles) < 25: return False
            prev = ctx.previous_candle
            cur = ctx.current_candle
            if not prev or not cur: return False

            past_candles = ctx.candles[-22:-2]
            swing_high = max(c.high for c in past_candles)
            swing_low = min(c.low for c in past_candles)

            atr = ctx.atr()

            if prev.high > swing_high and prev.close < swing_high:
                if cur.close < prev.low:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = max(atr, prev.high - cur.close)
                    return True

            if prev.low < swing_low and prev.close > swing_low:
                if cur.close > prev.high:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = max(atr, cur.close - prev.low)
                    return True

            return False

        @cls.register('funding_rate_arbitrage_entry')
        def funding_rate_arbitrage_entry(ctx):
            rsi = ctx.rsi()
            vol_z = ctx.volume_zscore()
            if rsi is None: return False
            
            if rsi > 80 and vol_z > 1.5:
                ctx.direction = "short"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.0
                return True
            elif rsi < 20 and vol_z > 1.5:
                ctx.direction = "long"
                ctx.entry_price = ctx.current_candle.close
                ctx.stop_distance = ctx.atr() * 1.0
                return True
            return False

        @cls.register('vwap_std_dev_reversion_entry')
        def vwap_std_dev_reversion_entry(ctx):
            vwaps = ctx.vwap()
            if len(vwaps) < 20 or len(ctx.candles) < 20: return False
            vwap_curr = vwaps[-1]
            
            closes = [c.close for c in ctx.candles[-20:]]
            diffs = [c - v for c, v in zip(closes, vwaps[-20:])]
            mean_diff = sum(diffs) / 20
            var = sum((x - mean_diff) ** 2 for x in diffs) / 20
            std_dev = var ** 0.5
            
            cur = ctx.current_candle
            vol_z = ctx.volume_zscore()
            atr = ctx.atr()

            if cur.close < vwap_curr - 2.5 * std_dev and vol_z > 1.5:
                if cur.close > cur.open:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = max(atr, abs(cur.close - (vwap_curr - 3.5 * std_dev)))
                    return True
            elif cur.close > vwap_curr + 2.5 * std_dev and vol_z > 1.5:
                if cur.close < cur.open:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = max(atr, abs((vwap_curr + 3.5 * std_dev) - cur.close))
                    return True
            return False

        @cls.register('filtered_trend_following_entry')
        def filtered_trend_following_entry(ctx):
            if len(ctx.candles) < 200: return False
            
            adx_proxy = 0.0
            period = 14
            up_moves = 0
            down_moves = 0
            for i in range(len(ctx.candles) - period, len(ctx.candles)):
                diff_h = ctx.candles[i].high - ctx.candles[i-1].high
                diff_l = ctx.candles[i-1].low - ctx.candles[i].low
                if diff_h > diff_l and diff_h > 0:
                    up_moves += diff_h
                elif diff_l > diff_h and diff_l > 0:
                    down_moves += diff_l
            if up_moves + down_moves > 0:
                adx_proxy = (abs(up_moves - down_moves) / (up_moves + down_moves)) * 100.0

            if adx_proxy < 25: return False

            ema_50 = ctx.ema(50)
            ema_200 = ctx.ema(200)
            ema_21 = ctx.ema(21)
            
            if not ema_50 or not ema_200 or not ema_21 or ema_50[-1] is None or ema_200[-1] is None or ema_21[-1] is None: return False
            e50, e200, e21 = ema_50[-1], ema_200[-1], ema_21[-1]
            
            cur = ctx.current_candle
            atr = ctx.atr()

            if cur.close > e50 and e50 > e200:
                if cur.low <= e21 + 0.1 * atr and cur.close > e21:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = 1.5 * atr
                    return True

            if cur.close < e50 and e50 < e200:
                if cur.high >= e21 - 0.1 * atr and cur.close < e21:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = 1.5 * atr
                    return True

            return False

        @cls.register('opening_range_breakout_entry')
        def opening_range_breakout_entry(ctx):
            if not ctx.current_candle or not ctx.previous_candle: return False
            cur = ctx.current_candle
            prev = ctx.previous_candle
            
            import datetime as dt_module
            ts = cur.timestamp
            if isinstance(ts, int):
                cur_dt = dt_module.datetime.utcfromtimestamp(ts)
            else:
                cur_dt = ts
                
            day_start = cur_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            range_end = cur_dt.replace(hour=0, minute=30, second=0, microsecond=0)
            if cur_dt <= range_end: return False
            
            opening_candles = []
            for c in ctx.candles:
                c_ts = c.timestamp
                if isinstance(c_ts, int):
                    c_dt = dt_module.datetime.utcfromtimestamp(c_ts)
                else:
                    c_dt = c_ts
                if day_start <= c_dt <= range_end:
                    opening_candles.append(c)
                    
            if not opening_candles: return False
            
            op_high = max(c.high for c in opening_candles)
            op_low = min(c.low for c in opening_candles)
            
            ema_50 = ctx.ema(50)
            ema_200 = ctx.ema(200)
            if not ema_50 or not ema_200 or ema_50[-1] is None or ema_200[-1] is None: return False
            e50, e200 = ema_50[-1], ema_200[-1]
            
            if prev.close <= op_high and cur.close > op_high and e50 > e200:
                ctx.direction = "long"
                ctx.entry_price = cur.close
                ctx.stop_distance = max(ctx.atr(), cur.close - op_low)
                return True
            elif prev.close >= op_low and cur.close < op_low and e50 < e200:
                ctx.direction = "short"
                ctx.entry_price = cur.close
                ctx.stop_distance = max(ctx.atr(), op_high - cur.close)
                return True
                
            return False

        @cls.register('bollinger_band_walk_entry')
        def bollinger_band_walk_entry(ctx):
            if len(ctx.candles) < 21: return False
            
            adx_proxy = 0.0
            period = 14
            up_moves = 0
            down_moves = 0
            for i in range(len(ctx.candles) - period, len(ctx.candles)):
                diff_h = ctx.candles[i].high - ctx.candles[i-1].high
                diff_l = ctx.candles[i-1].low - ctx.candles[i].low
                if diff_h > diff_l and diff_h > 0:
                    up_moves += diff_h
                elif diff_l > diff_h and diff_l > 0:
                    down_moves += diff_l
            if up_moves + down_moves > 0:
                adx_proxy = (abs(up_moves - down_moves) / (up_moves + down_moves)) * 100.0

            if adx_proxy < 30: return False

            closes = [c.close for c in ctx.candles[-20:]]
            mean = sum(closes) / 20
            variance = sum((x - mean) ** 2 for x in closes) / 20
            std_dev = variance ** 0.5
            upper_bb = mean + 2.0 * std_dev
            lower_bb = mean - 2.0 * std_dev

            cur = ctx.current_candle
            
            if cur.high >= upper_bb and cur.close > mean:
                ctx.direction = "long"
                ctx.entry_price = cur.close
                ctx.stop_distance = max(ctx.atr(), cur.close - mean)
                return True
                
            if cur.low <= lower_bb and cur.close < mean:
                ctx.direction = "short"
                ctx.entry_price = cur.close
                ctx.stop_distance = max(ctx.atr(), mean - cur.close)
                return True

            return False

        @cls.register('rsi_divergence_structure_entry')
        def rsi_divergence_structure_entry(ctx):
            if len(ctx.candles) < 20: return False
            rsi_series = MomentumEngine.rsi(ctx.candles, period=14)
            if len(rsi_series) < 20 or any(r is None for r in rsi_series[-20:]): return False
            
            closes = [c.close for c in ctx.candles]
            rsi_past = rsi_series[-15:-5]
            rsi_recent = rsi_series[-5:]
            close_past = closes[-15:-5]
            close_recent = closes[-5:]
            
            min_c_past, min_c_recent = min(close_past), min(close_recent)
            min_r_past, min_r_recent = min(rsi_past), min(rsi_recent)
            
            max_c_past, max_c_recent = max(close_past), max(close_recent)
            max_r_past, max_r_recent = max(rsi_past), max(rsi_recent)
            
            cur = ctx.current_candle
            prev = ctx.previous_candle
            if not prev or not cur: return False

            if min_c_recent < min_c_past and min_r_recent > min_r_past:
                if cur.close > prev.high:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = ctx.atr() * 1.5
                    return True

            if max_c_recent > max_c_past and max_r_recent < max_r_past:
                if cur.close < prev.low:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = ctx.atr() * 1.5
                    return True

            return False

        @cls.register('order_block_mitigation_entry')
        def order_block_mitigation_entry(ctx):
            if len(ctx.candles) < 30: return False
            
            cur = ctx.current_candle
            if not cur: return False
            atr = ctx.atr()
            
            for i in range(len(ctx.candles) - 15, 3, -1):
                c = ctx.candles[i]
                if c.close < c.open:
                    if (ctx.candles[i+1].close > ctx.candles[i+1].open and
                        ctx.candles[i+2].close > ctx.candles[i+2].open and
                        ctx.candles[i+3].close > c.high):
                        
                        if cur.low <= c.high and cur.low >= c.low and cur.close > c.low:
                            if (cur.close - cur.low) / (cur.high - cur.low + 0.0001) >= 0.5:
                                ctx.direction = "long"
                                ctx.entry_price = cur.close
                                ctx.stop_distance = max(atr, cur.close - c.low)
                                return True
            
            for i in range(len(ctx.candles) - 15, 3, -1):
                c = ctx.candles[i]
                if c.close > c.open:
                    if (ctx.candles[i+1].close < ctx.candles[i+1].open and
                        ctx.candles[i+2].close < ctx.candles[i+2].open and
                        ctx.candles[i+3].close < c.low):
                        
                        if cur.high >= c.low and cur.high <= c.high and cur.close < c.high:
                            if (cur.high - cur.close) / (cur.high - cur.low + 0.0001) >= 0.5:
                                ctx.direction = "short"
                                ctx.entry_price = cur.close
                                ctx.stop_distance = max(atr, c.high - cur.close)
                                return True
                                
            return False

        @cls.register('delta_neutral_basis_entry')
        def delta_neutral_basis_entry(ctx):
            vwaps = ctx.vwap()
            if len(vwaps) < 20 or len(ctx.candles) < 20: return False
            vwap_curr = vwaps[-1]
            closes = [c.close for c in ctx.candles[-20:]]
            mean_c = sum(closes) / 20
            std_c = (sum((x - mean_c) ** 2 for x in closes) / 20) ** 0.5
            
            cur = ctx.current_candle
            atr = ctx.atr()
            
            if abs(cur.close - vwap_curr) > 2.5 * std_c:
                ctx.direction = "long" if cur.close < vwap_curr else "short"
                ctx.entry_price = cur.close
                ctx.stop_distance = max(atr, 0.01 * cur.close)
                return True
            return False

        @cls.register('mtf_trend_alignment_entry')
        def mtf_trend_alignment_entry(ctx):
            # Macro Trend Alignment (4h, 1h) - Skipping 1d if not enough data
            tfs_macro = ["4h", "1h"]
            for tf in tfs_macro:
                series = ctx._mtf_series_up_to(tf)
                if not series or len(series) < 10: continue # Skip if not enough history for EMA
                
                ema = ctx.mtf_ema(tf, 20) # Faster EMA for alignment
                if ema is None: continue
                
                curr_price = series[-1].close
                if curr_price < ema: return False

            # Intermediate Momentum Alignment (15m, 5m)
            tfs_intermediate = ["15m", "5m"]
            for tf in tfs_intermediate:
                rsi = ctx.mtf_rsi(tf, 14)
                if rsi is None: continue
                if rsi < 45: return False # Slightly more permissive

            # Fine-grain Entry Trigger (1m)
            if not ctx.current_candle: return False
            
            # Simple EMA cross on 1m as trigger
            ema_fast = ctx.ema(9)
            if not ema_fast or len(ema_fast) < 2: return False
            
            cur = ctx.current_candle.close
            prev = ctx.previous_candle.close if ctx.previous_candle else cur
            
            if cur > ema_fast[-1] and prev <= ema_fast[-1]:
                ctx.direction = "long"
                ctx.entry_price = cur
                ctx.stop_distance = ctx.atr() * 1.5
                return True
                
            return False

        @cls.register('advanced_mtf_trend_alignment')
        def advanced_mtf_trend_alignment(ctx):
            # 1. Macro Bias (4h)
            ema_4h = ctx.mtf_ema("4h", 50)
            if ema_4h is None: return False
            
            series_4h = ctx._mtf_series_up_to("4h")
            if not series_4h: return False
            bias_bullish = series_4h[-1].close > ema_4h
            bias_bearish = series_4h[-1].close < ema_4h

            # 2. Intraday Momentum (1h)
            rsi_1h = ctx.mtf_rsi("1h", 14)
            if rsi_1h is None: return False
            
            # 3. Micro Trigger (1m/5m)
            cur = ctx.current_candle
            if not cur: return False
            
            vol_z = ctx.volume_zscore(20)
            atr_pct = ctx.atr_percent() # Average move in %

            if bias_bullish and rsi_1h > 55 and vol_z > 2.0:
                # Check for 1m breakout of recent high
                recent_high = max(c.high for c in ctx.candles[-10:-1])
                if cur.close > recent_high:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    # Set stop distance so that 2R = 1% minimum
                    # If 1% target is 2R, then R = 0.5%
                    ctx.stop_distance = max(cur.close * 0.005, ctx.atr() * 1.5)
                    return True

            if bias_bearish and rsi_1h < 45 and vol_z > 2.0:
                recent_low = min(c.low for c in ctx.candles[-10:-1])
                if cur.close < recent_low:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = max(cur.close * 0.005, ctx.atr() * 1.5)
                    return True

            return False

        @cls.register('ignition_momentum_1pct_edge')
        def ignition_momentum_1pct_edge(ctx):
            # Look for a volatility contraction followed by an expansion
            if len(ctx.candles) < 30: return False
            
            # Volatility contraction: ATR(20) < ATR(50)
            atr_20 = ctx.atr(20)
            atr_50 = ctx.atr(50)
            if atr_20 >= atr_50: return False
            
            # Expansion Trigger: Volume Surge + Price Displacement
            cur = ctx.current_candle
            prev = ctx.previous_candle
            vol_z = ctx.volume_zscore(14)
            
            if vol_z > 2.5: # Extreme volume surge
                move_pct = abs(cur.close - prev.close) / prev.close
                if move_pct > 0.002: # 0.2% move in a single 1m candle
                    ctx.direction = "long" if cur.close > prev.close else "short"
                    ctx.entry_price = cur.close
                    # We target a 1% move, so we set stop at 0.5% for 1:2 RR
                    ctx.stop_distance = cur.close * 0.005 
                    return True
            
            return False

        @cls.register('institutional_edge_sweep_mss')
        def institutional_edge_sweep_mss(ctx):
            """
            Institutional Sweep + MSS (Market Structure Shift)
            1. Sweep of 15m/1h Swing Level.
            2. Break of 1m/5m Structure in opposite direction.
            3. High Volume Confirmation.
            Targets a minimum 1% move with 1:2 RR.
            """
            # 1. Macro Bias (1h)
            ema_1h = ctx.mtf_ema("1h", 50)
            if ema_1h is None: return False
            series_1h = ctx._mtf_series_up_to("1h")
            if not series_1h: return False
            bias = "long" if series_1h[-1].close > ema_1h else "short"

            # 2. Find 15m Liquidity Level
            series_15m = ctx._mtf_series_up_to("15m")
            if len(series_15m) < 25: return False
            # Find recent swing points on 15m
            sp_15m = StructureDetector.swing_points(series_15m, lookback=5)
            if not sp_15m["highs"] or not sp_15m["lows"]: return False
            
            last_15m_high = sp_15m["highs"][-1]["price"]
            last_15m_low = sp_15m["lows"][-1]["price"]

            # 3. Detect Sweep (on 1m timeframe)
            cur = ctx.current_candle
            prev = ctx.previous_candle
            if not cur or not prev: return False
            
            sweep_long = False
            sweep_short = False
            
            # Sweep of 15m Low (potential Long)
            if prev.low < last_15m_low and prev.close > last_15m_low:
                sweep_long = True
            
            # Sweep of 15m High (potential Short)
            if prev.high > last_15m_high and prev.close < last_15m_high:
                sweep_short = True
            
            if not sweep_long and not sweep_short:
                # Also check if we are CURRENTLY sweeping and looking for MSS
                # We'll use a slightly larger window for the sweep check
                past_5 = ctx.candles[-6:-1]
                if any(c.low < last_15m_low and c.close > last_15m_low for c in past_5):
                    sweep_long = True
                if any(c.high > last_15m_high and c.close < last_15m_high for c in past_5):
                    sweep_short = True

            if not sweep_long and not sweep_short: return False

            # 4. MSS Trigger (1m)
            # Find recent 1m swing point to break
            sp_1m = ctx.swing_points(lookback=3)
            vol_z = ctx.volume_zscore(20)
            
            if sweep_long and bias == "long":
                if not sp_1m["highs"]: return False
                last_1m_high = sp_1m["highs"][-1]["price"]
                if cur.close > last_1m_high and vol_z > 1.5:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    # Targeting 1% move => 0.5% SL for 1:2 RR
                    ctx.stop_distance = max(cur.close * 0.005, cur.close - prev.low)
                    return True

            if sweep_short and bias == "short":
                if not sp_1m["lows"]: return False
                last_1m_low = sp_1m["lows"][-1]["price"]
                if cur.close < last_1m_low and vol_z > 1.5:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    ctx.stop_distance = max(cur.close * 0.005, prev.high - cur.close)
                    return True

            return False

        @cls.register('generic_breakout')
        def generic_breakout(ctx):
            sp = ctx.swing_points()
            if not sp['highs'] or not sp['lows'] or not ctx.current_candle or not ctx.previous_candle:
                return False
            
            last_high = sp['highs'][-1]['price']
            last_low = sp['lows'][-1]['price']
            cur = ctx.current_candle.close
            prev = ctx.previous_candle.close

            if cur > last_high and prev <= last_high:
                ctx.direction = "long"
                ctx.entry_price = cur
                ctx.stop_distance = ctx.atr() * 1.5
                return True
            elif cur < last_low and prev >= last_low:
                ctx.direction = "short"
                ctx.entry_price = cur
                ctx.stop_distance = ctx.atr() * 1.5
                return True
            return False

        @cls.register('advanced_mtf_trend_alignment_entry')
        def advanced_mtf_trend_alignment_entry(ctx):
            # 1. Macro Trend Alignment (1d, 4h, 1h)
            # Use EMA 50 to define trend. All 3 must align.
            tfs_macro = ["1d", "4h", "1h"]
            macro_trend = None
            
            for tf in tfs_macro:
                ema = ctx.mtf_ema(tf, 50)
                series = ctx._mtf_series_up_to(tf)
                if not series or ema is None: return False
                curr_price = series[-1].close
                
                tf_trend = "long" if curr_price > ema else "short"
                if macro_trend is None:
                    macro_trend = tf_trend
                elif macro_trend != tf_trend:
                    return False # Trends must align across 1d, 4h, 1h

            # 2. Intermediate Structure Confirmation (15m or 5m)
            # Wait for a breakout in the direction of the macro trend
            struct_confirmed = False
            for tf in ["15m", "5m"]:
                series = ctx._mtf_series_up_to(tf)
                if not series or len(series) < 5: continue
                # Structure shift / breakout: close > highest high of previous 4 candles
                recent_high = max(c.high for c in series[-5:-1])
                recent_low = min(c.low for c in series[-5:-1])
                curr_close = series[-1].close
                
                if macro_trend == "long" and curr_close > recent_high:
                    struct_confirmed = True
                    break
                elif macro_trend == "short" and curr_close < recent_low:
                    struct_confirmed = True
                    break
                    
            if not struct_confirmed:
                return False

            # 3. Fine-grain Entry Trigger (1m)
            # Entry on 1m following a candlestick pattern (Engulfing)
            if not ctx.current_candle or not ctx.previous_candle: return False
            cur = ctx.current_candle
            prev = ctx.previous_candle
            
            atr = ctx.atr()
            if atr == 0: return False

            # Bullish Engulfing for Long
            if macro_trend == "long":
                if prev.close < prev.open and cur.close > cur.open and cur.close > prev.open and cur.open < prev.close:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    # SL below recent swing low on 1m
                    recent_1m_low = min(c.low for c in ctx.candles[-5:])
                    ctx.stop_distance = cur.close - recent_1m_low
                    if ctx.stop_distance <= 0: ctx.stop_distance = atr * 1.5
                    return True

            # Bearish Engulfing for Short
            elif macro_trend == "short":
                if prev.close > prev.open and cur.close < cur.open and cur.close < prev.open and cur.open > prev.close:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    # SL above recent swing high on 1m
                    recent_1m_high = max(c.high for c in ctx.candles[-5:])
                    ctx.stop_distance = recent_1m_high - cur.close
                    if ctx.stop_distance <= 0: ctx.stop_distance = atr * 1.5
                    return True

            return False

        @cls.register('ignition_momentum_continuation_entry')
        def ignition_momentum_continuation_entry(ctx):
            """
            3-Bar Play / Ignition Momentum Continuation
            Identifies a massive institutional momentum candle, a brief resting candle,
            and enters on the continuation breakout. Designed to capture explosive 1%+ moves.
            """
            if len(ctx.candles) < 25: return False
            
            ign = ctx.candles[-3]
            rest = ctx.candles[-2]
            cur = ctx.candles[-1]
            
            closes = [c.close for c in ctx.candles[-23:-3]]
            if len(closes) < 20: return False
            vols = [c.volume for c in ctx.candles[-23:-3]]
            vol_sma = sum(vols) / len(vols)
            
            # Simplified ATR over last 14 periods before ignition
            tr_list = []
            for i in range(len(ctx.candles)-17, len(ctx.candles)-3):
                c = ctx.candles[i]
                p = ctx.candles[i-1]
                tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
                tr_list.append(tr)
            atr = sum(tr_list) / len(tr_list) if tr_list else 0.0
            if atr == 0: return False

            ign_range = ign.high - ign.low
            rest_range = rest.high - rest.low

            # LONG SETUP
            is_bull_ignition = (
                ign_range > 1.5 * atr and 
                ign.close > ign.open and 
                (ign.close - ign.low) > (ign_range * 0.7) and 
                ign.volume > 1.5 * vol_sma
            )
            
            is_bull_rest = (
                rest_range < ign_range * 0.6 and 
                rest.low > ign.low + (ign_range * 0.3) and # Stays in upper part
                rest.volume < ign.volume
            )
            
            if is_bull_ignition and is_bull_rest:
                if cur.close > rest.high:
                    ctx.direction = "long"
                    ctx.entry_price = cur.close
                    # Set minimum stop distance to 0.75% so 1.5R target = >1.1% move
                    min_stop = cur.close * 0.0075 
                    structural_stop = cur.close - rest.low
                    ctx.stop_distance = max(structural_stop, min_stop)
                    return True

            # SHORT SETUP
            is_bear_ignition = (
                ign_range > 1.5 * atr and 
                ign.close < ign.open and 
                (ign.high - ign.close) > (ign_range * 0.7) and 
                ign.volume > 1.5 * vol_sma
            )
            
            is_bear_rest = (
                rest_range < ign_range * 0.6 and 
                rest.high < ign.high - (ign_range * 0.3) and # Stays in lower part
                rest.volume < ign.volume
            )
            
            if is_bear_ignition and is_bear_rest:
                if cur.close < rest.low:
                    ctx.direction = "short"
                    ctx.entry_price = cur.close
                    min_stop = cur.close * 0.0075
                    structural_stop = rest.high - cur.close
                    ctx.stop_distance = max(structural_stop, min_stop)
                    return True
                    
            return False

# Load defaults when the module is imported
ConditionRegistry.load_defaults()
