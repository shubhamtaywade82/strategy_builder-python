import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import ta

# Assuming we have access to the data fetcher from our research folder
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "research"))
from mtf_research import BinanceUMKlineLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("TradingBot")

# ==============================================================================
# STRATEGY PARAMETERS
# ==============================================================================
SYMBOL = "SOLUSDT"
LEVERAGE = 10
RISK_PCT = 0.02
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.010

def calc_ema(series, period):
    return ta.trend.ema_indicator(series, window=period)

def calc_atr(df, period=14):
    return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)

def detect_fvg(df):
    """
    Returns Bullish FVG (1) if low[i] > high[i-2]
    Returns Bearish FVG (-1) if high[i] < low[i-2]
    """
    fvg = np.zeros(len(df))
    lows = df['low'].values
    highs = df['high'].values
    for i in range(2, len(df)):
        if lows[i] > highs[i-2]:  # Bullish FVG
            fvg[i] = 1
        elif highs[i] < lows[i-2]:  # Bearish FVG
            fvg[i] = -1
    return fvg

def run_bot():
    log.info("="*60)
    log.info(f"🚀 INITIALIZING LIVE TRADING BOT | {SYMBOL} 🚀")
    log.info("="*60)
    log.info(f"Target: +{TAKE_PROFIT_PCT*100}% | Stop: -{STOP_LOSS_PCT*100}% | Risk: {RISK_PCT*100}% | Lev: {LEVERAGE}x")
    log.info("Checking BOTH sides (Long / Short) using the 5-Condition Rule.")
    
    loader = BinanceUMKlineLoader()
    
    while True:
        try:
            # 1. Fetch data
            end_ms = int(time.time() * 1000)
            start_ms_1m = end_ms - (1440 * 60_000)    # 1 day of 1m
            start_ms_15m = end_ms - (100 * 900_000)   # 100 15m bars
            start_ms_1h = end_ms - (100 * 3_600_000)  # 100 1h bars
            start_ms_4h = end_ms - (300 * 14_400_000) # 300 4h bars
            
            df_1m = loader.fetch(SYMBOL, "1m", start_ms_1m, end_ms)
            df_15m = loader.fetch(SYMBOL, "15m", start_ms_15m, end_ms)
            df_1h = loader.fetch(SYMBOL, "1h", start_ms_1h, end_ms)
            df_4h = loader.fetch(SYMBOL, "4h", start_ms_4h, end_ms)
            
            if df_1m.empty or df_4h.empty:
                time.sleep(5)
                continue
                
            # Current Price
            current_price = df_1m['close'].iloc[-1]
            
            # --- CONDITION 1: 4H Trend ---
            df_4h['ema50'] = calc_ema(df_4h['close'], 50)
            df_4h['ema200'] = calc_ema(df_4h['close'], 200)
            c1_long = df_4h['ema50'].iloc[-1] > df_4h['ema200'].iloc[-1]
            c1_short = df_4h['ema50'].iloc[-1] < df_4h['ema200'].iloc[-1]
            
            # --- CONDITION 2: 1H Break of Structure (BOS) ---
            # Recent highest high / lowest low over last 5 bars
            recent_high = df_1h['high'].iloc[-6:-1].max()
            recent_low = df_1h['low'].iloc[-6:-1].min()
            c2_long = df_1h['close'].iloc[-1] > recent_high
            c2_short = df_1h['close'].iloc[-1] < recent_low
            
            # --- CONDITION 3: 15m Fair Value Gap ---
            df_15m['fvg'] = detect_fvg(df_15m)
            # Check if any bullish FVG occurred in the last 10 bars
            c3_long = (df_15m['fvg'].iloc[-10:] == 1).any()
            c3_short = (df_15m['fvg'].iloc[-10:] == -1).any()
            
            # --- CONDITION 4: Volatility (ATR 60th Percentile) ---
            df_1m['atr'] = calc_atr(df_1m, 14)
            atr_history = df_1m['atr'].dropna().values
            atr_60th = np.percentile(atr_history, 60)
            current_atr = atr_history[-1]
            c4_active = current_atr > atr_60th
            
            # --- CONDITION 5: Volume (70th Percentile) ---
            vol_history = df_1m['volume'].dropna().values
            vol_70th = np.percentile(vol_history, 70)
            current_vol = vol_history[-1]
            c5_active = current_vol > vol_70th
            
            # --- EVALUATE SIGNALS ---
            is_long = c1_long and c2_long and c3_long and c4_active and c5_active
            is_short = c1_short and c2_short and c3_short and c4_active and c5_active
            
            ts_str = df_1m['open_time'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
            log.info(f"[{ts_str}] Price: ${current_price:.2f} | C1(4H) L:{c1_long} S:{c1_short} | C2(1H) L:{c2_long} S:{c2_short} | Vol:{c5_active}")
            
            if is_long:
                log.warning(f"🟢 LONG SIGNAL GENERATED!")
                log.warning(f"Entry: ${current_price:.2f} | Stop: ${(current_price * (1-STOP_LOSS_PCT)):.2f} | Target: ${(current_price * (1+TAKE_PROFIT_PCT)):.2f}")
                time.sleep(60) # Prevent multiple triggers in the same minute
                
            elif is_short:
                log.warning(f"🔴 SHORT SIGNAL GENERATED!")
                log.warning(f"Entry: ${current_price:.2f} | Stop: ${(current_price * (1+STOP_LOSS_PCT)):.2f} | Target: ${(current_price * (1-TAKE_PROFIT_PCT)):.2f}")
                time.sleep(60)
                
            # Wait for next minute boundary
            seconds_past_minute = time.time() % 60
            sleep_time = 61 - seconds_past_minute
            time.sleep(sleep_time)
            
        except Exception as e:
            log.error(f"Error in loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()
