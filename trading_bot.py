import time
import logging
import requests
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
LEVERAGE = 10
RISK_PCT = 0.02
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.010

def get_tradable_symbols(top_n=4):
    # 1. Get valid perpetual symbols from exchangeInfo
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    valid_symbols = set()
    for s in data['symbols']:
        if s['status'] == 'TRADING' and s['contractType'] == 'PERPETUAL' and s['quoteAsset'] == 'USDT':
            valid_symbols.add(s['symbol'])

    # 2. Get 24hr ticker to sort by volume
    ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    tr = requests.get(ticker_url)
    tr.raise_for_status()
    tickers = tr.json()

    # Filter and sort
    filtered = [t for t in tickers if t['symbol'] in valid_symbols]
    filtered.sort(key=lambda x: float(x['quoteVolume']), reverse=True)

    return [t['symbol'] for t in filtered[:top_n]]

def calc_ema(series, period):
    return ta.trend.ema_indicator(series, window=period)

def calc_atr(df, period=14):
    return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)

def detect_fvg(df):
    fvg = np.zeros(len(df))
    lows = df['low'].values
    highs = df['high'].values
    for i in range(2, len(df)):
        if lows[i] > highs[i-2]:  # Bullish FVG
            fvg[i] = 1
        elif highs[i] < lows[i-2]:  # Bearish FVG
            fvg[i] = -1
    return fvg

def update_cache(loader, symbol, interval, end_ms, required_bars, cache_dict):
    step_ms = {
        "1m": 60_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000
    }[interval]

    key = f"{symbol}_{interval}"

    log.info(key)
    if key not in cache_dict:
        start_ms = end_ms - (required_bars * step_ms)
        df = loader.fetch(symbol, interval, start_ms, end_ms)
        cache_dict[key] = df
    else:
        df = cache_dict[key]
        last_time = int(df['open_time'].iloc[-1].timestamp() * 1000)
        # Fetch only what's missing plus current open candle (so basically from last_time)
        start_ms = last_time
        new_df = loader.fetch(symbol, interval, start_ms, end_ms)
        if not new_df.empty:
            df = pd.concat([df, new_df])
            df = df.drop_duplicates(subset=['open_time'], keep='last').sort_values('open_time')
            # Trim to required size
            df = df.iloc[-required_bars:]
            cache_dict[key] = df

    return cache_dict[key]

def run_bot():
    symbols = get_tradable_symbols()
    log.info("="*60)
    log.info(f"🚀 INITIALIZING LIVE TRADING BOT | {len(symbols)} SYMBOLS 🚀")
    log.info("="*60)
    log.info(f"Target: +{TAKE_PROFIT_PCT*100}% | Stop: -{STOP_LOSS_PCT*100}% | Risk: {RISK_PCT*100}% | Lev: {LEVERAGE}x")
    log.info("Checking BOTH sides (Long / Short) using the 5-Condition Rule.")

    loader = BinanceUMKlineLoader()
    cache = {}

    while True:
        try:
            for symbol in symbols:
                try:
                    end_ms = int(time.time() * 1000)

                    # 1. Update data incrementally
                    df_1m = update_cache(loader, symbol, "1m", end_ms, 1440, cache)
                    df_15m = update_cache(loader, symbol, "15m", end_ms, 100, cache)
                    df_1h = update_cache(loader, symbol, "1h", end_ms, 100, cache)
                    df_4h = update_cache(loader, symbol, "4h", end_ms, 300, cache)

                    if df_1m.empty or df_4h.empty or df_1h.empty or df_15m.empty:
                        continue

                    current_price = df_1m['close'].iloc[-1]

                    # --- CONDITION 1: 4H Trend ---
                    df_4h_c = df_4h.copy()
                    df_4h_c['ema50'] = calc_ema(df_4h_c['close'], 50)
                    df_4h_c['ema200'] = calc_ema(df_4h_c['close'], 200)
                    c1_long = df_4h_c['ema50'].iloc[-1] > df_4h_c['ema200'].iloc[-1]
                    c1_short = df_4h_c['ema50'].iloc[-1] < df_4h_c['ema200'].iloc[-1]

                    # --- CONDITION 2: 1H Break of Structure (BOS) ---
                    recent_high = df_1h['high'].iloc[-6:-1].max()
                    recent_low = df_1h['low'].iloc[-6:-1].min()
                    c2_long = df_1h['close'].iloc[-1] > recent_high
                    c2_short = df_1h['close'].iloc[-1] < recent_low

                    # --- CONDITION 3: 15m Fair Value Gap ---
                    df_15m_c = df_15m.copy()
                    df_15m_c['fvg'] = detect_fvg(df_15m_c)
                    c3_long = (df_15m_c['fvg'].iloc[-10:] == 1).any()
                    c3_short = (df_15m_c['fvg'].iloc[-10:] == -1).any()

                    # --- CONDITION 4: Volatility (ATR 60th Percentile) ---
                    df_1m_c = df_1m.copy()
                    df_1m_c['atr'] = calc_atr(df_1m_c, 14)
                    atr_history = df_1m_c['atr'].dropna().values
                    if len(atr_history) == 0:
                        continue
                    atr_60th = np.percentile(atr_history, 60)
                    current_atr = atr_history[-1]
                    c4_active = current_atr > atr_60th

                    # --- CONDITION 5: Volume (70th Percentile) ---
                    vol_history = df_1m_c['volume'].dropna().values
                    if len(vol_history) == 0:
                        continue
                    vol_70th = np.percentile(vol_history, 70)
                    current_vol = vol_history[-1]
                    c5_active = current_vol > vol_70th

                    # --- EVALUATE SIGNALS ---
                    is_long = c1_long and c2_long and c3_long and c4_active and c5_active
                    is_short = c1_short and c2_short and c3_short and c4_active and c5_active

                    if is_long or is_short:
                        ts_str = df_1m['open_time'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
                        log.info(f"[{ts_str}] {symbol} | Price: ${current_price:.2f} | C1(4H) L:{c1_long} S:{c1_short} | C2(1H) L:{c2_long} S:{c2_short} | Vol:{c5_active}")

                        if is_long:
                            log.warning(f"🟢 LONG SIGNAL GENERATED for {symbol}!")
                            log.warning(f"Entry: ${current_price:.2f} | Stop: ${(current_price * (1-STOP_LOSS_PCT)):.2f} | Target: ${(current_price * (1+TAKE_PROFIT_PCT)):.2f}")

                        elif is_short:
                            log.warning(f"🔴 SHORT SIGNAL GENERATED for {symbol}!")
                            log.warning(f"Entry: ${current_price:.2f} | Stop: ${(current_price * (1+STOP_LOSS_PCT)):.2f} | Target: ${(current_price * (1-TAKE_PROFIT_PCT)):.2f}")

                except Exception as e:
                    log.error(f"Error processing {symbol}: {e}")

            seconds_past_minute = time.time() % 60
            sleep_time = 61 - seconds_past_minute
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            log.error(f"Error in main loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()
