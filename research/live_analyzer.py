import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from run_research import build_dataset, select_feature_columns
from binance_meta import load_meta
from mtf_research import BinanceUMKlineLoader, build_mtf_features

def analyze_live(symbol="SOLUSDT"):
    meta = load_meta(symbol)
    
    # Run the standard build_dataset to get labels for training
    print(f"Fetching historical data and training models for {symbol}...")
    ds = build_dataset(symbol, days=45, meta=meta)
    feature_cols = select_feature_columns(ds)
    
    # Train the surrogate trees
    X_train = ds[feature_cols].to_numpy(float)
    y_long = ds["long_label"].to_numpy(int)
    y_short = ds["short_label"].to_numpy(int)
    
    dt_long = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
    dt_long.fit(X_train, y_long)
    
    dt_short = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
    dt_short.fit(X_train, y_short)
    
    print("\n" + "="*50)
    print("🤖 LIVE AI ANALYSIS ENGINE (Continuous Mode)")
    print("="*50)
    
    loader = BinanceUMKlineLoader()
    
    while True:
        # 1. Fetch data up to the current minute
        end = int(time.time() * 1000)
        start = end - 45 * 86_400_000
        frames = {tf: loader.fetch(symbol, tf, start, end) for tf in ["1m", "5m", "1h", "4h", "1d"]}
        feats = build_mtf_features(frames, base_tf="1m")
        # Filter features to only make decisions exactly at the close of 5m candles
        # (A 5m candle closes when the minute is 4, 9, 14, 19, etc. so entry is exactly at 0, 5, 10...)
        feats = feats[feats['open_time'].dt.minute % 5 == 4].copy()
        feats = feats.dropna().reset_index(drop=True)
        
        # 2. Extract the very last row (the current live minute)
        last_row_features = feats[feature_cols].iloc[-1].to_numpy(float).reshape(1, -1)
        
        pred_long = dt_long.predict(last_row_features)[0]
        pred_short = dt_short.predict(last_row_features)[0]
        
        # 3. Print the live signal
        current_time_str = str(feats['open_time'].iloc[-1])
        current_price = feats['close'].iloc[-1]
        
        print(f"\n[{current_time_str} UTC] | {symbol} Price: ${current_price:.2f}")
        
        if pred_long == 1:
            print("   🟢 ACTION: GO LONG (Target: +1.0% Price Move)")
        elif pred_short == 1:
            print("   🔴 ACTION: GO SHORT (Target: +1.0% Price Move)")
        else:
            print("   ⚪ ACTION: AVOID (No Edge)")
            
        # 4. Wait until the exact boundary of the next 5-minute candle
        current_seconds = time.time()
        # Find seconds past the last 5-minute boundary
        seconds_into_5m = current_seconds % 300
        # Wait until the next 5m boundary plus 1 second
        sleep_time = 301 - seconds_into_5m
        
        minutes_wait = int(sleep_time // 60)
        seconds_wait = int(sleep_time % 60)
        print(f"   [Sleeping for {minutes_wait}m {seconds_wait}s until the next 5-minute candle close...]")
        time.sleep(sleep_time)

if __name__ == "__main__":
    analyze_live("SOLUSDT")
