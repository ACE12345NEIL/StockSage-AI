import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import math
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Price_vs_SMA20", "Price_vs_SMA50",
    "BB_Position", "BB_Width",
    "Volume_Ratio", "High_Low_Range",
    "Volatility_20", "Daily_Return",
]

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def compute_bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, sma, lower

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["EMA_12"]  = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = close.ewm(span=26, adjust=False).mean()
    df["RSI"]     = compute_rsi(close)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger(close)
    df["Daily_Return"]    = close.pct_change()
    df["Volatility_20"]   = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    df["Price_vs_SMA20"]  = (close - df["SMA_20"]) / df["SMA_20"]
    df["Price_vs_SMA50"]  = (close - df["SMA_50"]) / df["SMA_50"]
    df["Volume_SMA20"]    = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"]    = df["Volume"] / (df["Volume_SMA20"] + 1e-9)
    df["High_Low_Range"]  = (df["High"] - df["Low"]) / (close + 1e-9)
    df["BB_Width"]        = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Mid"] + 1e-9)
    df["BB_Position"]     = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)
    return df

def generate_forecasts(ticker):
    t = yf.Ticker(ticker)
    df_raw = t.history(period="5y")
    if df_raw.empty:
        print(f"No data for {ticker}")
        return
    
    df = add_indicators(df_raw.copy())
    df = df.dropna(subset=FEATURE_COLS)
    
    current_price = float(df["Close"].iloc[-1])
    print(f"Current Price: {current_price:.2f}")
    
    # Horizons in business days: 1 week (5), 1 month (21), 3 months (63), 6 months (126), 1 year (252)
    horizons = {
        "1w": 5,
        "1m": 21,
        "3m": 63,
        "6m": 126,
        "1y": 252
    }
    
    predictions = {}
    
    for label, days in horizons.items():
        df_temp = df.copy()
        # Target is pct change in 'days' days
        df_temp["Target"] = (df_temp["Close"].shift(-days) - df_temp["Close"]) / df_temp["Close"]
        df_temp = df_temp.dropna(subset=["Target"])
        
        if len(df_temp) < 100:
            print(f"Too few samples for horizon {label}")
            predictions[label] = {"pct_change": 0.0, "price": current_price}
            continue
            
        X = df_temp[FEATURE_COLS].values
        y = df_temp["Target"].values
        
        # Use Ridge regression as a robust, non-overfitting linear estimator for long terms,
        # or XGBoost regressor. Let's see how Ridge behaves.
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Predict on latest data point
        latest_X = df[FEATURE_COLS].iloc[[-1]].values
        pred_pct = float(model.predict(latest_X)[0])
        pred_price = current_price * (1 + pred_pct)
        
        predictions[label] = {
            "pct_change": round(pred_pct * 100, 2),
            "price": round(pred_price, 2)
        }
        
    print(predictions)

if __name__ == "__main__":
    generate_forecasts("AAPL")
    generate_forecasts("WIPRO.NS")
