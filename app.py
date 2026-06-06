"""
AI-Powered Stock Analysis & Prediction System
Backend: Flask + yfinance + XGBoost + VADER Sentiment
Live news only (CSV dataset is loaded for optional offline training)
Author: College Project Demo
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
import json
import math
import os
warnings.filterwarnings("ignore")

# ── ML ─────────────────────────────────────────────────────────────────────────
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

# ── Sentiment ──────────────────────────────────────────────────────────────────
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
CORS(app)

# ── Custom JSON encoder: sanitise NaN / Inf / numpy scalars ───────────────────
class SafeJSONProvider(app.json_provider_class):
    def dumps(self, obj, **kw):
        return json.dumps(obj, default=_json_default, allow_nan=False, **kw)
    def loads(self, s, **kw):
        return json.loads(s, **kw)

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is pd.NaT or (isinstance(obj, float) and math.isnan(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

app.json_provider_class = SafeJSONProvider
app.json = SafeJSONProvider(app)

vader = SentimentIntensityAnalyzer()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD HISTORICAL NEWS CSV (for OFFLINE TRAINING only – not used in live API)
# ══════════════════════════════════════════════════════════════════════════════
NEWS_CSV_PATH = "data/combined_news.csv"
historical_news_df = None

def load_historical_news():
    global historical_news_df
    if not os.path.exists(NEWS_CSV_PATH):
        print("⚠️ Historical news CSV not found. (Will use only live news)")
        return False
    try:
        df = pd.read_csv(NEWS_CSV_PATH, parse_dates=["date"])
        df = df.dropna(subset=["title", "description"])
        df["timestamp"] = df["date"].apply(lambda x: int(x.timestamp()) if pd.notnull(x) else 0)
        df["categories"] = df["categories"].fillna("")
        df["matched_keywords"] = df["matched_keywords"].fillna("").str.lower()
        df["relevance_score"] = df["relevance_score"].fillna(0.5)
        df["impact_tier"] = df["impact_tier"].fillna("MEDIUM")
        df["has_negation"] = df["has_negation"].fillna(False)
        
        # Pre‑compute VADER for potential offline training
        print("🔄 Pre‑computing VADER sentiment for historical CSV (offline use)...")
        vader_static = SentimentIntensityAnalyzer()
        def get_vader_score(row):
            text = f"{row['title']} {row['description']}"
            return vader_static.polarity_scores(text)["compound"]
        df["base_vader"] = df.apply(get_vader_score, axis=1)
        
        historical_news_df = df
        print(f"✅ Loaded {len(df)} historical news articles (available for offline training)")
        return True
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

load_historical_news()

# ══════════════════════════════════════════════════════════════════════════════
# LIVE NEWS FUNCTIONS (Original – used for all API responses)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_news(ticker: str, company_name: str) -> list[dict]:
    """Fetch live news from yfinance."""
    articles = []
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news or []
        for item in raw_news[:15]:
            content = item.get("content") or {}
            if content and isinstance(content, dict):
                title   = content.get("title", "")
                summary = content.get("summary", "") or content.get("description", "")
                url_obj = content.get("canonicalUrl") or {}
                url     = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                prov_obj = content.get("provider") or {}
                source   = prov_obj.get("displayName", "") if isinstance(prov_obj, dict) else str(prov_obj)
                time_val = content.get("pubDate", 0)
            else:
                title   = item.get("title", "")
                summary = item.get("summary", "") or item.get("description", "")
                url     = item.get("link", "") or item.get("url", "")
                source  = item.get("publisher", "") or item.get("source", "")
                time_val = item.get("providerPublishTime", 0)
            if title:
                articles.append({
                    "title": title, "summary": summary, "url": url,
                    "source": source, "time": time_val,
                })
    except Exception:
        pass
    return articles

def fetch_sector_news(sector: str, industry: str, is_indian: bool) -> list[dict]:
    """Fetch sector/industry news via Google News RSS feed."""
    if not sector and not industry:
        return []
    term = industry if industry else sector
    query = f"Indian {term} industry news" if is_indian else f"{term} industry news"
    articles = []
    try:
        import urllib.parse, xml.etree.ElementTree as ET
        encoded_query = urllib.parse.quote(query)
        hl = "en-IN" if is_indian else "en-US"
        gl = "IN" if is_indian else "US"
        ceid = "IN:en" if is_indian else "US:en"
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl={hl}&gl={gl}&ceid={ceid}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall(".//item")[:15]:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                pubDate = item.find("pubDate").text if item.find("pubDate") is not None else ""
                source = item.find("source").text if item.find("source") is not None else ""
                time_val = pubDate
                try:
                    dt = datetime.strptime(pubDate, "%a, %d %b %Y %H:%M:%S %Z")
                    time_val = int(dt.timestamp())
                except Exception:
                    pass
                cleaned_title = title.split(" - ")[0] if " - " in title else title
                articles.append({
                    "title": cleaned_title, "summary": cleaned_title, "url": link,
                    "source": source, "time": time_val,
                })
    except Exception:
        pass
    return articles

def analyze_sentiment(articles: list[dict]) -> dict:
    """VADER sentiment for live news."""
    if not articles:
        return {
            "avg_score": 0.0, "positive_ratio": 0.0, "negative_ratio": 0.0,
            "neutral_ratio": 1.0, "news_impact_score": 0.0, "article_count": 0,
            "summary": "No recent news available — using neutral sentiment.",
        }
    scores, pos, neg, neu = [], 0, 0, 0
    for a in articles:
        text = f"{a['title']} {a['summary']}"
        s = vader.polarity_scores(text)
        compound = s["compound"]
        scores.append(compound)
        if compound >= 0.05:   pos += 1
        elif compound <= -0.05: neg += 1
        else:                   neu += 1
    n = len(scores)
    avg = np.mean(scores)
    weights = np.exp(-np.arange(n) * 0.1)
    impact = float(np.average(np.abs(scores), weights=weights[:n]))
    return {
        "avg_score": round(float(avg), 4),
        "positive_ratio": round(pos / n, 4),
        "negative_ratio": round(neg / n, 4),
        "neutral_ratio": round(neu / n, 4),
        "news_impact_score": round(impact, 4),
        "article_count": n,
        "articles": articles[:8],
        "summary": f"Analyzed {n} articles — {pos} positive, {neg} negative, {neu} neutral.",
    }

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Ticker Resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_ticker(query: str) -> dict:
    query = query.strip().upper()
    indian_map = {
        "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFOSYS": "INFY.NS",
        "INFY": "INFY.NS", "HDFC": "HDFCBANK.NS", "HDFCBANK": "HDFCBANK.NS",
        "WIPRO": "WIPRO.NS", "ICICIBANK": "ICICIBANK.NS", "ICICI": "ICICIBANK.NS",
        "TATAMOTORS": "TATAMOTORS.NS", "TATA": "TATAMOTORS.NS",
        "BAJFINANCE": "BAJFINANCE.NS", "SBIN": "SBIN.NS", "SBI": "SBIN.NS",
        "ADANI": "ADANIENT.NS", "ADANIENT": "ADANIENT.NS",
        "SUNPHARMA": "SUNPHARMA.NS", "ITCLTD": "ITC.NS", "ITC": "ITC.NS",
        "MARUTI": "MARUTI.NS", "ASIANPAINT": "ASIANPAINT.NS",
        "TITAN": "TITAN.NS", "NESTLEIND": "NESTLEIND.NS", "LT": "LT.NS",
        "KOTAKBANK": "KOTAKBANK.NS", "AXISBANK": "AXISBANK.NS",
        "BHARTIARTL": "BHARTIARTL.NS", "AIRTEL": "BHARTIARTL.NS",
        "HINDUNILVR": "HINDUNILVR.NS", "HUL": "HINDUNILVR.NS",
        "POWERGRID": "POWERGRID.NS", "NTPC": "NTPC.NS",
        "ONGC": "ONGC.NS", "COALINDIA": "COALINDIA.NS",
        "TECHM": "TECHM.NS", "HCLTECH": "HCLTECH.NS",
        "DRREDDY": "DRREDDY.NS", "CIPLA": "CIPLA.NS",
        "BAJAJ-AUTO": "BAJAJ-AUTO.NS", "BAJAJ": "BAJAJ-AUTO.NS",
        "ULTRACEMCO": "ULTRACEMCO.NS", "GRASIM": "GRASIM.NS",
    }
    candidates = [query]
    if query in indian_map:
        candidates.insert(0, indian_map[query])
    if not query.endswith((".NS", ".BO", ".L", ".AX")):
        candidates += [query + ".NS", query + ".BO"]
    for sym in candidates:
        try:
            t = yf.Ticker(sym)
            info = t.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price and price > 0:
                name = getattr(t.info, "longName", None) or getattr(t.info, "shortName", sym) or sym
                try:
                    full = t.info
                    name = full.get("longName") or full.get("shortName") or sym
                    exch = full.get("exchange", "")
                    curr = full.get("currency", "USD")
                except Exception:
                    exch, curr = "", "USD"
                return {"ticker": sym, "name": name, "exchange": exch, "currency": curr, "found": True}
        except Exception:
            continue
    return {"ticker": query, "name": query, "exchange": "", "currency": "USD", "found": False}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Technical Indicators
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: ML Prediction (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "Price_vs_SMA20", "Price_vs_SMA50",
    "BB_Position", "BB_Width",
    "Volume_Ratio", "High_Low_Range",
    "Volatility_20", "Daily_Return",
    "EMA_12", "EMA_26",
]

def build_ml_features(df: pd.DataFrame, sentiment: dict) -> pd.DataFrame:
    feat = df[FEATURE_COLS].copy()
    feat["sentiment_score"]  = sentiment["avg_score"]
    feat["positive_ratio"]   = sentiment["positive_ratio"]
    feat["negative_ratio"]   = sentiment["negative_ratio"]
    feat["news_impact"]      = sentiment["news_impact_score"]
    return feat

def train_and_predict(df: pd.DataFrame, sentiment: dict) -> dict:
    df2 = df.copy()
    df2 = df2.dropna(subset=FEATURE_COLS)
    df2["Target"] = (df2["Close"].shift(-1) > df2["Close"]).astype(int)
    df2 = df2.dropna(subset=["Target"])

    feat = build_ml_features(df2, sentiment)
    X = feat.values
    y = df2["Target"].values

    if len(X) < 60:
        return {
            "prob_increase": 0.5, "prob_decrease": 0.5,
            "predicted_trend": "NEUTRAL", "confidence": 0.50,
            "model_accuracy": None, "note": "Insufficient data for ML model.",
        }

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_s, y_train,
              eval_set=[(X_test_s, y_test)],
              verbose=False)

    acc = float(model.score(X_test_s, y_test))

    latest_feat = build_ml_features(df2.iloc[[-1]], sentiment)
    latest_scaled = scaler.transform(latest_feat.values)
    proba = model.predict_proba(latest_scaled)[0]

    prob_up   = float(proba[1])
    prob_down = float(proba[0])
    trend     = "BULLISH" if prob_up > 0.55 else ("BEARISH" if prob_down > 0.55 else "NEUTRAL")
    confidence = max(prob_up, prob_down)

    all_feat_names = FEATURE_COLS + ["sentiment_score","positive_ratio","negative_ratio","news_impact"]
    importances = dict(zip(all_feat_names, model.feature_importances_.tolist()))
    top5 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "prob_increase":   round(prob_up,   4),
        "prob_decrease":   round(prob_down, 4),
        "predicted_trend": trend,
        "confidence":      round(confidence, 4),
        "model_accuracy":  round(acc, 4),
        "top_features":    top5,
    }

def train_and_forecast(df: pd.DataFrame, sentiment: dict) -> dict:
    df_clean = df.copy().dropna(subset=FEATURE_COLS)
    if df_clean.empty:
        return {
            label: {"pct_change": 0.0, "price": 0.0, "trend": "FLAT"}
            for label in ["1w", "1m", "3m", "6m", "1y"]
        }
        
    current_price = float(df_clean["Close"].iloc[-1])
    
    horizons = {
        "1w": 5,
        "1m": 21,
        "3m": 63,
        "6m": 126,
        "1y": 252
    }
    
    predictions = {}
    
    for label, days in horizons.items():
        df_temp = df_clean.copy()
        df_temp["Target"] = (df_temp["Close"].shift(-days) - df_temp["Close"]) / df_temp["Close"]
        df_temp = df_temp.dropna(subset=["Target"])
        
        if len(df_temp) < 40:
            recent_return = float(df_clean["Close"].pct_change(periods=min(10, len(df_clean)-1)).iloc[-1])
            pred_pct = recent_return * (days / 10.0)
            pred_pct = max(min(pred_pct, 0.20), -0.20)
        else:
            try:
                feat = build_ml_features(df_temp, sentiment)
                X = feat.values
                y = df_temp["Target"].values
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = Ridge(alpha=10.0)
                model.fit(X_scaled, y)
                
                latest_df = df_clean.iloc[[-1]]
                latest_feat = build_ml_features(latest_df, sentiment)
                latest_scaled = scaler.transform(latest_feat.values)
                
                pred_pct = float(model.predict(latest_scaled)[0])
            except Exception:
                pred_pct = 0.0
        
        max_clip = (days / 252.0) * 1.0
        min_clip = -(days / 252.0) * 0.5
        pred_pct = max(min(pred_pct, max(max_clip, 0.05)), min(min_clip, -0.05))
        
        pred_price = current_price * (1 + pred_pct)
        trend = "UPWARD" if pred_pct > 0.01 else ("DOWNWARD" if pred_pct < -0.01 else "FLAT")
        
        predictions[label] = {
            "pct_change": round(pred_pct * 100, 2),
            "price": round(pred_price, 2),
            "trend": trend
        }
        
    return predictions

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════

def assess_risk(df: pd.DataFrame) -> dict:
    close = df["Close"]
    returns = close.pct_change().dropna()
    vol_annual = float(returns.std() * np.sqrt(252))
    roll_max = close.cummax()
    drawdown = (close - roll_max) / roll_max
    max_dd   = float(drawdown.min())
    avg_ret = float(returns.mean() * 252)
    rf = 0.06 if any(s in df.attrs.get("ticker", "") for s in [".NS", ".BO"]) else 0.04
    sharpe  = (avg_ret - rf) / (vol_annual + 1e-9)
    vol_score = min(vol_annual * 100, 50)
    dd_score  = min(abs(max_dd) * 100, 30)
    sharpe_pen = max(0, (1 - sharpe) * 5)
    risk_score = vol_score + dd_score + sharpe_pen
    risk_score = min(max(risk_score, 0), 100)

    if risk_score < 30:   category = "LOW"
    elif risk_score < 60: category = "MEDIUM"
    else:                 category = "HIGH"

    return {
        "volatility_annual":  round(vol_annual, 4),
        "max_drawdown":       round(max_dd, 4),
        "sharpe_ratio":       round(sharpe, 4),
        "avg_annual_return":  round(avg_ret, 4),
        "risk_score":         round(risk_score, 1),
        "risk_category":      category,
    }

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY: Recommendation Engine
# ══════════════════════════════════════════════════════════════════════════════

def generate_recommendation(ml: dict, sentiment: dict, risk: dict, df: pd.DataFrame) -> dict:
    reasons = []
    score   = 0.0

    ml_weight = 0.40
    ml_signal = ml["prob_increase"] - ml["prob_decrease"]
    score += ml_weight * ml_signal
    if ml["predicted_trend"] == "BULLISH":
        reasons.append(f"ML model predicts upward movement ({ml['prob_increase']*100:.1f}% probability)")
    elif ml["predicted_trend"] == "BEARISH":
        reasons.append(f"ML model predicts downward movement ({ml['prob_decrease']*100:.1f}% probability)")
    else:
        reasons.append("ML model shows mixed signals — sideways movement expected")

    sent_weight = 0.25
    sent_signal = sentiment["avg_score"]
    score += sent_weight * sent_signal

    company_score = sentiment.get("company", {}).get("avg_score", 0.0)
    sector_score = sentiment.get("sector", {}).get("avg_score", 0.0)
    company_count = sentiment.get("company", {}).get("article_count", 0)
    sector_count = sentiment.get("sector", {}).get("article_count", 0)

    if company_count > 0:
        if company_score > 0.05:
            reasons.append(f"Company news sentiment is bullish (score: {company_score:+.2f})")
        elif company_score < -0.05:
            reasons.append(f"Company news sentiment is bearish (score: {company_score:+.2f})")
        else:
            reasons.append(f"Company news sentiment is neutral (score: {company_score:+.2f})")
    else:
        reasons.append("No recent company-specific news available")

    if sector_count > 0:
        if sector_score > 0.05:
            reasons.append(f"Sector news sentiment is bullish (score: {sector_score:+.2f})")
        elif sector_score < -0.05:
            reasons.append(f"Sector news sentiment is bearish (score: {sector_score:+.2f})")
        else:
            reasons.append(f"Sector news sentiment is neutral (score: {sector_score:+.2f})")
    else:
        reasons.append("No recent sector news available")

    tech_weight = 0.25
    latest = df.iloc[-1]
    tech_signal = 0.0

    rsi = latest.get("RSI", 50)
    if rsi < 30:
        tech_signal += 1; reasons.append(f"RSI is oversold ({rsi:.1f}) — potential reversal upward")
    elif rsi > 70:
        tech_signal -= 1; reasons.append(f"RSI is overbought ({rsi:.1f}) — potential pullback")

    if latest.get("MACD", 0) > latest.get("MACD_Signal", 0):
        tech_signal += 0.5; reasons.append("MACD is above signal line — bullish crossover")
    else:
        tech_signal -= 0.5; reasons.append("MACD is below signal line — bearish momentum")

    if latest.get("Price_vs_SMA50", 0) > 0:
        tech_signal += 0.5; reasons.append("Price is trading above 50-day SMA — uptrend")
    else:
        tech_signal -= 0.5; reasons.append("Price is below 50-day SMA — downtrend")

    score += tech_weight * (tech_signal / 2)

    risk_weight = 0.10
    risk_adj = -((risk["risk_score"] - 50) / 100)
    score += risk_weight * risk_adj
    reasons.append(f"Risk category: {risk['risk_category']} (score {risk['risk_score']:.0f}/100)")

    confidence = min(abs(score) * 2, 1.0)
    if score > 0.15:
        action = "BUY"
    elif score < -0.15:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "action":      action,
        "confidence":  round(confidence * 100, 1),
        "score":       round(score, 4),
        "risk_note":   f"Risk is {risk['risk_category'].lower()} — {'exercise caution' if risk['risk_category']=='HIGH' else 'acceptable risk level'}",
        "reasons":     reasons,
    }

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/search", methods=["GET"])
def search_stock():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Query is required"}), 400
    result = resolve_ticker(q)
    return jsonify(result)

@app.route("/api/analyze", methods=["GET"])
def analyze_stock():
    ticker = request.args.get("ticker", "").strip()
    period = request.args.get("period", "1y")
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    try:
        t = yf.Ticker(ticker)
        df_raw = t.history(period="5y", auto_adjust=True)
        if df_raw.empty:
            df_raw = t.history(period=period, auto_adjust=True)
            if df_raw.empty:
                return jsonify({"error": f"No data found for {ticker}"}), 404

        info = {}
        try:
            info = t.info
        except Exception:
            pass

        df = add_indicators(df_raw.copy())
        df.attrs["ticker"] = ticker

        company_name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        is_indian = ticker.endswith((".NS", ".BO"))

        # ── LIVE NEWS ONLY (CSV not used here) ─────────────────────────────────
        company_articles = fetch_news(ticker, company_name)
        sector_articles = fetch_sector_news(sector, industry, is_indian)

        company_sent = analyze_sentiment(company_articles)
        sector_sent = analyze_sentiment(sector_articles)

        has_company = company_sent["article_count"] > 0
        has_sector = sector_sent["article_count"] > 0

        if has_company and has_sector:
            avg_score = 0.6 * company_sent["avg_score"] + 0.4 * sector_sent["avg_score"]
            pos_ratio = 0.6 * company_sent["positive_ratio"] + 0.4 * sector_sent["positive_ratio"]
            neg_ratio = 0.6 * company_sent["negative_ratio"] + 0.4 * sector_sent["negative_ratio"]
            neu_ratio = 0.6 * company_sent["neutral_ratio"] + 0.4 * sector_sent["neutral_ratio"]
            impact_score = 0.6 * company_sent["news_impact_score"] + 0.4 * sector_sent["news_impact_score"]
        elif has_company:
            avg_score = company_sent["avg_score"]
            pos_ratio = company_sent["positive_ratio"]
            neg_ratio = company_sent["negative_ratio"]
            neu_ratio = company_sent["neutral_ratio"]
            impact_score = company_sent["news_impact_score"]
        elif has_sector:
            avg_score = sector_sent["avg_score"]
            pos_ratio = sector_sent["positive_ratio"]
            neg_ratio = sector_sent["negative_ratio"]
            neu_ratio = sector_sent["neutral_ratio"]
            impact_score = sector_sent["news_impact_score"]
        else:
            avg_score = pos_ratio = neg_ratio = impact_score = 0.0
            neu_ratio = 1.0

        sentiment = {
            "avg_score": round(avg_score, 4),
            "positive_ratio": round(pos_ratio, 4),
            "negative_ratio": round(neg_ratio, 4),
            "neutral_ratio": round(neu_ratio, 4),
            "news_impact_score": round(impact_score, 4),
            "article_count": company_sent["article_count"] + sector_sent["article_count"],
            "company": company_sent,
            "sector": sector_sent,
            "summary": f"Company News: {company_sent['avg_score']:+.2f} ({company_sent['article_count']} articles) | "
                       f"Sector News: {sector_sent['avg_score']:+.2f} ({sector_sent['article_count']} articles)"
        }

        ml = train_and_predict(df, sentiment)
        risk = assess_risk(df)
        rec = generate_recommendation(ml, sentiment, risk, df)
        forecast = train_and_forecast(df, sentiment)

        # Chart data
        period_days = {"6mo": 126, "1y": 252, "2y": 504}
        days_limit = period_days.get(period, 252)
        chart_df = df.tail(days_limit).copy()
        chart_df.index = chart_df.index.strftime("%Y-%m-%d")
        def safe(x):
            if x is None: return None
            try:
                v = float(x)
            except (TypeError, ValueError):
                return None
            if math.isnan(v) or math.isinf(v):
                return None
            return round(v, 4)
        chart_data = {
            "dates": chart_df.index.tolist(),
            "close": [safe(v) for v in chart_df["Close"]],
            "open": [safe(v) for v in chart_df["Open"]],
            "high": [safe(v) for v in chart_df["High"]],
            "low": [safe(v) for v in chart_df["Low"]],
            "volume": [safe(v) for v in chart_df["Volume"]],
            "sma20": [safe(v) for v in chart_df["SMA_20"]],
            "sma50": [safe(v) for v in chart_df["SMA_50"]],
            "bb_upper": [safe(v) for v in chart_df["BB_Upper"]],
            "bb_lower": [safe(v) for v in chart_df["BB_Lower"]],
            "rsi": [safe(v) for v in chart_df["RSI"]],
            "macd": [safe(v) for v in chart_df["MACD"]],
            "macd_signal": [safe(v) for v in chart_df["MACD_Signal"]],
            "macd_hist": [safe(v) for v in chart_df["MACD_Hist"]],
        }

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        _close = safe(latest["Close"]) or 0
        _prev = safe(prev["Close"]) or _close
        change = round(_close - _prev, 4)
        change_pct = round((change / _prev * 100) if _prev else 0, 4)
        snapshot = {
            "price": _close,
            "change": change,
            "change_pct": change_pct,
            "volume": int(latest["Volume"]) if not math.isnan(float(latest["Volume"])) else 0,
            "rsi": safe(latest["RSI"]),
            "macd": safe(latest["MACD"]),
            "macd_signal": safe(latest["MACD_Signal"]),
            "sma20": safe(latest["SMA_20"]),
            "sma50": safe(latest["SMA_50"]),
            "bb_upper": safe(latest["BB_Upper"]),
            "bb_lower": safe(latest["BB_Lower"]),
            "volatility": safe(latest["Volatility_20"]),
        }

        return jsonify({
            "ticker": ticker,
            "name": company_name,
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "sector": sector,
            "industry": industry,
            "market_cap": info.get("marketCap"),
            "snapshot": snapshot,
            "chart": chart_data,
            "sentiment": sentiment,
            "ml": ml,
            "risk": risk,
            "recommendation": rec,
            "forecast": forecast,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Stock Analysis Backend — Live News Only (CSV loaded for offline training)")
    print("  Running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000, host="0.0.0.0")