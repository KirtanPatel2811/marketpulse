"""
main.py
FastAPI REST API for MarketPulse.

Endpoints:
  GET /health                          — API + model status
  GET /sentiment/{ticker}              — current sentiment + recent articles
  GET /sentiment/{ticker}/history      — historical daily sentiment
  GET /forecast/{ticker}               — 5-day price forecast
  GET /technicals/{ticker}             — current technical indicator values
  GET /news/{ticker}                   — latest news with sentiment scores
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarketPulse API",
    description="Real-time financial news sentiment + stock price forecasting",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """API status and model availability check."""
    import torch
    from config import MODELS_DIR

    models_available = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pt'):
            models_available.append(f.replace('lstm_','').replace('.pt','').upper())

    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_trained": models_available,
        "version": "1.0.0"
    }


@app.get("/sentiment/{ticker}")
def get_sentiment(ticker: str, days: int = Query(default=7, ge=1, le=30)):
    """
    Current sentiment score and recent scored articles for a ticker.
    Runs the full pipeline: fetch news → score with FinBERT → aggregate.
    """
    ticker = ticker.upper()
    try:
        from src.sentiment.sentiment_cache import run_full_sentiment_pipeline
        result = run_full_sentiment_pipeline(ticker, days=days)

        articles = result.get('articles', [])
        daily    = result.get('daily')

        avg_score = 0.0
        label     = 'neutral'
        if articles:
            import numpy as np
            scores    = [a.get('sentiment_score', 0) or 0 for a in articles]
            avg_score = round(float(np.mean(scores)), 4)
            label     = 'positive' if avg_score > 0.1 else 'negative' if avg_score < -0.1 else 'neutral'

        return {
            "ticker":        ticker,
            "sentiment_score": avg_score,
            "sentiment_label": label,
            "article_count": len(articles),
            "days_analysed": days,
            "articles": [
                {
                    "title":      a.get("title", ""),
                    "source":     a.get("source", ""),
                    "published":  a.get("published_at", ""),
                    "url":        a.get("url", ""),
                    "sentiment":  a.get("sentiment", "neutral"),
                    "score":      a.get("sentiment_score", 0),
                    "confidence": a.get("sentiment_confidence", 0),
                }
                for a in articles[:10]
            ]
        }
    except Exception as e:
        logger.error(f"Sentiment error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment/{ticker}/history")
def get_sentiment_history(
    ticker: str,
    days: int = Query(default=30, ge=1, le=90)
):
    """Historical daily sentiment scores for a ticker."""
    ticker = ticker.upper()
    try:
        from src.sentiment.sentiment_cache import get_scored_articles
        from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment

        articles = get_scored_articles(ticker, days=days)
        daily_df = aggregate_daily_sentiment(articles)

        if daily_df.empty:
            return {"ticker": ticker, "days": days, "history": []}

        history = []
        for _, row in daily_df.iterrows():
            history.append({
                "date":            str(row['Date'])[:10],
                "sentiment_score": round(float(row['sentiment_score']), 4),
                "sentiment_label": row['sentiment_label'],
                "article_count":   int(row['article_count'])
            })

        return {"ticker": ticker, "days": days, "history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{ticker}")
def get_forecast(ticker: str):
    """5-day price forecast using LSTM + Prophet."""
    ticker = ticker.upper()
    try:
        from src.features.feature_builder import build_feature_matrix
        from src.models.lstm_forecaster import load_model, predict_next_days
        from src.models.baseline_prophet import train_prophet, prophet_forecast
        import pandas as pd

        df = build_feature_matrix(ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        # LSTM
        try:
            model, close_scaler, feature_cols = load_model(ticker)
            lstm_prices = predict_next_days(model, df, feature_cols, close_scaler)
        except FileNotFoundError:
            lstm_prices = [None] * 5

        # Prophet
        prophet_model = train_prophet(df)
        prophet_fc    = prophet_forecast(prophet_model)

        last_close = float(df['Close'].iloc[-1])
        last_date  = pd.to_datetime(df['Date'].iloc[-1])
        future_dates = pd.bdate_range(
            start=last_date + pd.tseries.offsets.BDay(1), periods=5
        )

        forecasts = []
        for i, (d, pp) in enumerate(zip(future_dates, prophet_fc['yhat'].values)):
            lp = lstm_prices[i] if lstm_prices[i] is not None else None
            forecasts.append({
                "date":          d.strftime('%Y-%m-%d'),
                "lstm":          lp,
                "prophet":       round(float(pp), 2),
                "prophet_lower": round(float(prophet_fc['yhat_lower'].values[i]), 2),
                "prophet_upper": round(float(prophet_fc['yhat_upper'].values[i]), 2),
                "lstm_change_pct":    round((lp - last_close) / last_close * 100, 2) if lp else None,
                "prophet_change_pct": round((float(pp) - last_close) / last_close * 100, 2),
            })

        return {
            "ticker":     ticker,
            "last_close": last_close,
            "last_date":  str(last_date.date()),
            "forecasts":  forecasts
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/technicals/{ticker}")
def get_technicals(ticker: str):
    """Current technical indicator values for a ticker."""
    ticker = ticker.upper()
    try:
        from src.data.price_fetcher import fetch_price_data
        from src.features.technical_indicators import compute_all_indicators

        df = fetch_price_data(ticker, period_years=1)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

        df = compute_all_indicators(df).dropna()
        last = df.iloc[-1]

        rsi = float(last['RSI_14'])
        macd_hist = float(last['MACD_histogram'])
        bb_pos = float(last['BB_position'])

        return {
            "ticker":        ticker,
            "date":          str(last['Date'].date()),
            "close":         round(float(last['Close']), 2),
            "RSI_14":        round(rsi, 2),
            "RSI_signal":    "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
            "MACD":          round(float(last['MACD']), 4),
            "MACD_signal":   round(float(last['MACD_signal']), 4),
            "MACD_histogram":round(macd_hist, 4),
            "MACD_trend":    "bullish" if macd_hist > 0 else "bearish",
            "BB_upper":      round(float(last['BB_upper']), 2),
            "BB_lower":      round(float(last['BB_lower']), 2),
            "BB_position":   round(bb_pos, 4),
            "BB_signal":     "near_upper" if bb_pos > 0.8 else "near_lower" if bb_pos < 0.2 else "mid",
            "EMA_20":        round(float(last['EMA_20']), 2),
            "EMA_50":        round(float(last['EMA_50']), 2),
            "ATR_normalised":round(float(last['ATR_normalised']), 6),
            "daily_return":  round(float(last['Daily_return']), 6),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/news/{ticker}")
def get_news(
    ticker: str,
    limit: int = Query(default=10, ge=1, le=50),
    days:  int = Query(default=7, ge=1, le=30)
):
    """Latest news articles with sentiment scores for a ticker."""
    ticker = ticker.upper()
    try:
        from src.sentiment.sentiment_cache import get_scored_articles
        articles = get_scored_articles(ticker, days=days)[:limit]

        return {
            "ticker":  ticker,
            "count":   len(articles),
            "articles": [
                {
                    "title":      a.get("title", ""),
                    "source":     a.get("source", ""),
                    "published":  a.get("published_at", ""),
                    "url":        a.get("url", ""),
                    "sentiment":  a.get("sentiment", "neutral"),
                    "score":      round(a.get("sentiment_score", 0) or 0, 4),
                    "confidence": round(a.get("sentiment_confidence", 0) or 0, 4),
                }
                for a in articles
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))