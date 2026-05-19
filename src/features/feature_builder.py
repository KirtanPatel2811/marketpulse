"""
feature_builder.py
Merges technical indicators + sentiment scores into the final feature matrix
that gets fed into the LSTM.

This is the last step before model training.
Input:  raw price data + raw scored articles
Output: one clean DataFrame per ticker, one row per trading day,
        all features normalised, no NaN rows
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.technical_indicators import compute_all_indicators, get_feature_columns
from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment, align_sentiment_with_prices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling sentiment features derived from the daily sentiment score.

    WHY rolling features:
    - sentiment_score alone is noisy (one bad article spikes it)
    - 3-day rolling mean smooths out single-article noise
    - sentiment_momentum captures whether sentiment is improving/worsening
    - sentiment_volatility captures uncertainty (high spread = uncertain market)

    These rolling features let the LSTM learn from sentiment trends,
    not just today's score.
    """
    df = df.copy()

    if 'sentiment_score' not in df.columns:
        logger.warning("sentiment_score column missing — filling with 0")
        df['sentiment_score'] = 0.0

    df['sentiment_ma3']     = df['sentiment_score'].rolling(3).mean().round(4)
    df['sentiment_ma7']     = df['sentiment_score'].rolling(7).mean().round(4)
    df['sentiment_momentum']= df['sentiment_score'].diff(3).round(4)
    df['sentiment_vol']     = df['sentiment_score'].rolling(7).std().round(4)

    return df


def normalise_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Normalise all feature columns to [-1, 1] range using min-max scaling.

    WHY normalise:
    - LSTM is sensitive to input scale. RSI is 0-100, Daily_return is
      0.001-0.05, ATR is 0.001-0.02. Without normalisation, RSI would
      dominate just because its values are larger.
    - We save the scaler params (min/max per column) so we can
      inverse-transform predictions back to real prices.

    Returns df with normalised columns + a dict of scaler params.
    """
    scaler_params = {}
    df = df.copy()

    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Feature column '{col}' not found — skipping")
            continue

        col_min = df[col].min()
        col_max = df[col].max()
        col_range = col_max - col_min

        if col_range == 0:
            df[f'{col}_norm'] = 0.0
        else:
            df[f'{col}_norm'] = ((df[col] - col_min) / col_range * 2 - 1).round(6)

        scaler_params[col] = {'min': col_min, 'max': col_max}

    return df, scaler_params


def build_feature_matrix(
    ticker: str,
    price_df: pd.DataFrame = None,
    scored_articles: list = None,
    normalise: bool = False
) -> pd.DataFrame:
    """
    Main function — builds the complete feature matrix for a ticker.

    Args:
        ticker:          Stock symbol
        price_df:        From price_fetcher (fetched fresh if None)
        scored_articles: From sentiment_cache (fetched fresh if None)
        normalise:       Whether to normalise features (True for model training)

    Returns:
        Clean DataFrame ready for LSTM training:
        - One row per trading day
        - All technical + sentiment features
        - No NaN rows
        - Sorted by date ascending
    """
    # Step 1: get price data
    if price_df is None:
        from src.data.price_fetcher import fetch_price_data
        logger.info(f"Fetching price data for {ticker}...")
        price_df = fetch_price_data(ticker, period_years=2)

    if price_df.empty:
        logger.error(f"No price data available for {ticker}")
        return pd.DataFrame()

    # Step 2: compute technical indicators
    df = compute_all_indicators(price_df)

    # Step 3: get sentiment data
    if scored_articles is None:
        from src.sentiment.sentiment_cache import get_scored_articles
        logger.info(f"Loading cached sentiment for {ticker}...")
        scored_articles = get_scored_articles(ticker, days=30)

    # Step 4: aggregate sentiment to daily scores
    if scored_articles:
        sentiment_df = aggregate_daily_sentiment(scored_articles)
        df = align_sentiment_with_prices(sentiment_df, df)
    else:
        logger.warning("No scored articles — using neutral sentiment (0.0) for all days")
        df['sentiment_score'] = 0.0

    # Step 5: add rolling sentiment features
    df = add_sentiment_features(df)

    # Step 6: drop NaN rows (indicator warmup period — first ~50 rows)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    logger.info(f"Dropped {before - after} NaN rows (indicator warmup). "
                f"Clean rows: {after}")

    # Step 7: optionally normalise
    if normalise:
        feature_cols = get_feature_columns() + [
            'sentiment_ma3', 'sentiment_ma7',
            'sentiment_momentum', 'sentiment_vol'
        ]
        df, scaler_params = normalise_features(df, feature_cols)
        logger.info(f"Features normalised. Scaler params saved for {len(scaler_params)} columns.")
        df.attrs['scaler_params'] = scaler_params

    logger.info(f"Feature matrix ready: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


if __name__ == '__main__':
    print("=" * 55)
    print("Testing feature_builder.py for AAPL")
    print("=" * 55)

    df = build_feature_matrix('AAPL')

    feature_cols = get_feature_columns() + [
        'sentiment_ma3', 'sentiment_ma7', 'sentiment_momentum', 'sentiment_vol'
    ]

    print(f"\nFull matrix shape: {df.shape}")
    print(f"\nDate range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"  {col:25s}  min={vals.min():+.4f}  max={vals.max():+.4f}  "
                  f"mean={vals.mean():+.4f}")

    print(f"\nLast 5 rows (key features):")
    show = ['Date', 'Close', 'RSI_14', 'MACD_histogram',
            'BB_position', 'sentiment_score', 'sentiment_ma3']
    print(df[show].tail(5).to_string(index=False))

    print(f"\nDays with actual sentiment (non-zero): "
          f"{(df['sentiment_score'] != 0).sum()} / {len(df)}")