"""
sentiment_aggregator.py
Aggregates per-article FinBERT scores into a daily sentiment score timeline.

WHY aggregation matters:
- The LSTM needs one sentiment value per day, not per article
- Some days have 10 articles, some have 2 — raw counts are inconsistent
- We use confidence-weighted mean so a high-confidence negative article
  pulls the score down more than an uncertain neutral one

Daily score range: -1.0 (very negative) to +1.0 (very positive)
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def aggregate_daily_sentiment(scored_articles: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of scored articles into a daily sentiment DataFrame.

    Args:
        scored_articles: Articles with sentiment_score and published_at fields

    Returns:
        DataFrame with columns:
          Date              — trading date
          sentiment_score   — confidence-weighted mean of article scores
          sentiment_pos     — fraction of positive articles that day
          sentiment_neg     — fraction of negative articles that day
          sentiment_neu     — fraction of neutral articles that day
          article_count     — number of articles that day
          sentiment_label   — 'positive', 'negative', or 'neutral'

    WHY confidence-weighted mean:
        A score of +0.95 (very confident positive) should outweigh
        a score of +0.51 (barely positive). Simple mean treats them equally.
        Weighted mean: sum(score * confidence) / sum(confidence)
    """
    if not scored_articles:
        logger.warning("No scored articles to aggregate.")
        return pd.DataFrame()

    df = pd.DataFrame(scored_articles)

    # Parse and normalise date column
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
    df = df.dropna(subset=['published_at'])
    df['Date'] = df['published_at'].dt.tz_localize(None).dt.normalize()

    required = ['sentiment_score', 'sentiment_confidence']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Missing columns in scored articles: {missing}")
        return pd.DataFrame()

    def weighted_mean(group):
        scores = group['sentiment_score'].values
        weights = group['sentiment_confidence'].values

        # Avoid division by zero
        if weights.sum() == 0:
            return pd.Series({
                'sentiment_score': 0.0,
                'sentiment_pos': 0.0,
                'sentiment_neg': 0.0,
                'sentiment_neu': 1.0,
                'article_count': len(group)
            })

        weighted_score = np.average(scores, weights=weights)

        labels = group['sentiment'].values if 'sentiment' in group.columns else []
        n = len(labels)
        pos = np.sum(labels == 'positive') / n if n > 0 else 0
        neg = np.sum(labels == 'negative') / n if n > 0 else 0
        neu = np.sum(labels == 'neutral') / n if n > 0 else 0

        return pd.Series({
            'sentiment_score': round(float(weighted_score), 4),
            'sentiment_pos': round(float(pos), 4),
            'sentiment_neg': round(float(neg), 4),
            'sentiment_neu': round(float(neu), 4),
            'article_count': n
        })

    daily = df.groupby('Date').apply(weighted_mean).reset_index()

    # Add human-readable label
    def label(score):
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'

    daily['sentiment_label'] = daily['sentiment_score'].apply(label)
    daily = daily.sort_values('Date').reset_index(drop=True)

    logger.info(f"Aggregated {len(df)} articles into {len(daily)} daily sentiment rows")
    return daily


def align_sentiment_with_prices(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge daily sentiment scores onto the price DataFrame.
    Trading days with no news get sentiment_score = 0 (neutral).

    WHY left join on prices: We always have a price row for every trading day.
    News coverage is sparse — some days have no articles. We fill those with
    neutral (0.0) rather than dropping the row, so the LSTM sees a complete
    sequence with no gaps.

    Args:
        sentiment_df: Output of aggregate_daily_sentiment()
        price_df:     Output of price_fetcher.fetch_price_data()

    Returns:
        price_df with sentiment columns added, filled forward for missing days
    """
    if sentiment_df.empty:
        logger.warning("Empty sentiment_df — adding zero sentiment columns to prices.")
        price_df = price_df.copy()
        price_df['sentiment_score'] = 0.0
        price_df['sentiment_pos'] = 0.0
        price_df['sentiment_neg'] = 0.0
        price_df['sentiment_neu'] = 1.0
        price_df['article_count'] = 0
        return price_df

    price_df = price_df.copy()
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

    merged = pd.merge(price_df, sentiment_df, on='Date', how='left')

    # Fill missing sentiment days with neutral
    sentiment_cols = ['sentiment_score', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu']
    merged[sentiment_cols] = merged[sentiment_cols].fillna(0.0)
    merged['article_count'] = merged['article_count'].fillna(0).astype(int)
    merged['sentiment_label'] = merged['sentiment_label'].fillna('neutral')

    logger.info(f"Merged sentiment onto {len(merged)} price rows. "
                f"Days with news: {(merged['article_count'] > 0).sum()}")
    return merged


if __name__ == '__main__':
    print("=" * 55)
    print("Testing sentiment_aggregator.py")
    print("=" * 55)

    # Simulate scored articles
    from datetime import datetime, timedelta
    import random

    random.seed(42)
    fake_articles = []
    base = datetime(2026, 4, 28)

    sample_data = [
        ("Apple reports record Q2 earnings", "positive", 0.95, 0.92),
        ("Apple stock dips on supply concerns", "negative", -0.72, 0.80),
        ("Apple says iPhone demand remains strong", "positive", 0.88, 0.89),
        ("Apple neutral product announcement", "neutral", 0.02, 0.76),
        ("Apple revenue beats analyst estimates", "positive", 0.91, 0.93),
        ("Apple guidance disappoints investors", "negative", -0.65, 0.78),
    ]

    for i, (title, label, score, conf) in enumerate(sample_data):
        day_offset = i % 3
        fake_articles.append({
            'title': title,
            'published_at': (base + timedelta(days=day_offset)).isoformat() + 'Z',
            'sentiment': label,
            'sentiment_score': score,
            'sentiment_confidence': conf
        })

    daily = aggregate_daily_sentiment(fake_articles)
    print(f"\nDaily sentiment (from {len(fake_articles)} articles):")
    print(daily[['Date', 'sentiment_score', 'sentiment_label', 'article_count']].to_string(index=False))