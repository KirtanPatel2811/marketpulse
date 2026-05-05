"""
sentiment_cache.py
Connects FinBERT scoring to the SQLite cache.

WHY this exists as a separate module:
- finbert_model.py only knows how to score text — it has no DB knowledge
- data_store.py only knows how to read/write rows — no ML knowledge
- This module is the bridge: load unscored articles → score → save back
- Running it repeatedly is safe: it only processes articles with sentiment=NULL
"""

import logging
import sqlite3
import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import CACHE_DB_PATH
from src.sentiment.finbert_model import score_articles
from src.data.data_store import load_news_articles, get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def score_and_save(ticker: str, days: int = 7) -> int:
    """
    Load all unscored articles for a ticker, run FinBERT, save scores back.

    Args:
        ticker: Stock symbol e.g. 'AAPL'
        days:   How many days back to look for unscored articles

    Returns:
        Number of articles newly scored
    """
    # Load only articles not yet scored (sentiment IS NULL in DB)
    unscored = load_news_articles(ticker, days=days, unscored_only=True)

    if not unscored:
        logger.info(f"No unscored articles found for {ticker} — cache is up to date.")
        return 0

    logger.info(f"Found {len(unscored)} unscored articles for {ticker}. Running FinBERT...")

    # Score them
    scored = score_articles(unscored)

    # Write scores back to SQLite
    conn = get_connection()
    cursor = conn.cursor()
    saved = 0

    for article in scored:
        try:
            cursor.execute("""
                UPDATE news_articles
                SET sentiment       = ?,
                    sentiment_score = ?
                WHERE ticker = ? AND url = ?
            """, (
                article.get('sentiment', 'neutral'),
                article.get('sentiment_score', 0.0),
                ticker.upper(),
                article.get('url', '')
            ))
            saved += cursor.rowcount
        except Exception as e:
            logger.error(f"Error saving sentiment for '{article.get('title','')}': {e}")

    conn.commit()
    conn.close()

    logger.info(f"Saved sentiment scores for {saved} articles")
    return saved


def get_scored_articles(ticker: str, days: int = 7) -> List[Dict]:
    """
    Return articles that already have sentiment scores.
    Used by the aggregator and dashboard.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, title, description, url, published_at, source,
               sentiment, sentiment_score
        FROM news_articles
        WHERE ticker = ?
        AND sentiment IS NOT NULL
        AND published_at >= date('now', ?)
        ORDER BY published_at DESC
    """, (ticker.upper(), f'-{days} days'))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def run_full_sentiment_pipeline(ticker: str, days: int = 7) -> dict:
    """
    End-to-end: score any unscored articles then return daily sentiment.
    This is the single function the scheduler and dashboard call.

    Returns dict with:
        scored_count  — how many articles were newly scored
        daily         — DataFrame of daily sentiment scores
        articles      — list of all scored articles
    """
    from src.data.news_fetcher import fetch_news_for_ticker
    from src.data.data_store import save_news_articles
    from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment

    # Step 1: fetch fresh news and cache it
    logger.info(f"=== Sentiment pipeline for {ticker} ===")
    fresh_articles = fetch_news_for_ticker(ticker, days_back=days)
    if fresh_articles:
        saved = save_news_articles(fresh_articles)
        logger.info(f"Cached {saved} new articles")

    # Step 2: score any unscored articles
    scored_count = score_and_save(ticker, days=days)

    # Step 3: load all scored articles and aggregate
    scored_articles = get_scored_articles(ticker, days=days)
    daily_df = aggregate_daily_sentiment(scored_articles)

    return {
        'scored_count': scored_count,
        'daily': daily_df,
        'articles': scored_articles
    }


if __name__ == '__main__':
    print("=" * 55)
    print("Testing full sentiment pipeline for AAPL")
    print("=" * 55)

    result = run_full_sentiment_pipeline('AAPL', days=5)

    print(f"\nNewly scored articles: {result['scored_count']}")

    print(f"\nAll scored articles ({len(result['articles'])}):")
    for a in result['articles']:
        score = a.get('sentiment_score', 0)
        label = a.get('sentiment', 'neutral')
        emoji = '🟢' if label == 'positive' else '🔴' if label == 'negative' else '⚪'
        print(f"  {emoji} [{score:+.3f}] {a['title'][:65]}")

    print(f"\nDaily sentiment timeline:")
    if not result['daily'].empty:
        print(result['daily'][['Date','sentiment_score','sentiment_label','article_count']].to_string(index=False))
    else:
        print("  No daily data yet.")