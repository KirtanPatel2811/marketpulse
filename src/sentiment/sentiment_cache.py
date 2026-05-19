"""
sentiment_cache.py
Connects FinBERT scoring to the SQLite cache.
"""

import logging
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
    unscored = load_news_articles(ticker, days=days, unscored_only=True)

    if not unscored:
        logger.info(f"No unscored articles for {ticker} — cache is up to date.")
        return 0

    logger.info(f"Found {len(unscored)} unscored articles. Running FinBERT...")
    scored = score_articles(unscored)

    conn = get_connection()
    cursor = conn.cursor()
    saved = 0

    for article in scored:
        try:
            cursor.execute("""
                UPDATE news_articles
                SET sentiment            = ?,
                    sentiment_score      = ?,
                    sentiment_confidence = ?,
                    sentiment_positive   = ?,
                    sentiment_negative   = ?,
                    sentiment_neutral    = ?
                WHERE ticker = ? AND url = ?
            """, (
                article.get('sentiment', 'neutral'),
                article.get('sentiment_score', 0.0),
                article.get('sentiment_confidence', 0.0),
                article.get('sentiment_positive', 0.0),
                article.get('sentiment_negative', 0.0),
                article.get('sentiment_neutral', 1.0),
                ticker.upper(),
                article.get('url', '')
            ))
            saved += cursor.rowcount
        except Exception as e:
            logger.error(f"Error saving sentiment for '{article.get('title','')}': {e}")

    conn.commit()
    conn.close()
    logger.info(f"Saved full sentiment scores for {saved} articles")
    return saved


def get_scored_articles(ticker: str, days: int = 7) -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ticker, title, description, url, published_at, source,
               sentiment, sentiment_score, sentiment_confidence,
               sentiment_positive, sentiment_negative, sentiment_neutral
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
    End-to-end: fetch fresh news → score unscored → return daily sentiment.
    This is the single function the scheduler and dashboard call.
    """
    from src.data.news_fetcher import fetch_news_for_ticker
    from src.data.data_store import save_news_articles
    from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment

    logger.info(f"=== Sentiment pipeline for {ticker} ===")

    # Step 1: fetch and cache fresh news
    fresh = fetch_news_for_ticker(ticker, days_back=days)
    if fresh:
        saved = save_news_articles(fresh)
        logger.info(f"Cached {saved} new articles")

    # Step 2: score unscored articles
    scored_count = score_and_save(ticker, days=days)

    # Step 3: aggregate into daily scores
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

    # Re-score existing articles so confidence column gets populated
    ticker = 'AAPL'
    conn = __import__('sqlite3').connect(CACHE_DB_PATH)
    conn.execute("UPDATE news_articles SET sentiment=NULL, sentiment_score=NULL, sentiment_confidence=NULL WHERE ticker=?", (ticker,))
    conn.commit()
    conn.close()
    print("Reset scores so we can re-score with full columns...\n")

    result = run_full_sentiment_pipeline(ticker, days=7)

    print(f"\nNewly scored: {result['scored_count']} articles")
    print(f"\nAll scored articles ({len(result['articles'])}):")
    for a in result['articles']:
        score = a.get('sentiment_score', 0) or 0
        label = a.get('sentiment', 'neutral')
        conf  = a.get('sentiment_confidence', 0) or 0
        emoji = '🟢' if label == 'positive' else '🔴' if label == 'negative' else '⚪'
        print(f"  {emoji} [{score:+.3f}] conf={conf:.2f} | {a['title'][:60]}")

    print(f"\nDaily sentiment timeline:")
    if not result['daily'].empty:
        print(result['daily'][['Date','sentiment_score','sentiment_label','article_count']].to_string(index=False))
    else:
        print("  No daily data (articles may all be from today — need spread across days)")