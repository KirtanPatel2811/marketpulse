"""
data_store.py
SQLite cache layer for price data and news articles.

WHY caching matters:
- NewsAPI free tier: only 100 requests/day. Without cache, a 30-min
  scheduler would burn through that in under an hour.
- FinBERT inference is slow. We never re-score an article we already have.
- yfinance has no hard limit but rate-limits aggressive scrapers.

Design: two tables — price_data and news_articles.
Each row is uniquely identified so re-inserting is safe (upsert pattern).
"""

import sqlite3
import pandas as pd
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import CACHE_DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite connection.
    Creates the cache directory if it doesn't exist yet.
    """
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def init_db() -> None:
    """
    Create all tables if they don't already exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.

    WHY separate tables: price data and news have very different shapes
    and query patterns. Price is queried by date range, news by ticker + date.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Price data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            date        TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL NOT NULL,
            volume      INTEGER,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)   -- prevents duplicate rows on re-fetch
        )
    """)

    # News articles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT NOT NULL,
            title         TEXT NOT NULL,
            description   TEXT,
            content       TEXT,
            url           TEXT,
            published_at  TEXT,
            source        TEXT,
            fetched_at    TEXT,
            sentiment     TEXT,      -- filled in Phase 2 by FinBERT
            sentiment_score REAL,    -- numeric score: -1.0 to +1.0
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, url)      -- deduplicate by URL
        )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialised at: {CACHE_DB_PATH}")


def save_price_data(df: pd.DataFrame) -> int:
    """
    Save price DataFrame to SQLite. Upserts — safe to call repeatedly.

    Returns: number of rows inserted/updated
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to save_price_data — skipping.")
        return 0

    conn = get_connection()
    cursor = conn.cursor()
    rows_saved = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO price_data (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume
            """, (
                row['Ticker'],
                str(row['Date'].date()) if hasattr(row['Date'], 'date') else str(row['Date']),
                float(row.get('Open', 0)),
                float(row.get('High', 0)),
                float(row.get('Low', 0)),
                float(row['Close']),
                int(row.get('Volume', 0))
            ))
            rows_saved += 1
        except Exception as e:
            logger.error(f"Error saving price row {row.get('Date')}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Saved {rows_saved} price rows for {df['Ticker'].iloc[0]}")
    return rows_saved


def load_price_data(ticker: str, days: int = 365) -> pd.DataFrame:
    """
    Load price data from cache for a ticker.

    Args:
        ticker: Stock symbol
        days:   How many recent days to load (default 365)

    Returns: DataFrame with same structure as fetch_price_data output
    """
    conn = get_connection()

    query = """
        SELECT date as Date, open as Open, high as High, low as Low,
               close as Close, volume as Volume, ticker as Ticker
        FROM price_data
        WHERE ticker = ?
        AND date >= date('now', ?)
        ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn, params=(ticker.upper(), f'-{days} days'))
    conn.close()

    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"Loaded {len(df)} cached price rows for {ticker}")

    return df


def save_news_articles(articles: List[Dict]) -> int:
    """
    Save news articles to SQLite. Upserts by URL — safe to call repeatedly.
    Articles without a URL are skipped.

    Returns: number of new articles inserted
    """
    if not articles:
        return 0

    conn = get_connection()
    cursor = conn.cursor()
    rows_saved = 0

    for article in articles:
        if not article.get('url'):
            continue
        try:
            cursor.execute("""
                INSERT INTO news_articles
                    (ticker, title, description, content, url, published_at, source, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, url) DO NOTHING
            """, (
                article.get('ticker', ''),
                article.get('title', ''),
                article.get('description', ''),
                article.get('content', ''),
                article.get('url', ''),
                article.get('published_at', ''),
                article.get('source', ''),
                article.get('fetched_at', datetime.now().isoformat())
            ))
            rows_saved += cursor.rowcount
        except Exception as e:
            logger.error(f"Error saving article '{article.get('title', '')}': {e}")

    conn.commit()
    conn.close()
    logger.info(f"Saved {rows_saved} new articles")
    return rows_saved


def load_news_articles(ticker: str, days: int = 7, unscored_only: bool = False) -> List[Dict]:
    """
    Load news articles from cache.

    Args:
        ticker:        Stock symbol
        days:          How many recent days to load
        unscored_only: If True, only return articles not yet scored by FinBERT
                       (used in Phase 2 to avoid re-running sentiment model)

    Returns: List of article dicts
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT * FROM news_articles
        WHERE ticker = ?
        AND published_at >= date('now', ?)
    """
    params = [ticker.upper(), f'-{days} days']

    if unscored_only:
        query += " AND sentiment IS NULL"

    query += " ORDER BY published_at DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    articles = [dict(row) for row in rows]
    logger.info(f"Loaded {len(articles)} articles for {ticker} (unscored_only={unscored_only})")
    return articles


def get_cache_stats() -> Dict:
    """
    Return a summary of what's currently in the cache.
    Useful for debugging and the dashboard status panel.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest FROM price_data")
    price_stats = dict(cursor.fetchone())

    cursor.execute("SELECT COUNT(*) as count FROM news_articles")
    news_count = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count FROM news_articles WHERE sentiment IS NOT NULL")
    scored_count = cursor.fetchone()['count']

    conn.close()

    return {
        'price_rows': price_stats['count'],
        'price_date_range': f"{price_stats['earliest']} to {price_stats['latest']}",
        'news_articles': news_count,
        'sentiment_scored': scored_count,
        'unscored': news_count - scored_count
    }


if __name__ == '__main__':
    print("=" * 50)
    print("Testing data_store.py")
    print("=" * 50)

    # Init DB
    init_db()

    # Test with price data
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.data.price_fetcher import fetch_price_data
    from src.data.news_fetcher import fetch_news_for_ticker

    print("\n--- Price Data ---")
    df = fetch_price_data('AAPL', period_years=1)
    saved = save_price_data(df)
    print(f"Saved {saved} price rows")

    loaded = load_price_data('AAPL', days=30)
    print(f"Loaded back {len(loaded)} rows from cache")
    print(loaded.tail(3))

    print("\n--- News Articles ---")
    articles = fetch_news_for_ticker('AAPL', days_back=3)
    saved_news = save_news_articles(articles)
    print(f"Saved {saved_news} new articles")

    loaded_news = load_news_articles('AAPL', days=3)
    print(f"Loaded {len(loaded_news)} articles from cache")
    for a in loaded_news[:3]:
        print(f"  [{a['source']}] {a['title'][:70]}")

    print("\n--- Cache Stats ---")
    stats = get_cache_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
