"""
scheduler.py
APScheduler-based auto-refresh pipeline for MarketPulse.

Runs every 30 minutes:
  1. Fetch fresh news for all tracked tickers
  2. Score unscored articles with FinBERT
  3. Fetch latest price data

WHY 30 minutes: NewsAPI free tier is 100 req/day.
With 3 tickers refreshed every 30 min = 6 req/hour = 144 req/day.
We cache aggressively so most refreshes hit 0 new articles
and cost 0 API calls if nothing new has been published.
"""

import logging
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from apscheduler.schedulers.blocking import BlockingScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TRACKED_TICKERS = ['AAPL', 'MSFT', 'NVDA']


def refresh_ticker(ticker: str):
    """Full refresh cycle for one ticker: prices + news + FinBERT scoring."""
    logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing {ticker}...")
    try:
        from src.data.price_fetcher import fetch_price_data
        from src.data.data_store import save_price_data
        df = fetch_price_data(ticker, period_years=2)
        if not df.empty:
            saved = save_price_data(df)
            logger.info(f"  {ticker} prices: {saved} rows upserted")

        from src.sentiment.sentiment_cache import run_full_sentiment_pipeline
        result = run_full_sentiment_pipeline(ticker, days=7)
        logger.info(f"  {ticker} sentiment: {result['scored_count']} articles newly scored")

    except Exception as e:
        logger.error(f"  {ticker} refresh failed: {e}")


def refresh_all():
    """Refresh all tracked tickers. Called by scheduler every 30 minutes."""
    logger.info(f"=== Scheduled refresh ({len(TRACKED_TICKERS)} tickers) ===")
    for ticker in TRACKED_TICKERS:
        refresh_ticker(ticker)
    logger.info("=== Refresh complete ===\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MarketPulse scheduler')
    parser.add_argument('--once', action='store_true',
                        help='Run one refresh immediately and exit')
    args = parser.parse_args()

    if args.once:
        logger.info("Running one-time refresh...")
        refresh_all()
        logger.info("Done.")
    else:
        logger.info("MarketPulse scheduler starting...")
        logger.info(f"Tracking: {TRACKED_TICKERS}")
        logger.info("Refresh interval: every 30 minutes")
        logger.info("Running initial refresh now...\n")
        refresh_all()

        scheduler = BlockingScheduler(timezone='UTC')
        scheduler.add_job(
            func=refresh_all,
            trigger='interval',
            minutes=30,
            id='marketpulse_refresh',
            name='MarketPulse 30-min refresh',
            replace_existing=True
        )
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped.")