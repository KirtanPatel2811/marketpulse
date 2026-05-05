"""
news_fetcher.py
Fetches financial news articles for any ticker using NewsAPI.
Handles rate limits, deduplication, and relevance filtering.
"""

import requests
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import NEWSAPI_KEY, DEFAULT_TICKER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# Financial keywords — article must contain at least one of these to be kept
# WHY: NewsAPI free tier returns loosely matched results. This filter removes
# food blogs, sports articles, and other noise that happens to mention "Apple".
FINANCE_KEYWORDS = [
    'stock', 'shares', 'earnings', 'revenue', 'profit', 'market', 'invest',
    'analyst', 'forecast', 'quarterly', 'nasdaq', 'nyse', 'trading', 'price',
    'valuation', 'dividend', 'fiscal', 'guidance', 'outlook', 'sec', 'ipo',
    'acquisition', 'merger', 'buyback', 'shareholder', 'portfolio', 'equity'
]


def is_financial_article(article: Dict) -> bool:
    """
    Returns True only if the article is actually about finance/stocks.
    Checks title + description for at least one financial keyword.
    WHY: Free NewsAPI tier returns noisy results. A food blog titled
    'Refried or Die' should not go into our sentiment model.
    """
    text = (
        (article.get('title') or '') + ' ' +
        (article.get('description') or '') + ' ' +
        (article.get('content') or '')
    ).lower()

    return any(kw in text for kw in FINANCE_KEYWORDS)


def fetch_news(
    ticker: str = DEFAULT_TICKER,
    company_name: str = None,
    days_back: int = 7,
    max_articles: int = 20
) -> List[Dict]:
    """
    Fetch recent financial news articles for a stock ticker.

    Args:
        ticker:       Stock symbol e.g. 'AAPL'
        company_name: Full company name e.g. 'Apple'
        days_back:    How many days back to search (max 30 on free tier)
        max_articles: Max articles to return

    Returns:
        List of dicts with keys: ticker, title, description, content,
        url, published_at, source, fetched_at
    """
    if not NEWSAPI_KEY:
        logger.error("NEWSAPI_KEY not found in .env file. Please add it.")
        return []

    # Tighter query: require both company name and a finance term
    # WHY: "Apple stock" is far more precise than "AAPL" OR "Apple"
    if company_name:
        query = f'"{company_name}" AND (stock OR earnings OR shares OR market)'
    else:
        query = f'"{ticker}" AND (stock OR earnings OR shares OR market)'

    from_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    to_date = datetime.today().strftime('%Y-%m-%d')

    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': min(max_articles * 2, 100),  # fetch extra, then filter
        'apiKey': NEWSAPI_KEY
    }

    logger.info(f"Fetching news: query='{query}' from {from_date} to {to_date}")

    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)

        if response.status_code == 429:
            logger.warning("NewsAPI rate limit hit (100 req/day). Using cached data.")
            return []

        if response.status_code == 401:
            logger.error("NewsAPI key is invalid. Check your .env file.")
            return []

        response.raise_for_status()
        data = response.json()

        if data.get('status') != 'ok':
            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []

        articles = data.get('articles', [])
        logger.info(f"Raw articles from API: {len(articles)}")

        cleaned = []
        seen_titles = set()

        for article in articles:
            title = (article.get('title') or '').strip()

            if not title or title == '[Removed]':
                continue
            if title in seen_titles:
                continue

            # Build candidate dict first so we can run the relevance filter
            candidate = {
                'ticker': ticker.upper(),
                'title': title,
                'description': article.get('description', '') or '',
                'content': article.get('content', '') or '',
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'fetched_at': datetime.now().isoformat()
            }

            # Drop articles with no financial relevance
            if not is_financial_article(candidate):
                logger.debug(f"Filtered out non-financial: {title[:60]}")
                continue

            seen_titles.add(title)
            cleaned.append(candidate)

            if len(cleaned) >= max_articles:
                break

        logger.info(f"Returning {len(cleaned)} relevant articles after filtering")
        return cleaned

    except requests.exceptions.Timeout:
        logger.error("NewsAPI request timed out.")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("No internet connection or NewsAPI unreachable.")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching news: {e}")
        return []


TICKER_COMPANY_MAP = {
    'AAPL':         'Apple',
    'GOOGL':        'Alphabet Google',
    'MSFT':         'Microsoft',
    'AMZN':         'Amazon',
    'TSLA':         'Tesla',
    'NVDA':         'Nvidia',
    'META':         'Meta',
    'RELIANCE.NS':  'Reliance Industries',
    'TCS.NS':       'Tata Consultancy',
    'INFY.NS':      'Infosys',
}


def fetch_news_for_ticker(ticker: str, days_back: int = 7) -> List[Dict]:
    """
    Main function called by the rest of the pipeline.
    Auto-resolves company name and applies relevance filtering.
    """
    company_name = TICKER_COMPANY_MAP.get(ticker.upper())
    return fetch_news(ticker, company_name=company_name, days_back=days_back)


if __name__ == '__main__':
    print("=" * 50)
    print("Testing news_fetcher.py with AAPL")
    print("=" * 50)

    articles = fetch_news_for_ticker('AAPL', days_back=5)

    if articles:
        print(f"\nTotal relevant articles: {len(articles)}")
        print(f"\nFirst article:")
        print(f"  Title:  {articles[0]['title']}")
        print(f"  Source: {articles[0]['source']}")
        print(f"  Date:   {articles[0]['published_at']}")
        print(f"\nAll titles:")
        for i, a in enumerate(articles, 1):
            print(f"  {i:2}. [{a['source']}] {a['title'][:75]}")
    else:
        print("No articles returned. Check your NEWSAPI_KEY in .env")