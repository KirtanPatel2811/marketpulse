"""
price_fetcher.py
Fetches historical OHLCV price data for any ticker using yfinance.
Handles errors, validates data, and returns a clean DataFrame.

WHY yfinance: No API key needed, supports both US (AAPL) and Indian
(RELIANCE.NS) tickers, returns clean OHLCV data going back years.
"""

import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DEFAULT_TICKER

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_price_data(
    ticker: str = DEFAULT_TICKER,
    period_years: int = 3,
    end_date: datetime = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker.

    Args:
        ticker:       Stock ticker symbol e.g. 'AAPL' or 'RELIANCE.NS'
        period_years: How many years of history to fetch (default 3)
        end_date:     End date for data (default today)

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume, Ticker
        Returns empty DataFrame on failure.

    WHY 3 years default: Enough data to train LSTM (needs ~500+ rows),
    captures multiple market cycles, but not so large it slows things down.
    """
    if end_date is None:
        end_date = datetime.today()

    start_date = end_date - timedelta(days=period_years * 365)

    logger.info(f"Fetching price data for {ticker} from {start_date.date()} to {end_date.date()}")

    try:
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=True   # adjusts for splits and dividends automatically
        )

        if df.empty:
            logger.error(f"No data returned for ticker: {ticker}. Check the symbol is valid.")
            return pd.DataFrame()

        # Clean up the DataFrame
        df = df.reset_index()
        df = df.rename(columns={'index': 'Date'})

        # Keep only the columns we need
        cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in cols_to_keep if c in df.columns]]

        # Ensure Date column is date only (no time component)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()

        # Add ticker column so we know which stock this is when caching
        df['Ticker'] = ticker.upper()

        # Drop rows where Close is NaN (can happen on holidays in some markets)
        df = df.dropna(subset=['Close'])
        df = df.sort_values('Date').reset_index(drop=True)

        logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        logger.info(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        logger.info(f"Latest close price: {df['Close'].iloc[-1]:.2f}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        return pd.DataFrame()


def get_latest_price(ticker: str = DEFAULT_TICKER) -> dict:
    """
    Get just the latest price info for a ticker.
    Used by the dashboard for the current price display.

    Returns dict with: ticker, price, change, change_pct, volume, date
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.fast_info

        return {
            'ticker': ticker.upper(),
            'price': round(info.last_price, 2),
            'previous_close': round(info.previous_close, 2),
            'change': round(info.last_price - info.previous_close, 2),
            'change_pct': round(((info.last_price - info.previous_close) / info.previous_close) * 100, 2),
            'volume': int(info.last_volume),
            'date': datetime.today().strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.error(f"Failed to get latest price for {ticker}: {e}")
        return {}


def validate_ticker(ticker: str) -> bool:
    """
    Check whether a ticker symbol is valid before running the full pipeline.
    WHY: Prevents silent failures downstream when user types a wrong symbol.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.fast_info
        return info.last_price is not None and info.last_price > 0
    except Exception:
        return False


if __name__ == '__main__':
    print("=" * 50)
    print("Testing price_fetcher.py with AAPL")
    print("=" * 50)

    # Test 1: validate ticker
    print(f"\nIs AAPL valid? {validate_ticker('AAPL')}")
    print(f"Is FAKEXYZ valid? {validate_ticker('FAKEXYZ')}")

    # Test 2: fetch historical data
    df = fetch_price_data('AAPL', period_years=1)
    if not df.empty:
        print(f"\nShape: {df.shape}")
        print(f"\nFirst 3 rows:\n{df.head(3)}")
        print(f"\nLast 3 rows:\n{df.tail(3)}")
        print(f"\nData types:\n{df.dtypes}")

    # Test 3: latest price
    latest = get_latest_price('AAPL')
    if latest:
        print(f"\nLatest price info:")
        for k, v in latest.items():
            print(f"  {k}: {v}")
