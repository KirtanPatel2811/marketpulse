"""
technical_indicators.py
Computes technical analysis features from OHLCV price data.

WHY these specific indicators:
- RSI:     Momentum — tells us if stock is overbought/oversold
- MACD:    Trend direction and strength
- BB:      Volatility — how wide the price range is
- EMA:     Smoothed trend (less noise than SMA)
- OBV:     Volume-price relationship — confirms price moves
- ATR:     Volatility measure — used for position sizing in real trading

These are the features a professional quant would use as baseline inputs.
Adding sentiment on top is what makes MarketPulse different.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index — momentum oscillator (0 to 100).
    > 70: overbought (potential sell signal)
    < 30: oversold  (potential buy signal)
    """
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    df[f'RSI_{period}'] = rsi.round(4)
    return df


def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.
    MACD line crossing above signal = bullish momentum
    MACD line crossing below signal = bearish momentum
    Histogram = MACD - Signal (positive = bullish, negative = bearish)
    """
    ema_fast   = df['Close'].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df['Close'].ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line

    df['MACD']           = macd_line.round(4)
    df['MACD_signal']    = signal_line.round(4)
    df['MACD_histogram'] = histogram.round(4)
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands — volatility bands around a moving average.
    BB_width: wide = high volatility, narrow = low volatility (squeeze)
    BB_position: 0 = at lower band, 1 = at upper band, 0.5 = at midline
    WHY BB_position over raw bands: it's normalised across different
    price levels so AAPL ($200) and RELIANCE.NS (₹1400) are comparable.
    """
    sma    = df['Close'].rolling(window=period).mean()
    std_   = df['Close'].rolling(window=period).std()
    upper  = sma + (std_ * std)
    lower  = sma - (std_ * std)
    width  = (upper - lower) / sma.replace(0, np.nan)

    band_range = (upper - lower).replace(0, np.nan)
    position   = (df['Close'] - lower) / band_range

    df['BB_upper']    = upper.round(4)
    df['BB_lower']    = lower.round(4)
    df['BB_width']    = width.round(4)
    df['BB_position'] = position.round(4)
    return df


def add_ema(df: pd.DataFrame, periods: list = [20, 50]) -> pd.DataFrame:
    """
    Exponential Moving Averages.
    EMA_20 > EMA_50: short-term uptrend (golden cross territory)
    EMA_20 < EMA_50: short-term downtrend (death cross territory)
    EMA_cross: positive = bullish, negative = bearish
    """
    for p in periods:
        df[f'EMA_{p}'] = df['Close'].ewm(span=p, adjust=False).mean().round(4)

    if 20 in periods and 50 in periods:
        df['EMA_cross'] = (df['EMA_20'] - df['EMA_50']).round(4)

    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    On-Balance Volume — cumulative volume indicator.
    Rising OBV with rising price = confirmed uptrend
    Rising OBV with falling price = potential reversal up
    WHY normalise: raw OBV values are huge (billions) and not
    comparable across stocks. We use 20-day rate of change instead.
    """
    direction = np.sign(df['Close'].diff()).fillna(0)
    obv       = (direction * df['Volume']).cumsum()
    df['OBV'] = obv

    # Normalise: 20-day % change in OBV
    df['OBV_change'] = df['OBV'].pct_change(periods=20).round(4)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average True Range — volatility measure.
    High ATR = large daily price swings (volatile)
    Low ATR  = small daily price swings (calm)
    We normalise by Close price so it is comparable across stocks.
    """
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr        = true_range.ewm(com=period - 1, min_periods=period).mean()

    df['ATR']            = atr.round(4)
    df['ATR_normalised'] = (atr / df['Close']).round(6)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional price-derived features that help the LSTM.
    - Daily return: % change in close price
    - Log return:   log(close_t / close_t-1) — more normally distributed
    - High-Low pct: intraday range as % of close
    - Gap:          overnight gap (open vs previous close)
    """
    df['Daily_return']  = df['Close'].pct_change().round(6)
    df['Log_return']    = np.log(df['Close'] / df['Close'].shift(1)).round(6)
    df['HL_pct']        = ((df['High'] - df['Low']) / df['Close']).round(6)
    df['Gap']           = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).round(6)
    df['Volume_change'] = df['Volume'].pct_change().round(6)
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — applies all indicators to a price DataFrame.

    Args:
        df: DataFrame from price_fetcher with columns Date, Open, High,
            Low, Close, Volume, Ticker

    Returns:
        Same DataFrame with ~20 new feature columns added.
        First 50 rows will have NaN for indicators that need warmup
        (e.g. 50-day EMA needs 50 rows of history).
        These are dropped before feeding to the LSTM.
    """
    if df.empty:
        logger.error("Empty DataFrame passed to compute_all_indicators")
        return df

    logger.info(f"Computing technical indicators for {df['Ticker'].iloc[0]} "
                f"({len(df)} rows)")

    df = df.copy()
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_ema(df, periods=[20, 50])
    df = add_obv(df)
    df = add_atr(df)
    df = add_price_features(df)

    # Count how many NaN rows we have (indicator warmup period)
    nan_rows = df[df.isnull().any(axis=1)].shape[0]
    logger.info(f"Indicators computed. NaN warmup rows: {nan_rows} "
                f"(will be dropped before model training)")

    return df


def get_feature_columns() -> list:
    """
    Returns the list of feature column names used as LSTM inputs.
    Excludes raw price columns (we predict Close, so it cannot be an input).
    Call this from feature_builder.py to keep column lists in sync.
    """
    return [
        # Price features
        'Daily_return', 'Log_return', 'HL_pct', 'Gap', 'Volume_change',
        # Momentum
        'RSI_14',
        # Trend
        'MACD', 'MACD_signal', 'MACD_histogram', 'EMA_cross',
        # Volatility
        'BB_width', 'BB_position', 'ATR_normalised',
        # Volume
        'OBV_change',
        # Sentiment (added by feature_builder.py)
        'sentiment_score',
    ]


if __name__ == '__main__':
    print("=" * 55)
    print("Testing technical_indicators.py with AAPL")
    print("=" * 55)

    import sys
    sys.path.insert(0, os.path.abspath('.'))
    from src.data.price_fetcher import fetch_price_data

    df = fetch_price_data('AAPL', period_years=1)
    df = compute_all_indicators(df)

    print(f"\nShape after indicators: {df.shape}")
    print(f"\nAll columns:\n{list(df.columns)}")
    print(f"\nLast 3 rows (selected columns):")

    show_cols = ['Date', 'Close', 'RSI_14', 'MACD', 'BB_width',
                 'BB_position', 'EMA_cross', 'ATR_normalised', 'Daily_return']
    print(df[show_cols].tail(3).to_string(index=False))

    print(f"\nRows with any NaN: {df.isnull().any(axis=1).sum()}")
    print(f"Clean rows (after dropping NaN): {df.dropna().shape[0]}")

    # Quick sanity checks
    last = df.dropna().iloc[-1]
    rsi  = last['RSI_14']
    print(f"\nSanity checks on latest data:")
    print(f"  RSI_14 = {rsi:.1f} ({'overbought >70' if rsi > 70 else 'oversold <30' if rsi < 30 else 'neutral 30-70'})")
    print(f"  MACD histogram = {last['MACD_histogram']:+.3f} ({'bullish' if last['MACD_histogram'] > 0 else 'bearish'})")
    print(f"  BB position = {last['BB_position']:.2f} (0=lower band, 1=upper band)")
    print(f"  ATR normalised = {last['ATR_normalised']:.4f} ({last['ATR_normalised']*100:.2f}% daily volatility)")