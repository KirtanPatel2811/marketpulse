"""
baseline_prophet.py
Prophet baseline forecaster.

WHY Prophet as baseline:
- Runs in seconds, no GPU needed
- Handles trend + seasonality automatically
- If LSTM cannot beat Prophet, sentiment features are not adding value
- Prophet is our floor: a model that cannot outperform this is useless

Facebook Prophet expects a DataFrame with columns 'ds' (date) and 'y' (value).
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import FORECAST_HORIZON

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_prophet(df: pd.DataFrame) -> object:
    """
    Train a Prophet model on historical closing prices.

    Args:
        df: Feature matrix with Date and Close columns

    Returns:
        Fitted Prophet model
    """
    from prophet import Prophet

    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Date']),
        'y':  df['Close'].values
    })

    model = Prophet(
        daily_seasonality=False,   # stocks don't have intraday patterns here
        weekly_seasonality=True,   # Mon-Fri trading patterns
        yearly_seasonality=True,   # earnings season patterns
        changepoint_prior_scale=0.05,  # how flexible the trend is
        interval_width=0.95
    )

    # Suppress Prophet's verbose output
    import logging as _logging
    _logging.getLogger('prophet').setLevel(_logging.WARNING)
    _logging.getLogger('cmdstanpy').setLevel(_logging.WARNING)

    model.fit(prophet_df)
    logger.info("Prophet model fitted successfully")
    return model


def prophet_forecast(model, periods: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Generate forecast for next N trading days.

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    future = model.make_future_dataframe(periods=periods, freq='B')  # B = business days
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    return result.reset_index(drop=True)


def prophet_backtest(df: pd.DataFrame, test_size: int = 90) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward backtest for Prophet.
    Trains on all data before each test date, predicts 5 days ahead.
    Returns actual and predicted arrays for evaluation.

    WHY walk-forward (not simple train/test split):
    Each prediction uses only data available at that point in time.
    This simulates real-world usage and prevents look-ahead bias.
    """
    from prophet import Prophet
    import logging as _logging
    _logging.getLogger('prophet').setLevel(_logging.ERROR)
    _logging.getLogger('cmdstanpy').setLevel(_logging.ERROR)

    actuals    = []
    predictions = []

    # We predict 1 day ahead for simplicity in backtesting
    for i in range(len(df) - test_size, len(df) - 1):
        train_slice = df.iloc[:i]

        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(train_slice['Date']),
            'y':  train_slice['Close'].values
        })

        try:
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=1, freq='B')
            fc     = m.predict(future)
            pred   = fc['yhat'].iloc[-1]

            actual = df['Close'].iloc[i]
            actuals.append(actual)
            predictions.append(pred)

        except Exception as e:
            logger.warning(f"Prophet backtest failed at index {i}: {e}")
            continue

        if len(actuals) % 20 == 0:
            logger.info(f"  Prophet backtest: {len(actuals)}/{test_size} done...")

    return np.array(actuals), np.array(predictions)


if __name__ == '__main__':
    print("=" * 55)
    print("Testing Prophet baseline for AAPL")
    print("=" * 55)

    from src.features.feature_builder import build_feature_matrix

    df = build_feature_matrix('AAPL')

    print(f"\nFitting Prophet on {len(df)} rows...")
    model = train_prophet(df)

    forecast = prophet_forecast(model, periods=FORECAST_HORIZON)
    last_close = df['Close'].iloc[-1]

    print(f"\n5-day Prophet forecast for AAPL:")
    print(f"  Last close: ${last_close:.2f}")
    for _, row in forecast.iterrows():
        change = ((row['yhat'] - last_close) / last_close) * 100
        arrow  = '▲' if row['yhat'] > last_close else '▼'
        print(f"  {row['ds'].strftime('%Y-%m-%d')}: "
              f"${row['yhat']:.2f}  {arrow} {change:+.2f}%  "
              f"[${row['yhat_lower']:.2f} - ${row['yhat_upper']:.2f}]")