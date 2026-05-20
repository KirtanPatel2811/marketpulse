"""
test_features.py
Tests for technical indicators and feature builder.
Run with: pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))


def make_price_df(n=100):
    """Generate a synthetic OHLCV DataFrame for testing."""
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        'Date':   dates,
        'Open':   close * 0.99,
        'High':   close * 1.02,
        'Low':    close * 0.98,
        'Close':  close,
        'Volume': np.random.randint(1_000_000, 5_000_000, n),
        'Ticker': 'TEST'
    })


class TestTechnicalIndicators:

    def test_rsi_range(self):
        from src.features.technical_indicators import add_rsi
        df = add_rsi(make_price_df(100))
        rsi = df['RSI_14'].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_columns_exist(self):
        from src.features.technical_indicators import add_macd
        df = add_macd(make_price_df(100))
        assert 'MACD' in df.columns
        assert 'MACD_signal' in df.columns
        assert 'MACD_histogram' in df.columns

    def test_bollinger_bands(self):
        from src.features.technical_indicators import add_bollinger_bands
        df = add_bollinger_bands(make_price_df(100))
        clean = df.dropna()
        assert (clean['BB_upper'] >= clean['Close']).mean() > 0.8
        assert (clean['BB_lower'] <= clean['Close']).mean() > 0.8

    def test_ema_columns(self):
        from src.features.technical_indicators import add_ema
        df = add_ema(make_price_df(100), periods=[20, 50])
        assert 'EMA_20' in df.columns
        assert 'EMA_50' in df.columns
        assert 'EMA_cross' in df.columns

    def test_compute_all_indicators_shape(self):
        from src.features.technical_indicators import compute_all_indicators
        df = compute_all_indicators(make_price_df(200))
        assert df.shape[0] == 200
        assert df.shape[1] > 20  # should have added many columns

    def test_clean_rows_after_dropna(self):
        from src.features.technical_indicators import compute_all_indicators
        df = compute_all_indicators(make_price_df(200)).dropna()
        assert len(df) >= 150  # at least 150 clean rows from 200


class TestFeatureBuilder:

    def test_sentiment_features_added(self):
        from src.features.feature_builder import add_sentiment_features
        df = make_price_df(60)
        df['sentiment_score'] = 0.1
        df = add_sentiment_features(df)
        assert 'sentiment_ma3' in df.columns
        assert 'sentiment_ma7' in df.columns
        assert 'sentiment_momentum' in df.columns

    def test_missing_sentiment_fills_zero(self):
        from src.features.feature_builder import add_sentiment_features
        df = make_price_df(60)
        # No sentiment_score column
        df = add_sentiment_features(df)
        assert 'sentiment_score' in df.columns
        assert (df['sentiment_score'] == 0.0).all()

    def test_get_feature_columns_returns_list(self):
        from src.features.technical_indicators import get_feature_columns
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 10
        assert 'sentiment_score' in cols
        assert 'RSI_14' in cols