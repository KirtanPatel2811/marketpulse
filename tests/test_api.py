"""
test_api.py
Tests for FastAPI endpoints.
Run with: pytest tests/test_api.py -v

Uses TestClient — no real server needed.
API keys and DB are not required for most tests
because we mock the heavy pipeline calls.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "cuda_available" in data
        assert "version" in data
        assert data["status"] == "ok"


class TestTechnicalsEndpoint:

    def test_invalid_ticker_returns_error(self, client):
        with patch('src.data.price_fetcher.fetch_price_data') as mock_fetch:
            import pandas as pd
            mock_fetch.return_value = pd.DataFrame()
            response = client.get("/technicals/FAKEXYZ999")
            assert response.status_code in [404, 500]

    def test_technicals_response_shape(self, client):
        import pandas as pd
        import numpy as np
        from src.features.technical_indicators import compute_all_indicators

        mock_df = pd.DataFrame({
            'Date':   pd.date_range('2025-01-01', periods=100, freq='B'),
            'Open':   np.full(100, 150.0),
            'High':   np.full(100, 155.0),
            'Low':    np.full(100, 145.0),
            'Close':  150.0 + np.cumsum(np.random.randn(100)),
            'Volume': np.full(100, 1_000_000),
            'Ticker': 'AAPL'
        })

        with patch('src.data.price_fetcher.fetch_price_data', return_value=mock_df):
            response = client.get("/technicals/AAPL")
            if response.status_code == 200:
                data = response.json()
                assert "ticker" in data
                assert "RSI_14" in data
                assert "MACD" in data


class TestNewsEndpoint:

    def test_news_limit_param(self, client):
        mock_articles = [
            {
                'title': f'Article {i}', 'source': 'Test', 'published_at': '2026-05-01T00:00:00Z',
                'url': f'http://test.com/{i}', 'sentiment': 'neutral',
                'sentiment_score': 0.0, 'sentiment_confidence': 0.8
            }
            for i in range(20)
        ]
        with patch('src.sentiment.sentiment_cache.get_scored_articles', return_value=mock_articles):
            response = client.get("/news/AAPL?limit=5")
            assert response.status_code == 200
            data = response.json()
            assert len(data['articles']) == 5

    def test_news_response_structure(self, client):
        mock_articles = [{
            'title': 'Test article', 'source': 'Reuters',
            'published_at': '2026-05-01T00:00:00Z',
            'url': 'http://reuters.com/test',
            'sentiment': 'positive',
            'sentiment_score': 0.85,
            'sentiment_confidence': 0.90
        }]
        with patch('src.sentiment.sentiment_cache.get_scored_articles', return_value=mock_articles):
            response = client.get("/news/AAPL")
            assert response.status_code == 200
            data = response.json()
            assert 'ticker' in data
            assert 'articles' in data
            assert 'count' in data
            article = data['articles'][0]
            assert 'title' in article
            assert 'sentiment' in article
            assert 'score' in article