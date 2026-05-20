"""
test_sentiment.py
Tests for the FinBERT sentiment pipeline.
Run with: pytest tests/test_sentiment.py -v
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath('.'))


class TestSentimentScoring:
    """Tests for finbert_model.py"""

    def test_positive_sentence(self):
        from src.sentiment.finbert_model import score_text
        result = score_text("Apple reports record-breaking quarterly earnings beating all estimates")
        assert result['label'] == 'positive'
        assert result['score'] > 0.5
        assert result['numeric_score'] > 0

    def test_negative_sentence(self):
        from src.sentiment.finbert_model import score_text
        result = score_text("Apple stock crashes after massive revenue miss and guidance cut")
        assert result['label'] == 'negative'
        assert result['score'] > 0.5
        assert result['numeric_score'] < 0

    def test_empty_text_returns_neutral(self):
        from src.sentiment.finbert_model import score_text
        result = score_text("")
        assert result['label'] == 'neutral'
        assert result['numeric_score'] == 0.0

    def test_score_keys_present(self):
        from src.sentiment.finbert_model import score_text
        result = score_text("Apple announces new product launch")
        required_keys = ['label', 'score', 'numeric_score', 'positive', 'negative', 'neutral']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_probabilities_sum_to_one(self):
        from src.sentiment.finbert_model import score_text
        result = score_text("Apple quarterly results meet expectations")
        total = result['positive'] + result['negative'] + result['neutral']
        assert abs(total - 1.0) < 0.01

    def test_score_articles_batch(self):
        from src.sentiment.finbert_model import score_articles
        articles = [
            {'title': 'Apple beats earnings estimates', 'description': 'Strong iPhone demand'},
            {'title': 'Apple stock drops on weak guidance', 'description': 'Revenue outlook disappoints'},
        ]
        scored = score_articles(articles)
        assert len(scored) == 2
        for a in scored:
            assert 'sentiment' in a
            assert 'sentiment_score' in a
            assert 'sentiment_confidence' in a


class TestSentimentAggregation:
    """Tests for sentiment_aggregator.py"""

    def test_empty_articles_returns_empty_df(self):
        from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment
        result = aggregate_daily_sentiment([])
        assert result.empty

    def test_aggregation_produces_correct_columns(self):
        from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment
        articles = [
            {
                'published_at': '2026-05-01T10:00:00Z',
                'sentiment': 'positive',
                'sentiment_score': 0.85,
                'sentiment_confidence': 0.90
            },
            {
                'published_at': '2026-05-01T14:00:00Z',
                'sentiment': 'negative',
                'sentiment_score': -0.70,
                'sentiment_confidence': 0.75
            }
        ]
        df = aggregate_daily_sentiment(articles)
        assert not df.empty
        assert 'sentiment_score' in df.columns
        assert 'sentiment_label' in df.columns
        assert 'article_count' in df.columns

    def test_single_positive_day(self):
        from src.sentiment.sentiment_aggregator import aggregate_daily_sentiment
        articles = [{
            'published_at': '2026-05-01T10:00:00Z',
            'sentiment': 'positive',
            'sentiment_score': 0.90,
            'sentiment_confidence': 0.92
        }]
        df = aggregate_daily_sentiment(articles)
        assert df.iloc[0]['sentiment_label'] == 'positive'
        assert df.iloc[0]['sentiment_score'] > 0