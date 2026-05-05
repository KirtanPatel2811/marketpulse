"""
finbert_model.py
FinBERT inference pipeline for financial news sentiment analysis.

WHY FinBERT over generic BERT:
- Trained specifically on financial text (10-K filings, earnings calls, news)
- Understands domain terms: "beats estimates", "guidance cut", "margin expansion"
- Generic models misclassify financial language e.g. "Apple crushed earnings"
  (positive) might be read as negative by a model trained on general text.

Model: ProsusAI/finbert from HuggingFace
Output: Positive / Negative / Neutral with confidence score
GPU: Runs on RTX 3060 — ~50-100 articles/second
"""

import torch
import logging
import os
import sys
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import FINBERT_MODEL, DEVICE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve device — fall back to CPU gracefully if CUDA unavailable
_device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
logger.info(f"FinBERT will run on: {_device}")

# Module-level model cache so we only load weights once per session
# WHY: Loading FinBERT takes ~3-4 seconds. Caching means the second call
# to score_articles() is instant instead of reloading from disk.
_tokenizer = None
_model = None


def load_model():
    """
    Load FinBERT tokenizer and model into memory.
    Downloads weights from HuggingFace on first run (~440MB), then caches locally.
    Safe to call multiple times — returns immediately if already loaded.
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return  # already loaded

    logger.info(f"Loading FinBERT model: {FINBERT_MODEL}")
    logger.info("First run will download ~440MB of model weights. Please wait...")

    try:
        _tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
        _model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
        _model.to(_device)
        _model.eval()  # inference mode — disables dropout
        logger.info("FinBERT loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load FinBERT: {e}")
        raise


def score_text(text: str) -> Dict:
    """
    Score a single piece of text with FinBERT.

    Args:
        text: The text to score (headline + description works best)

    Returns:
        Dict with keys:
          label         — 'positive', 'negative', or 'neutral'
          score         — confidence of the predicted label (0.0 to 1.0)
          numeric_score — directional score: +score if positive,
                          -score if negative, 0 if neutral
          positive      — raw probability for positive class
          negative      — raw probability for negative class
          neutral       — raw probability for neutral class

    WHY numeric_score: When aggregating daily sentiment, we need a single
    number per article. +0.9 means strongly positive, -0.8 means strongly
    negative, ~0 means neutral. This is what gets averaged into the
    daily sentiment score that feeds the LSTM.
    """
    load_model()

    if not text or not text.strip():
        return {
            'label': 'neutral', 'score': 1.0, 'numeric_score': 0.0,
            'positive': 0.0, 'negative': 0.0, 'neutral': 1.0
        }

    try:
        # FinBERT max input is 512 tokens — truncate long articles
        inputs = _tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # FinBERT label order: positive=0, negative=1, neutral=2
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
        predicted_idx = probs.argmax()
        label = label_map[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Numeric score for aggregation
        if label == 'positive':
            numeric_score = float(probs[0])
        elif label == 'negative':
            numeric_score = -float(probs[1])
        else:
            numeric_score = 0.0

        return {
            'label': label,
            'score': round(confidence, 4),
            'numeric_score': round(numeric_score, 4),
            'positive': round(float(probs[0]), 4),
            'negative': round(float(probs[1]), 4),
            'neutral': round(float(probs[2]), 4)
        }

    except Exception as e:
        logger.error(f"Error scoring text: {e}")
        return {
            'label': 'neutral', 'score': 0.0, 'numeric_score': 0.0,
            'positive': 0.0, 'negative': 0.0, 'neutral': 0.0
        }


def score_articles(articles: List[Dict]) -> List[Dict]:
    """
    Score a list of news article dicts in batch.

    Args:
        articles: List of article dicts from news_fetcher or data_store.
                  Each must have at least a 'title' key.

    Returns:
        Same list with sentiment fields added to each article:
        sentiment, sentiment_score, positive, negative, neutral

    WHY title + description: The headline carries the most signal.
    The description adds context without exceeding 512 tokens.
    Body content is usually truncated by NewsAPI anyway.
    """
    load_model()

    if not articles:
        return []

    logger.info(f"Scoring {len(articles)} articles with FinBERT on {_device}")
    scored = []

    for i, article in enumerate(articles):
        # Combine title and description for richer context
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title}. {description}".strip()

        result = score_text(text)

        # Merge sentiment results back into article dict
        enriched = dict(article)
        enriched['sentiment'] = result['label']
        enriched['sentiment_score'] = result['numeric_score']
        enriched['sentiment_positive'] = result['positive']
        enriched['sentiment_negative'] = result['negative']
        enriched['sentiment_neutral'] = result['neutral']
        enriched['sentiment_confidence'] = result['score']

        scored.append(enriched)

        if (i + 1) % 5 == 0:
            logger.info(f"  Scored {i+1}/{len(articles)} articles...")

    pos = sum(1 for a in scored if a['sentiment'] == 'positive')
    neg = sum(1 for a in scored if a['sentiment'] == 'negative')
    neu = sum(1 for a in scored if a['sentiment'] == 'neutral')
    logger.info(f"Sentiment summary — Positive: {pos}, Negative: {neg}, Neutral: {neu}")

    return scored


if __name__ == '__main__':
    print("=" * 55)
    print("Testing finbert_model.py")
    print("=" * 55)

    test_sentences = [
        "Apple reports record-breaking Q2 2026 results: $29.6B profit on $111.2B revenue",
        "Apple stock drops 5% after weak iPhone sales guidance disappoints investors",
        "Apple says Mac Studio and Mac Mini will be in short supply for months",
        "Apple leads global market for satellite-connected smartphones",
        "Tim Cook says Mac Mini is being snapped up for AI work faster than expected",
    ]

    print(f"\nDevice: {_device}")
    print("\nLoading model (downloads ~440MB on first run)...\n")

    for sentence in test_sentences:
        result = score_text(sentence)
        bar = "+" * int(result['positive'] * 20) + "-" * int(result['negative'] * 20)
        print(f"Text:    {sentence[:70]}")
        print(f"Label:   {result['label'].upper():8} | Score: {result['numeric_score']:+.3f} | [{bar}]")
        print(f"         pos={result['positive']:.3f}  neg={result['negative']:.3f}  neu={result['neutral']:.3f}")
        print()