import sys
sys.path.insert(0, '.')
from src.features.feature_builder import build_feature_matrix
from src.features.technical_indicators import get_feature_columns
from src.models.lstm_forecaster import train_lstm

for ticker in ['MSFT', 'NVDA']:
    print(f'\nTraining {ticker}...')
    df = build_feature_matrix(ticker)
    cols = get_feature_columns() + ['sentiment_ma3','sentiment_ma7','sentiment_momentum','sentiment_vol']
    cols = [c for c in cols if c in df.columns]
    model, history = train_lstm(df, cols, ticker=ticker, epochs=50)
    best = min(history['val_loss'])
    print(f'{ticker} done. Best val loss: {best:.6f}')