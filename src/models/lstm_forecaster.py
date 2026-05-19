"""
lstm_forecaster.py
PyTorch LSTM for stock price forecasting using price + sentiment features.

Architecture:
  Input:  sequence of 30 days x 15 features
  LSTM:   2 layers, hidden_size=128, dropout=0.2
  Output: next 5 days of normalised close price

WHY LSTM over simple RNN:
  LSTMs have a cell state that can carry information across long sequences.
  Stock patterns (earnings cycles, seasonal trends) span weeks — a simple
  RNN forgets them. LSTM's forget/input/output gates handle this.

WHY 30-day sequence:
  Captures one full trading month of context. Shorter = misses monthly
  patterns. Longer = too much noise, slower training.

WHY predict 5 days:
  Anything beyond a week is noise for individual stocks. 5 days is one
  trading week — actionable and realistic.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import os
import sys
from typing import Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import SEQUENCE_LENGTH, FORECAST_HORIZON, MODELS_DIR, DEVICE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')


class LSTMForecaster(nn.Module):
    """
    2-layer LSTM with dropout for multi-step stock price forecasting.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 forecast_horizon: int = FORECAST_HORIZON):
        super().__init__()

        self.hidden_size      = hidden_size
        self.num_layers       = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True   # input shape: (batch, sequence, features)
        )

        self.dropout = nn.Dropout(dropout)

        # Output head: hidden state → forecast_horizon predictions
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, forecast_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            (batch_size, forecast_horizon) — predicted normalised close prices
        """
        lstm_out, _ = self.lstm(x)
        # Use only the last timestep's hidden state for prediction
        # WHY: the last hidden state summarises the entire sequence
        last_hidden = self.dropout(lstm_out[:, -1, :])
        return self.fc(last_hidden)


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    sequence_length: int = SEQUENCE_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert the feature matrix into (X, y) sequences for LSTM training.

    For each position t, we create:
      X[t] = features from day (t - sequence_length) to day (t-1)
      y[t] = normalised close prices from day t to day (t + horizon - 1)

    Args:
        df:               Feature matrix from feature_builder
        feature_cols:     Which columns to use as input features
        sequence_length:  Days of history per sample (default 30)
        forecast_horizon: Days to predict ahead (default 5)

    Returns:
        X:            (n_samples, sequence_length, n_features)
        y:            (n_samples, forecast_horizon)
        close_scaler: dict with min/max to inverse-transform predictions
    """
    # Normalise close price separately so we can inverse-transform later
    close_vals = df['Close'].values
    close_min  = close_vals.min()
    close_max  = close_vals.max()
    close_norm = (close_vals - close_min) / (close_max - close_min)

    close_scaler = {'min': close_min, 'max': close_max}

    # Normalise input features — clip OBV_change inf values first
    feature_data = df[feature_cols].copy()
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.fillna(0)

    # Min-max normalise each feature column
    for col in feature_cols:
        col_min = feature_data[col].min()
        col_max = feature_data[col].max()
        if col_max - col_min > 0:
            feature_data[col] = (feature_data[col] - col_min) / (col_max - col_min)
        else:
            feature_data[col] = 0.0

    features = feature_data.values

    X, y = [], []
    total = len(df)

    for i in range(sequence_length, total - forecast_horizon + 1):
        X.append(features[i - sequence_length:i])
        y.append(close_norm[i:i + forecast_horizon])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    logger.info(f"Sequences: X={X.shape}, y={y.shape}")
    logger.info(f"Close price range: ${close_min:.2f} - ${close_max:.2f}")

    return X, y, close_scaler


def train_lstm(
    df: pd.DataFrame,
    feature_cols: list,
    ticker: str = 'AAPL',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2
) -> Tuple[LSTMForecaster, dict]:
    """
    Train the LSTM model on the feature matrix.

    Args:
        df:            Feature matrix from feature_builder
        feature_cols:  Input feature column names
        ticker:        Stock symbol (used for saving model)
        epochs:        Training epochs (50 is enough for this data size)
        batch_size:    Samples per gradient update
        learning_rate: Adam optimizer LR
        val_split:     Fraction of data held out for validation

    Returns:
        Trained LSTMForecaster model
        Training history dict
    """
    logger.info(f"Preparing training sequences for {ticker}...")
    X, y, close_scaler = prepare_sequences(df, feature_cols)

    # Chronological train/val split — never shuffle time series data
    # WHY: shuffling leaks future data into training (look-ahead bias)
    split_idx  = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(_device)
    y_train_t = torch.FloatTensor(y_train).to(_device)
    X_val_t   = torch.FloatTensor(X_val).to(_device)
    y_val_t   = torch.FloatTensor(y_val).to(_device)

    # Build model
    input_size = X_train.shape[2]
    model = LSTMForecaster(input_size=input_size).to(_device)
    logger.info(f"Model: input_size={input_size}, device={_device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'close_scaler': close_scaler}
    best_val_loss  = float('inf')
    best_model_state = None

    logger.info(f"Training for {epochs} epochs on {_device}...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        # Mini-batch training
        for i in range(0, len(X_train_t), batch_size):
            xb = X_train_t[i:i + batch_size]
            yb = y_train_t[i:i + batch_size]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        train_loss = np.mean(train_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} — "
                        f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                        f"best_val={best_val_loss:.6f}")

    # Restore best weights
    model.load_state_dict(best_model_state)
    logger.info(f"Training complete. Best val_loss: {best_val_loss:.6f}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'lstm_{ticker.lower()}.pt')
    torch.save({
        'model_state': best_model_state,
        'close_scaler': close_scaler,
        'feature_cols': feature_cols,
        'input_size': input_size,
        'history': history
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    return model, history


def predict_next_days(
    model: LSTMForecaster,
    df: pd.DataFrame,
    feature_cols: list,
    close_scaler: dict
) -> List[float]:
    """
    Generate a 5-day price forecast using the most recent data.

    Args:
        model:        Trained LSTMForecaster
        df:           Feature matrix (needs at least SEQUENCE_LENGTH rows)
        feature_cols: Input feature columns
        close_scaler: Dict with min/max for inverse transformation

    Returns:
        List of 5 predicted closing prices in dollars
    """
    model.eval()

    # Get the last SEQUENCE_LENGTH rows as input
    feature_data = df[feature_cols].copy().tail(SEQUENCE_LENGTH)
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalise
    for col in feature_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max - col_min > 0:
            feature_data[col] = (feature_data[col] - col_min) / (col_max - col_min)
        else:
            feature_data[col] = 0.0

    x = torch.FloatTensor(feature_data.values).unsqueeze(0).to(_device)

    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()[0]

    # Inverse-transform from normalised to dollars
    close_min = close_scaler['min']
    close_max = close_scaler['max']
    predictions = pred_norm * (close_max - close_min) + close_min

    return [round(float(p), 2) for p in predictions]


def load_model(ticker: str) -> Tuple[LSTMForecaster, dict, list]:
    """
    Load a saved LSTM model from disk.

    Returns: (model, close_scaler, feature_cols)
    """
    model_path = os.path.join(MODELS_DIR, f'lstm_{ticker.lower()}.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model for {ticker} at {model_path}")

    checkpoint = torch.load(model_path, map_location=_device)

    model = LSTMForecaster(
        input_size=checkpoint['input_size']
    ).to(_device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    logger.info(f"Loaded LSTM model for {ticker} from {model_path}")
    return model, checkpoint['close_scaler'], checkpoint['feature_cols']


if __name__ == '__main__':
    print("=" * 55)
    print("Training LSTM for AAPL")
    print("=" * 55)

    from src.features.feature_builder import build_feature_matrix
    from src.features.technical_indicators import get_feature_columns

    # Build feature matrix
    df = build_feature_matrix('AAPL')

    feature_cols = get_feature_columns() + [
        'sentiment_ma3', 'sentiment_ma7',
        'sentiment_momentum', 'sentiment_vol'
    ]
    # Keep only columns that exist in df
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

    # Train
    model, history = train_lstm(
        df, feature_cols, ticker='AAPL', epochs=50
    )

    # Show training curve summary
    train_losses = history['train_loss']
    val_losses   = history['val_loss']
    print(f"\nTraining summary:")
    print(f"  Epoch  1: train={train_losses[0]:.6f}  val={val_losses[0]:.6f}")
    print(f"  Epoch 25: train={train_losses[24]:.6f}  val={val_losses[24]:.6f}")
    print(f"  Epoch 50: train={train_losses[49]:.6f}  val={val_losses[49]:.6f}")
    print(f"  Best val loss: {min(val_losses):.6f}")

    # Forecast
    close_scaler = history['close_scaler']
    forecast = predict_next_days(model, df, feature_cols, close_scaler)
    last_close = df['Close'].iloc[-1]

    print(f"\n5-day forecast for AAPL:")
    print(f"  Last close: ${last_close:.2f}")
    for i, price in enumerate(forecast, 1):
        change = ((price - last_close) / last_close) * 100
        arrow  = '▲' if price > last_close else '▼'
        print(f"  Day {i}: ${price:.2f}  {arrow} {change:+.2f}%")