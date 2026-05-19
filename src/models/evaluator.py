"""
evaluator.py
Evaluates and compares all forecasting models.

Metrics explained (for resume/README):
- MAE:  Mean Absolute Error — average dollar error (interpretable)
- RMSE: Root Mean Squared Error — penalises large errors more than MAE
- MAPE: Mean Absolute Percentage Error — % error, comparable across stocks
- Directional Accuracy: did we predict UP or DOWN correctly?
         Most important for trading — being $2 wrong but right direction
         is better than being $0.50 wrong but wrong direction.
- Random Walk Baseline: predicts tomorrow = today. Any model that cannot
         beat this is worthless. This is our absolute floor.
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray,
                          previous: np.ndarray) -> float:
    """
    What fraction of the time did we correctly predict UP vs DOWN?
    actual_dir   = sign(actual    - previous_close)
    predicted_dir= sign(predicted - previous_close)
    """
    actual_dir    = np.sign(actual    - previous)
    predicted_dir = np.sign(predicted - previous)
    return float(np.mean(actual_dir == predicted_dir) * 100)


def random_walk_baseline(closes: np.ndarray) -> np.ndarray:
    """
    Naive baseline: tomorrow's price = today's price.
    If your model can't beat this, it has no predictive power.
    """
    return closes[:-1]


def evaluate_all_models(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Run backtests for all models and return a comparison table.

    Uses the last 60 trading days as the test window.
    Predicts 1 day ahead for fair comparison across all models.

    Returns:
        DataFrame with rows = models, columns = metrics
    """
    TEST_SIZE = 60
    results   = {}

    closes = df['Close'].values

    # ── Random Walk Baseline ──────────────────────────────
    logger.info("Evaluating: Random Walk baseline...")
    test_actual   = closes[-TEST_SIZE:]
    rw_predicted  = closes[-TEST_SIZE - 1:-1]
    previous      = closes[-TEST_SIZE - 1:-1]

    results['Random Walk'] = {
        'MAE':  mae(test_actual, rw_predicted),
        'RMSE': rmse(test_actual, rw_predicted),
        'MAPE': mape(test_actual, rw_predicted),
        'Dir_Accuracy': directional_accuracy(test_actual, rw_predicted, previous)
    }

    # ── Prophet ───────────────────────────────────────────
    logger.info("Evaluating: Prophet (walk-forward backtest, ~60 fits)...")
    try:
        from src.models.baseline_prophet import prophet_backtest
        p_actual, p_pred = prophet_backtest(df, test_size=TEST_SIZE)

        if len(p_actual) > 0:
            p_prev = closes[len(closes) - len(p_actual) - 1:-1][:len(p_actual)]
            results['Prophet'] = {
                'MAE':  mae(p_actual, p_pred),
                'RMSE': rmse(p_actual, p_pred),
                'MAPE': mape(p_actual, p_pred),
                'Dir_Accuracy': directional_accuracy(p_actual, p_pred, p_prev)
            }
    except Exception as e:
        logger.error(f"Prophet evaluation failed: {e}")

    # ── LSTM ──────────────────────────────────────────────
    logger.info("Evaluating: LSTM (rolling window backtest)...")
    try:
        from src.models.lstm_forecaster import (
            LSTMForecaster, prepare_sequences, load_model
        )
        import torch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, close_scaler, saved_cols = load_model('AAPL')

        # Use saved feature cols
        X, y, cs = prepare_sequences(df, saved_cols, forecast_horizon=1)

        # Test on last TEST_SIZE samples
        X_test = torch.FloatTensor(X[-TEST_SIZE:]).to(device)
        y_test = y[-TEST_SIZE:, 0]

        model.eval()
        with torch.no_grad():
            # Retrain output projection for 1-step if needed
            pred_norm = model(X_test).cpu().numpy()[:, 0]

        c_min = close_scaler['min']
        c_max = close_scaler['max']
        lstm_pred   = pred_norm * (c_max - c_min) + c_min
        lstm_actual = y_test    * (c_max - c_min) + c_min
        lstm_prev   = closes[-TEST_SIZE - 1:-1]

        results['LSTM'] = {
            'MAE':  mae(lstm_actual, lstm_pred),
            'RMSE': rmse(lstm_actual, lstm_pred),
            'MAPE': mape(lstm_actual, lstm_pred),
            'Dir_Accuracy': directional_accuracy(lstm_actual, lstm_pred, lstm_prev)
        }
    except Exception as e:
        logger.error(f"LSTM evaluation failed: {e}")
        import traceback; traceback.print_exc()

    # ── Build results table ───────────────────────────────
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            'Model':         model_name,
            'MAE ($)':       round(metrics['MAE'], 2),
            'RMSE ($)':      round(metrics['RMSE'], 2),
            'MAPE (%)':      round(metrics['MAPE'], 2),
            'Dir_Acc (%)':   round(metrics['Dir_Accuracy'], 1)
        })

    results_df = pd.DataFrame(rows)
    return results_df


if __name__ == '__main__':
    print("=" * 60)
    print("Model Evaluation — AAPL")
    print("=" * 60)

    from src.features.feature_builder import build_feature_matrix
    from src.features.technical_indicators import get_feature_columns

    df = build_feature_matrix('AAPL')
    feature_cols = get_feature_columns() + [
        'sentiment_ma3', 'sentiment_ma7',
        'sentiment_momentum', 'sentiment_vol'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    results_df = evaluate_all_models(df, feature_cols)

    print(f"\n{'='*60}")
    print("RESULTS (last 60 trading days, 1-day-ahead prediction)")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    print(f"\nInterpretation:")
    if 'LSTM' in results_df['Model'].values and 'Random Walk' in results_df['Model'].values:
        lstm_mape = results_df.loc[results_df['Model']=='LSTM', 'MAPE (%)'].values[0]
        rw_mape   = results_df.loc[results_df['Model']=='Random Walk', 'MAPE (%)'].values[0]
        lstm_dir  = results_df.loc[results_df['Model']=='LSTM', 'Dir_Acc (%)'].values[0]

        print(f"  Random Walk MAPE: {rw_mape:.2f}%  (baseline floor)")
        print(f"  LSTM MAPE:        {lstm_mape:.2f}%")
        if lstm_mape < rw_mape:
            print(f"  ✓ LSTM beats random walk on MAPE")
        else:
            print(f"  ✗ LSTM does not beat random walk yet — more training data needed")
        print(f"  LSTM directional accuracy: {lstm_dir:.1f}%  (50% = coin flip)")