import os
from dotenv import load_dotenv

load_dotenv()

# API keys
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')

# Defaults
DEFAULT_TICKER = os.getenv('DEFAULT_TICKER', 'AAPL')
CACHE_DB_PATH = os.getenv('CACHE_DB_PATH', 'data/cache/marketpulse.db')
NEWS_REFRESH_MINUTES = int(os.getenv('NEWS_REFRESH_MINUTES', 30))

# Model settings
SEQUENCE_LENGTH = 30        # days of history fed into LSTM
FORECAST_HORIZON = 5        # days to forecast ahead
FINBERT_MODEL = 'ProsusAI/finbert'
DEVICE = 'cuda'             # change to 'cpu' if GPU issues arise

# Paths
MODELS_DIR = 'models'
DATA_DIR = 'data'
CACHE_DIR = 'data/cache'
