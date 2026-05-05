# MarketPulse

Real-time financial news sentiment analysis + stock price forecasting dashboard.

## What it does
- Pulls live financial news for any stock ticker (AAPL, RELIANCE.NS, etc.)
- Runs FinBERT sentiment analysis on each article
- Aggregates sentiment into a daily sentiment score timeline
- Trains an LSTM + sentiment-enhanced forecasting model
- Forecasts the next 5 days of price movement
- Displays everything in a live Streamlit dashboard with Plotly charts
- Exposes a FastAPI endpoint for programmatic access

## Tech stack
| Layer | Technology |
|---|---|
| Price data | yfinance |
| News data | NewsAPI |
| Sentiment | FinBERT (ProsusAI/finbert) |
| Indicators | ta, pandas-ta |
| Forecasting | Prophet, PyTorch LSTM |
| Storage | SQLite + SQLAlchemy |
| API | FastAPI |
| Dashboard | Streamlit |

## Setup

### 1. Create conda environment
`ash
conda create -n marketpulse python=3.10 -y
conda activate marketpulse
`

### 2. Install dependencies
`ash
pip install -r requirements.txt
`

### 3. Configure API keys
`ash
cp .env.example .env
# Edit .env and add your NewsAPI and Alpha Vantage keys
`

### 4. Run the dashboard
`ash
streamlit run src/app/streamlit_app.py
`

### 5. Run the API
`ash
uvicorn src.api.main:app --reload
`

## Project structure
marketpulse/
├── src/
│   ├── data/          # News + price fetchers, SQLite cache, scheduler
│   ├── sentiment/     # FinBERT inference + daily aggregation
│   ├── features/      # Technical indicators + feature builder
│   ├── models/        # Prophet, LSTM, ensemble, evaluator
│   ├── api/           # FastAPI routes + Pydantic schemas
│   └── app/           # Streamlit dashboard
├── data/cache/        # SQLite databases (gitignored)
├── models/            # Saved model weights (gitignored)
├── notebooks/         # Exploration notebooks
└── tests/             # Pytest test suite
## Author
B.Tech CSE — CSPIT, CHARUSAT
