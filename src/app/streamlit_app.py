"""
streamlit_app.py
MarketPulse interactive dashboard — 5 screens:
  1. Ticker Overview    — live price + sentiment gauge + top news
  2. Sentiment Timeline — daily sentiment overlaid on price chart
  3. Forecast           — 5-day LSTM + Prophet predictions
  4. Technical Analysis — RSI, MACD, Bollinger Bands
  5. News Feed          — colour-coded articles with sentiment scores
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached data loaders (cache for 5 min to avoid re-running FinBERT on refresh) ──


@st.cache_data(ttl=300)
def load_price(ticker, years=2):
    from src.data.price_fetcher import fetch_price_data

    return fetch_price_data(ticker, period_years=years)


@st.cache_data(ttl=300)
def load_feature_matrix(ticker):
    from src.features.feature_builder import build_feature_matrix

    return build_feature_matrix(ticker)


@st.cache_data(ttl=300)
def load_sentiment_pipeline(ticker, days=7):
    from src.sentiment.sentiment_cache import run_full_sentiment_pipeline

    return run_full_sentiment_pipeline(ticker, days=days)


@st.cache_data(ttl=300)
def load_latest_price(ticker):
    from src.data.price_fetcher import get_latest_price

    return get_latest_price(ticker)


@st.cache_resource
def load_lstm(ticker):
    try:
        from src.models.lstm_forecaster import load_model

        return load_model(ticker)
    except FileNotFoundError:
        return None, None, None


# ── Sidebar ──────────────────────────────────────────────

with st.sidebar:
    st.title("📈 MarketPulse")
    st.caption("Real-time sentiment + price forecasting")
    st.divider()

    ticker = (
        st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="US tickers: AAPL, MSFT, NVDA. Indian: RELIANCE.NS, TCS.NS",
        )
        .upper()
        .strip()
    )

    screen = st.radio(
        "Screen",
        [
            "Overview",
            "Sentiment Timeline",
            "Forecast",
            "Technical Analysis",
            "News Feed",
        ],
        index=0,
    )

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("Cache clears every 5 minutes automatically.")

# ── Load data (shared across all screens) ────────────────

with st.spinner(f"Loading data for {ticker}..."):
    price_df = load_price(ticker)
    latest = load_latest_price(ticker)

if price_df.empty:
    st.error(f"❌ Could not fetch data for '{ticker}'. Check the ticker symbol.")
    st.stop()

# ── SCREEN 1: OVERVIEW ───────────────────────────────────

if screen == "Overview":
    st.title(f"📊 {ticker} — Overview")

    # Top metrics row
    if latest:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Last Price",
            f"${latest['price']:,.2f}",
            f"{latest['change']:+.2f} ({latest['change_pct']:+.2f}%)",
        )
        c2.metric("Previous Close", f"${latest['previous_close']:,.2f}")
        c3.metric("Volume", f"{latest['volume']:,}")
        c4.metric("As of", latest["date"])

    st.divider()

    # Candlestick chart — last 90 days
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Price Chart (90 days)")
        df90 = price_df.tail(90)
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df90["Date"],
                    open=df90["Open"],
                    high=df90["High"],
                    low=df90["Low"],
                    close=df90["Close"],
                    name=ticker,
                    increasing_line_color="#00c896",
                    decreasing_line_color="#ff4b4b",
                )
            ]
        )
        fig.update_layout(
            height=380,
            xaxis_rangeslider_visible=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment Gauge")
        result = load_sentiment_pipeline(ticker, days=7)
        articles = result.get("articles", [])

        if articles:
            pos = sum(1 for a in articles if a.get("sentiment") == "positive")
            neg = sum(1 for a in articles if a.get("sentiment") == "negative")
            neu = sum(1 for a in articles if a.get("sentiment") == "neutral")
            total = len(articles)
            avg_score = np.mean([a.get("sentiment_score", 0) or 0 for a in articles])

            # Gauge chart
            fig_g = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=round((avg_score + 1) * 50, 1),
                    title={"text": "Sentiment Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00c896" if avg_score > 0 else "#ff4b4b"},
                        "steps": [
                            {"range": [0, 35], "color": "rgba(255,75,75,0.2)"},
                            {"range": [35, 65], "color": "rgba(255,165,0,0.2)"},
                            {"range": [65, 100], "color": "rgba(0,200,150,0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": "white", "width": 3},
                            "thickness": 0.75,
                            "value": round((avg_score + 1) * 50, 1),
                        },
                    },
                )
            )
            fig_g.update_layout(
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Breakdown
            st.markdown(
                f"🟢 **{pos}** positive &nbsp; 🔴 **{neg}** negative &nbsp; ⚪ **{neu}** neutral"
            )
            st.caption(f"Based on {total} articles (last 7 days)")
        else:
            st.info("No sentiment data available yet.")

    st.divider()
    st.subheader("📰 Top News (last 7 days)")

    result = load_sentiment_pipeline(ticker, days=7)
    articles = result.get("articles", [])[:5]

    if articles:
        for a in articles:
            label = a.get("sentiment", "neutral")
            score = a.get("sentiment_score", 0) or 0
            color = (
                "#00c896"
                if label == "positive"
                else "#ff4b4b" if label == "negative" else "#888888"
            )
            emoji = (
                "🟢" if label == "positive" else "🔴" if label == "negative" else "⚪"
            )
            with st.container():
                st.markdown(
                    f"{emoji} **[{a['title'][:90]}]({a.get('url','#')})**  \n"
                    f"<span style='color:{color}; font-size:12px'>{label.upper()} {score:+.3f}</span> "
                    f"<span style='color:gray; font-size:12px'>— {a.get('source','')} · {a.get('published_at','')[:10]}</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("No news articles in cache. Click Refresh Data.")

# ── SCREEN 2: SENTIMENT TIMELINE ─────────────────────────

elif screen == "Sentiment Timeline":
    st.title(f"📉 {ticker} — Sentiment Timeline")

    result = load_sentiment_pipeline(ticker, days=30)
    daily_df = result.get("daily", pd.DataFrame())

    if daily_df.empty:
        st.warning("No daily sentiment data yet. Run the sentiment pipeline first.")
        st.stop()

    df_plot = price_df.tail(60).copy()
    df_plot["Date"] = pd.to_datetime(df_plot["Date"])
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=[f"{ticker} Close Price", "Daily Sentiment Score"],
    )

    fig.add_trace(
        go.Scatter(
            x=df_plot["Date"],
            y=df_plot["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#4a9eff", width=2),
        ),
        row=1,
        col=1,
    )

    colors = [
        "#00c896" if s > 0.1 else "#ff4b4b" if s < -0.1 else "#888888"
        for s in daily_df["sentiment_score"]
    ]

    fig.add_trace(
        go.Bar(
            x=daily_df["Date"],
            y=daily_df["sentiment_score"],
            name="Daily Sentiment",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=2, col=1)

    fig.update_layout(
        height=550,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Sentiment Table")
    st.dataframe(
        daily_df[
            ["Date", "sentiment_score", "sentiment_label", "article_count"]
        ].sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ── SCREEN 3: FORECAST ───────────────────────────────────

elif screen == "Forecast":
    st.title(f"🔮 {ticker} — 5-Day Forecast")

    with st.spinner("Loading feature matrix and model..."):
        df = load_feature_matrix(ticker)
        model, close_scaler, feature_cols = load_lstm(ticker)

    if model is None:
        st.warning(
            f"No trained LSTM model found for {ticker}. Run `python src/models/lstm_forecaster.py` first."
        )
        st.stop()

    from src.models.lstm_forecaster import predict_next_days
    from src.models.baseline_prophet import train_prophet, prophet_forecast

    # LSTM forecast
    lstm_forecast = predict_next_days(model, df, feature_cols, close_scaler)

    # Prophet forecast
    with st.spinner("Running Prophet..."):
        prophet_model = train_prophet(df)
        prophet_fc = prophet_forecast(prophet_model)

    last_close = df["Close"].iloc[-1]
    last_date = pd.to_datetime(df["Date"].iloc[-1])

    import pandas.tseries.offsets as offsets

    future_dates = pd.bdate_range(start=last_date + offsets.BDay(1), periods=5)

    # Build forecast table
    rows = []
    for i, (d, lp, pp) in enumerate(
        zip(future_dates, lstm_forecast, prophet_fc["yhat"].values)
    ):
        rows.append(
            {
                "Date": d.strftime("%Y-%m-%d"),
                "LSTM ($)": lp,
                "Prophet ($)": round(float(pp), 2),
                "LSTM Δ%": round((lp - last_close) / last_close * 100, 2),
                "Prophet Δ%": round((float(pp) - last_close) / last_close * 100, 2),
            }
        )
    fc_df = pd.DataFrame(rows)

    # Metrics
    c1, c2 = st.columns(2)
    c1.metric("Last Close", f"${last_close:.2f}")
    c2.metric(
        "LSTM Day 1",
        f"${lstm_forecast[0]:.2f}",
        f"{(lstm_forecast[0]-last_close)/last_close*100:+.2f}%",
    )

    # Chart
    fig = go.Figure()
    hist = df.tail(30)
    fig.add_trace(
        go.Scatter(
            x=hist["Date"],
            y=hist["Close"],
            mode="lines",
            name="Historical",
            line=dict(color="#4a9eff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_dates),
            y=lstm_forecast,
            mode="lines+markers",
            name="LSTM Forecast",
            line=dict(color="#00c896", width=2, dash="dash"),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(future_dates),
            y=prophet_fc["yhat"].values,
            mode="lines+markers",
            name="Prophet Forecast",
            line=dict(color="#ffa500", width=2, dash="dot"),
            marker=dict(size=8),
        )
    )
    fig.add_vrect(
        x0=str(future_dates[0].date()),
        x1=str(future_dates[-1].date()),
        fillcolor="rgba(255,255,255,0.03)",
        line_width=0,
    )
    fig.update_layout(
        height=420,
        title=f"{ticker} — 5-Day Forecast",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

# ── SCREEN 4: TECHNICAL ANALYSIS ─────────────────────────

elif screen == "Technical Analysis":
    st.title(f"🔧 {ticker} — Technical Analysis")

    from src.features.technical_indicators import compute_all_indicators

    df_tech = compute_all_indicators(price_df).dropna().tail(90)

    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(
            x=df_tech["Date"],
            y=df_tech["RSI_14"],
            mode="lines",
            name="RSI(14)",
            line=dict(color="#4a9eff"),
        )
    )
    fig_rsi.add_hline(
        y=70,
        line_dash="dash",
        line_color="#ff4b4b",
        opacity=0.7,
        annotation_text="Overbought (70)",
    )
    fig_rsi.add_hline(
        y=30,
        line_dash="dash",
        line_color="#00c896",
        opacity=0.7,
        annotation_text="Oversold (30)",
    )
    fig_rsi.update_layout(
        height=250,
        title="RSI (14)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    macd_colors = [
        "#00c896" if v >= 0 else "#ff4b4b" for v in df_tech["MACD_histogram"]
    ]
    fig_macd = go.Figure()
    fig_macd.add_trace(
        go.Bar(
            x=df_tech["Date"],
            y=df_tech["MACD_histogram"],
            name="Histogram",
            marker_color=macd_colors,
            opacity=0.7,
        )
    )
    fig_macd.add_trace(
        go.Scatter(
            x=df_tech["Date"],
            y=df_tech["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="#4a9eff"),
        )
    )
    fig_macd.add_trace(
        go.Scatter(
            x=df_tech["Date"],
            y=df_tech["MACD_signal"],
            mode="lines",
            name="Signal",
            line=dict(color="#ffa500"),
        )
    )
    fig_macd.update_layout(
        height=250,
        title="MACD",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(
        go.Scatter(
            x=df_tech["Date"],
            y=df_tech["BB_upper"],
            mode="lines",
            name="Upper Band",
            line=dict(color="#ff4b4b", dash="dash"),
            opacity=0.7,
        )
    )
    fig_bb.add_trace(
        go.Scatter(
            x=df_tech["Date"],
            y=df_tech["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#4a9eff", width=2),
        )
    )
    fig_bb.add_trace(
        go.Scatter(
            x=df_tech["Date"],
            y=df_tech["BB_lower"],
            mode="lines",
            name="Lower Band",
            line=dict(color="#00c896", dash="dash"),
            opacity=0.7,
            fill="tonexty",
            fillcolor="rgba(74,158,255,0.05)",
        )
    )
    fig_bb.update_layout(
        height=300,
        title="Bollinger Bands (20, 2σ)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bb, use_container_width=True)

    # Current values
    last = df_tech.iloc[-1]
    st.subheader("Current Values")
    c1, c2, c3, c4 = st.columns(4)
    rsi_val = last["RSI_14"]
    rsi_label = (
        "🔴 Overbought"
        if rsi_val > 70
        else "🟢 Oversold" if rsi_val < 30 else "⚪ Neutral"
    )
    c1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_label)
    c2.metric(
        "MACD Histogram",
        f"{last['MACD_histogram']:+.3f}",
        "Bullish" if last["MACD_histogram"] > 0 else "Bearish",
    )
    c3.metric(
        "BB Position",
        f"{last['BB_position']:.2f}",
        (
            "Near upper"
            if last["BB_position"] > 0.8
            else "Near lower" if last["BB_position"] < 0.2 else "Mid"
        ),
    )
    c4.metric("ATR (volatility)", f"{last['ATR_normalised']*100:.2f}%")

# ── SCREEN 5: NEWS FEED ───────────────────────────────────

elif screen == "News Feed":
    st.title(f"📰 {ticker} — Live News Feed")

    result = load_sentiment_pipeline(ticker, days=7)
    articles = result.get("articles", [])

    if not articles:
        st.info("No articles in cache. Click 'Refresh Data' in the sidebar.")
        st.stop()

    st.caption(f"{len(articles)} articles found for {ticker} (last 7 days)")

    filter_col1, filter_col2 = st.columns([1, 3])
    sentiment_filter = filter_col1.selectbox(
        "Filter by sentiment", ["All", "Positive", "Negative", "Neutral"]
    )

    if sentiment_filter != "All":
        articles = [
            a
            for a in articles
            if a.get("sentiment", "").lower() == sentiment_filter.lower()
        ]

    for a in articles:
        label = a.get("sentiment", "neutral")
        score = a.get("sentiment_score", 0) or 0
        conf = a.get("sentiment_confidence", 0) or 0

        bg = (
            "#0d2b1a"
            if label == "positive"
            else "#2b0d0d" if label == "negative" else "#1a1a1a"
        )
        emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "⚪"

        with st.container():
            st.markdown(
                f"""<div style='background:{bg}; border-radius:8px; padding:12px 16px; margin-bottom:8px;'>
                <b>{emoji} <a href="{a.get('url','#')}" target="_blank" style="color:white; text-decoration:none;">
                {a['title'][:100]}</a></b><br>
                <span style='font-size:12px; color:#aaa;'>
                {a.get('source','')} · {a.get('published_at','')[:10]} &nbsp;|&nbsp;
                {label.upper()} {score:+.3f} &nbsp;|&nbsp; conf: {conf:.2f}
                </span></div>""",
                unsafe_allow_html=True,
            )
