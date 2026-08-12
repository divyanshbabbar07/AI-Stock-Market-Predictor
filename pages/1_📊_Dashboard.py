
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import textwrap

st.set_page_config(
    page_title="NSE AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# NSE WATCHLIST
# =========================================================
STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "ITC": "ITC.NS",
    "Wipro": "WIPRO.NS",
    "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
}

# =========================================================
# SAFE HTML RENDERER
# Prevents Streamlit from displaying HTML as code
# =========================================================
def html(content):
    st.markdown(
        textwrap.dedent(content).strip(),
        unsafe_allow_html=True
    )

# =========================================================
# CSS
# =========================================================
html("""
<style>
.stApp {
    background: #06110d;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #07130f;
    border-right: 1px solid #183b2d;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.brand {
    font-size: 32px;
    font-weight: 800;
    color: #f2f7f5;
}

.subtitle {
    color: #7f918a;
    font-size: 13px;
    margin-top: 3px;
    margin-bottom: 20px;
}

.ticker-strip {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding: 5px 0 18px 0;
}

.ticker {
    min-width: 165px;
    background: linear-gradient(145deg, #0d281e, #081812);
    border: 1px solid #174b38;
    border-radius: 14px;
    padding: 11px 14px;
    box-shadow: 0 0 20px rgba(0,255,160,.035);
}

.ticker-name {
    color: #b9c9c3;
    font-size: 12px;
    white-space: nowrap;
}

.ticker-price {
    color: #f3f7f5;
    font-size: 18px;
    font-weight: 800;
    margin-top: 4px;
}

.ticker-change {
    font-size: 12px;
    font-weight: 700;
    margin-top: 2px;
}

.positive {
    color: #26e69a !important;
}

.negative {
    color: #ff6574 !important;
}

.panel {
    background: linear-gradient(145deg, #0a2118, #07150f);
    border: 1px solid #174633;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 0 28px rgba(0,255,160,.025);
}

.panel-title {
    color: #d9e6e1;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: .3px;
    margin-bottom: 15px;
}

.stock-name {
    color: #f1f6f4;
    font-size: 23px;
    font-weight: 800;
}

.muted {
    color: #758a82;
    font-size: 12px;
}

.price {
    color: #f4f8f6;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.watch-row {
    display: grid;
    grid-template-columns: 1.7fr 1fr .8fr;
    gap: 4px;
    align-items: center;
    padding: 10px 2px;
    border-bottom: 1px solid #143127;
    font-size: 11px;
}

.watch-symbol {
    color: #c8d6d1;
    font-weight: 600;
}

.watch-price {
    text-align: right;
    color: #dce7e3;
}

.watch-change {
    text-align: right;
    font-weight: 700;
}

.detail-label {
    color: #6f847b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 14px;
}

.detail-value {
    color: #e8f0ed;
    font-size: 18px;
    font-weight: 700;
    margin-top: 3px;
}

.activity {
    background: linear-gradient(145deg, #081a13, #06120d);
    border: 1px solid #174633;
    border-radius: 18px;
    padding: 18px;
    margin-top: 18px;
}

.activity-row {
    padding: 9px 0;
    border-bottom: 1px solid #143127;
    color: #aebeb8;
    font-size: 12px;
}

.activity-row:last-child {
    border-bottom: none;
}

.activity-tag {
    color: #32e79c;
    font-weight: 800;
    margin-right: 10px;
}

[data-testid="stMetric"] {
    background: rgba(8, 27, 19, .75);
    border: 1px solid #163b2d;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stMetricLabel"] {
    color: #71857d;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid #1d5c43;
    background: #0b2b20;
}

.stButton > button:hover {
    border-color: #2ae69b;
    color: #2ae69b;
}
</style>
""")

# =========================================================
# DATA FUNCTIONS
# =========================================================
@st.cache_data(ttl=60)
def get_history(ticker, period="3mo", interval="1d"):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        if "Datetime" in data.columns:
            data["Date"] = data["Datetime"]
        elif "Date" not in data.columns:
            data["Date"] = pd.to_datetime(data.index)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        return data.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_quote(ticker):
    hist = get_history(ticker, "5d", "1d")

    if hist.empty:
        return None

    last = hist.iloc[-1]

    close = float(last["Close"])
    previous = (
        float(hist.iloc[-2]["Close"])
        if len(hist) > 1
        else float(last["Open"])
    )

    change = close - previous
    pct = (change / previous * 100) if previous else 0

    return {
        "open": float(last["Open"]),
        "high": float(last["High"]),
        "low": float(last["Low"]),
        "close": close,
        "volume": float(last["Volume"]) if pd.notna(last["Volume"]) else 0,
        "change": change,
        "pct": pct,
    }


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 📈 NSE AI")
st.sidebar.caption("Market Watch")

selected_company = st.sidebar.selectbox(
    "Select NSE Company",
    list(STOCKS.keys())
)

period = st.sidebar.selectbox(
    "Chart Period",
    ["1mo", "3mo", "6mo", "1y"],
    index=1
)

interval = st.sidebar.selectbox(
    "Chart Interval",
    ["1d", "1h"],
    index=0
)

if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

ticker = STOCKS[selected_company]

# =========================================================
# TOP TICKER STRIP
# =========================================================
ticker_cards = []

for name, symbol in list(STOCKS.items())[:6]:
    q = get_quote(symbol)

    if q is None:
        continue

    cls = "positive" if q["pct"] >= 0 else "negative"
    sign = "+" if q["pct"] >= 0 else ""

    ticker_cards.append(
        '<div class="ticker">'
        f'<div class="ticker-name">{name}</div>'
        f'<div class="ticker-price">₹{q["close"]:,.2f}</div>'
        f'<div class="ticker-change {cls}">{sign}{q["pct"]:.2f}%</div>'
        '</div>'
    )

ticker_strip = (
    '<div class="ticker-strip">'
    + "".join(ticker_cards)
    + "</div>"
)

st.markdown(ticker_strip, unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
html(f"""
<div class="brand">📈 AI Stock Market Predictor</div>
<div class="subtitle">
NSE India • Live/Latest Market Dashboard • AI/ML Prediction
</div>
""")

# =========================================================
# MAIN DATA
# =========================================================
history = get_history(ticker, period, interval)
quote = get_quote(ticker)

if history.empty or quote is None:
    st.error(
        f"Unable to load market data for {selected_company}. "
        "Try Refresh Market Data."
    )
    st.stop()

# =========================================================
# THREE COLUMN TERMINAL
# =========================================================
left, center, right = st.columns(
    [1.05, 2.45, 1.15],
    gap="medium"
)

# =========================================================
# LEFT - MARKET WATCH
# =========================================================
with left:
    watch_rows = []

    for name, symbol in STOCKS.items():
        q = get_quote(symbol)

        if q is None:
            continue

        cls = "positive" if q["pct"] >= 0 else "negative"
        sign = "+" if q["pct"] >= 0 else ""

        watch_rows.append(
            '<div class="watch-row">'
            f'<span class="watch-symbol">● {name[:18]}</span>'
            f'<span class="watch-price">₹{q["close"]:,.2f}</span>'
            f'<span class="watch-change {cls}">{sign}{q["pct"]:.2f}%</span>'
            '</div>'
        )

    watch_html = (
        '<div class="panel">'
        '<div class="panel-title">MARKET WATCH</div>'
        + "".join(watch_rows)
        + "</div>"
    )

    st.markdown(watch_html, unsafe_allow_html=True)

# =========================================================
# CENTER - MAIN CHART
# =========================================================
with center:

    change_cls = "positive" if quote["pct"] >= 0 else "negative"
    sign = "+" if quote["pct"] >= 0 else ""

    header_html = (
        '<div class="panel">'
        f'<div class="stock-name">{selected_company}</div>'
        f'<div class="muted">EXCH: NSE • {ticker}</div>'
        f'<div class="price">₹{quote["close"]:,.2f}</div>'
        f'<div class="{change_cls}">{sign}{quote["pct"]:.2f}%</div>'
        '</div>'
    )

    st.markdown(header_html, unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=history["Date"],
            open=history["Open"],
            high=history["High"],
            low=history["Low"],
            close=history["Close"],
            increasing_line_color="#20e69a",
            decreasing_line_color="#ff5f70",
            increasing_fillcolor="#20e69a",
            decreasing_fillcolor="#8d3a46",
            name=selected_company,
        )
    )

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#0a1b14",
        plot_bgcolor="#0a1b14",
        font=dict(color="#9dafaa"),
        xaxis=dict(
            showgrid=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#16352a",
            side="right",
        ),
        hovermode="x unified",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    a, b, c, d, e = st.columns(5)

    a.metric("OPEN", f"₹{quote['open']:,.2f}")
    b.metric("HIGH", f"₹{quote['high']:,.2f}")
    c.metric("LOW", f"₹{quote['low']:,.2f}")
    d.metric("CHANGE", f"{quote['pct']:+.2f}%")
    e.metric("VOLUME", f"{quote['volume']:,.0f}")

# =========================================================
# RIGHT - DETAILS
# =========================================================
with right:

    cls = "positive" if quote["pct"] >= 0 else "negative"
    sign = "+" if quote["pct"] >= 0 else ""

    details_html = (
        '<div class="panel">'
        '<div class="panel-title">DETAILS</div>'
        f'<div class="stock-name">{selected_company}</div>'
        f'<div class="muted">NSE • {ticker}</div>'

        '<div class="detail-label">Last Traded Price</div>'
        f'<div class="detail-value">₹{quote["close"]:,.2f}</div>'

        '<div class="detail-label">Change</div>'
        f'<div class="detail-value {cls}">{sign}{quote["pct"]:.2f}%</div>'

        '<div class="detail-label">Day High</div>'
        f'<div class="detail-value">₹{quote["high"]:,.2f}</div>'

        '<div class="detail-label">Day Low</div>'
        f'<div class="detail-value">₹{quote["low"]:,.2f}</div>'

        '<div class="detail-label">Open</div>'
        f'<div class="detail-value">₹{quote["open"]:,.2f}</div>'

        '<div class="detail-label">Volume</div>'
        f'<div class="detail-value">{quote["volume"]:,.0f}</div>'

        '</div>'
    )

    st.markdown(details_html, unsafe_allow_html=True)

    st.info("AI prediction is available from the Prediction page.")

# =========================================================
# ACTIVITY LOG
# =========================================================
now = datetime.now().strftime("%H:%M:%S")

activity_html = (
    '<div class="activity">'
    '<div class="panel-title">ACTIVITY LOG</div>'

    '<div class="activity-row">'
    f'<span class="activity-tag">{now}</span>'
    f'MARKET &nbsp; Live market data refreshed for {selected_company}'
    '</div>'

    '<div class="activity-row">'
    f'<span class="activity-tag">{now}</span>'
    f'DATA &nbsp; {len(history):,} candles loaded'
    '</div>'

    '<div class="activity-row">'
    f'<span class="activity-tag">{now}</span>'
    f'NSE &nbsp; {ticker} market data connected'
    '</div>'

    '<div class="activity-row">'
    f'<span class="activity-tag">{now}</span>'
    'SYSTEM &nbsp; AI Stock Market Predictor ready'
    '</div>'

    '</div>'
)

st.markdown(activity_html, unsafe_allow_html=True)

st.caption(
    "⚠️ Data is provided through the selected market-data provider. "
    "This dashboard is for educational purposes and is not financial advice."
)
