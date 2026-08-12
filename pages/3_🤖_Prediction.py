
import streamlit as st
import yfinance as yf
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="🤖",
    layout="wide",
)

# =========================================================
# NSE STOCK LIST
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

FEATURES = ["Open", "High", "Low", "Volume", "Year", "Month", "Day"]

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.stApp {
    background: #06110d;
}

[data-testid="stSidebar"] {
    background: #07130f;
    border-right: 1px solid #183b2d;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
}

.title {
    font-size: 32px;
    font-weight: 800;
    color: #f2f7f5;
}

.subtitle {
    color: #7f918a;
    font-size: 13px;
    margin-bottom: 25px;
}

.card {
    background: linear-gradient(145deg,#0b2118,#07150f);
    border: 1px solid #174633;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
}

.card-title {
    font-size: 15px;
    font-weight: 800;
    color: #d9e6e1;
    margin-bottom: 12px;
}

.prediction {
    font-size: 42px;
    font-weight: 900;
    color: #25e69a;
}

.current {
    font-size: 28px;
    font-weight: 800;
    color: #f1f6f4;
}

.positive {
    color: #25e69a;
    font-weight: 800;
}

.negative {
    color: #ff6574;
    font-weight: 800;
}

.muted {
    color: #788b84;
    font-size: 12px;
}

.feature-box {
    background: #081812;
    border: 1px solid #163b2d;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stMetric"] {
    background: #081b13;
    border: 1px solid #163b2d;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADER
# =========================================================
@st.cache_resource
def load_model():
    candidates = [
        "stock_model.pkl",
        "stock_model_v2.pkl",
        "model.pkl",
        "random_forest_model.pkl",
    ]

    locations = []

    # Project root
    root = Path(__file__).resolve().parent.parent
    for filename in candidates:
        locations.append(root / filename)

    # Current working directory
    for filename in candidates:
        locations.append(Path.cwd() / filename)

    for path in locations:
        if path.exists():
            try:
                return joblib.load(path), path.name
            except Exception as e:
                return None, f"{path.name}: {e}"

    return None, "No model file found"


# =========================================================
# MARKET DATA
# =========================================================
@st.cache_data(ttl=60)
def get_latest_data(ticker):
    data = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        auto_adjust=False,
            progress=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🤖 AI PREDICTION")

company = st.sidebar.selectbox(
    "Select NSE Company",
    list(STOCKS.keys())
)

if st.sidebar.button("🔄 Refresh Prediction", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

ticker = STOCKS[company]

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="title">🤖 AI Stock Prediction</div>'
    '<div class="subtitle">'
    'NSE Market Data • Machine Learning Prediction • Next-Day Close'
    '</div>',
    unsafe_allow_html=True,
)

# =========================================================
# LOAD MODEL + DATA
# =========================================================
model, model_status = load_model()
data = get_latest_data(ticker)

if model is None:
    st.error("ML model could not be loaded.")
    st.info(
        "Put your saved model file in the main SMP PROJECT folder. "
        "Supported names: stock_model.pkl, stock_model_v2.pkl, model.pkl."
    )
    st.code(
        "SMP PROJECT/\n"
        "├── app.py\n"
        "├── stock_model.pkl   ← your trained model\n"
        "└── pages/\n"
        "    └── 3_🤖_Prediction.py"
    )
    st.stop()

if data.empty:
    st.error(f"Unable to retrieve market data for {company}.")
    st.stop()

latest = data.iloc[-1]

# =========================================================
# FEATURE CREATION - MATCH TRAINING MODEL
# =========================================================

# We need enough historical data to calculate MA30
data = get_latest_data(ticker)

if data.empty or len(data) < 30:
    st.error("Not enough historical data to calculate MA7 and MA30.")
    st.stop()

# Make sure data is sorted
data = data.sort_values("Date").reset_index(drop=True)

# Previous day's closing price
data["Previous_Close"] = data["Close"].shift(1)

# Daily return
data["Daily_Return"] = (
    (data["Close"] - data["Previous_Close"])
    / data["Previous_Close"]
)

# Moving averages
data["MA7"] = data["Close"].rolling(window=7).mean()
data["MA30"] = data["Close"].rolling(window=30).mean()

# Remove rows where indicators cannot be calculated
data = data.dropna().reset_index(drop=True)

# Latest available row
latest = data.iloc[-1]

# Date
date_value = pd.to_datetime(latest["Date"])

# =========================================================
# CREATE EXACT 11 MODEL FEATURES
# =========================================================

X = pd.DataFrame([{
    "Open": float(latest["Open"]),
    "High": float(latest["High"]),
    "Low": float(latest["Low"]),
    "Volume": float(latest["Volume"]),
    "Year": int(date_value.year),
    "Month": int(date_value.month),
    "Day": int(date_value.day),

    "Daily_Return": float(latest["Daily_Return"]),
    "MA30": float(latest["MA30"]),
    "MA7": float(latest["MA7"]),
    "Previous_Close": float(latest["Previous_Close"]),
}])

# EXACT ORDER USED BY YOUR MODEL
if hasattr(model, "feature_names_in_"):
    
    trained_features = list(model.feature_names_in_)

    st.write("Model expects:")
    st.write(trained_features)

    st.write("Dashboard provides:")
    st.write(list(X.columns))

    # Reorder X exactly as the model expects
    X = X.reindex(columns=trained_features)

else:
    st.error(
        "This saved model does not contain feature_names_in_. "
        "We need to check the original training code."
    )
    st.stop()
# =========================================================
# PREDICTION
# =========================================================

try:
    prediction = float(model.predict(X)[0])

except Exception as e:
    st.error("Prediction failed.")
    st.code(str(e))
    st.write("Features being sent to model:")
    st.write(X)
    st.stop()

current = float(latest["Close"])

difference = prediction - current

percentage = (
    difference / current * 100
    if current != 0
    else 0
)

direction = "UP" if difference >= 0 else "DOWN"

arrow = "↑" if difference >= 0 else "↓"

direction_class = (
    "positive"
    if difference >= 0
    else "negative"
)

# =========================================================
# MAIN PREDICTION CARD
# =========================================================
left, right = st.columns([1.8, 1], gap="large")

with left:
    st.markdown(
        '<div class="card">'
        '<div class="card-title">AI NEXT-DAY PREDICTION</div>'
        f'<div class="muted">{company} • NSE • {ticker}</div>'
        f'<div class="prediction">₹{prediction:,.2f}</div>'
        f'<div class="{direction_class}">'
        f'{arrow} {direction} &nbsp; ₹{abs(difference):,.2f} '
        f'({abs(percentage):.2f}%)'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        '<div class="card">'
        '<div class="card-title">CURRENT MARKET</div>'
        f'<div class="current">₹{current:,.2f}</div>'
        '<div class="muted">Latest available close</div>'
        '<br>'
        f'<div class="muted">MODEL: {model_status}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# =========================================================
# COMPARISON
# =========================================================
st.markdown(
    '<div class="card-title">PREDICTION SUMMARY</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Current Close", f"₹{current:,.2f}")
c2.metric("Predicted Close", f"₹{prediction:,.2f}")
c3.metric("Difference", f"₹{difference:+,.2f}")
c4.metric("Expected Change", f"{percentage:+.2f}%")

# =========================================================
# FEATURES SENT TO MODEL
# =========================================================
st.markdown(
    '<div class="card">'
    '<div class="card-title">MODEL INPUTS</div>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4, f5, f6, f7 = st.columns(7)

f1.metric("Open", f"₹{X['Open'].iloc[0]:,.2f}")
f2.metric("High", f"₹{X['High'].iloc[0]:,.2f}")
f3.metric("Low", f"₹{X['Low'].iloc[0]:,.2f}")
f4.metric("Volume", f"{X['Volume'].iloc[0]:,.0f}")
f5.metric("Year", str(X["Year"].iloc[0]))
f6.metric("Month", str(X["Month"].iloc[0]))
f7.metric("Day", str(X["Day"].iloc[0]))

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RECENT MARKET DATA
# =========================================================
st.markdown(
    '<div class="card">'
    '<div class="card-title">RECENT NSE DATA</div>',
    unsafe_allow_html=True,
)

display_cols = [
    col for col in ["Date", "Open", "High", "Low", "Close", "Volume"]
    if col in data.columns
]

recent = data[display_cols].tail(10).copy()

for col in ["Open", "High", "Low", "Close"]:
    if col in recent.columns:
        recent[col] = recent[col].round(2)

st.dataframe(
    recent,
    use_container_width=True,
    hide_index=True,
)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# IMPORTANT MODEL NOTE
# =========================================================
st.warning(
    "Important: Current ML model was trained on S&P 500 data. "
    "This page can technically feed NSE features into it, but the "
    "prediction is NOT an NSE-trained prediction. For meaningful NSE "
    "accuracy, the model should be retrained using NSE historical data "
    "with the same 7 features."
)

