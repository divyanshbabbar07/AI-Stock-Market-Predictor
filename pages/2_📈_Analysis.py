import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Stock Market Analysis")

df = pd.read_csv("sp500.csv")

# Correlation Matrix
st.subheader("🔥 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Viridis"
)

st.plotly_chart(fig, use_container_width=True)

# Moving Average
st.subheader("📉 30-Day Moving Average")

df["MA30"] = df["Close"].rolling(30).mean()

fig2 = px.line(
    df,
    x="Date",
    y=["Close", "MA30"]
)

st.plotly_chart(fig2, use_container_width=True)