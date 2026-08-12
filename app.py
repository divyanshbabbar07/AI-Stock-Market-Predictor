from PIL import Image
import streamlit as st

logo = Image.open(r"C:\Users\Divyansh Babbar\Downloads\SMP PROJECT\assets\assets\logo.png")

st.sidebar.image(logo, width=180)

st.sidebar.title("📈 AI PoweredStock MarketPrediction")

st.sidebar.success("Navigation")

st.sidebar.info(
"""
Developed by

Divyansh Babbar
"""
)


st.set_page_config(
    page_title="AI Powered Stock Market Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Powered Stock Market Prediction")

st.markdown("""
Welcome to the **AI powered Stock Market Prediction System**

👈 Use the sidebar to navigate through the application.

### Features
- 📊 Dashboard
- 📈 Data Analysis
- 🤖 AI Prediction
- 📉 Model Performance
- ℹ️ About Project
""")

st.success("Model Loaded Successfully ✅")

