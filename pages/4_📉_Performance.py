import streamlit as st

st.title("📉 Model Performance")

st.subheader("Algorithm Comparison")

data = {
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "R² Score": [
        0.9999,
        -0.3431,
        -0.3467
    ],
    "MAE": [
        6.566,
        768.193,
        769.261
    ],
    "RMSE": [
        10.734,
        1172.827,
        1174.389
    ]
}

st.dataframe(data, use_container_width=True)

st.success("🏆 Best Model : Linear Regression")