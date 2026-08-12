import streamlit as st
from google import genai

st.set_page_config(
    page_title="AI Stock Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Stock Assistant")
st.caption("AI-powered assistant for your Stock Market Prediction system")

# Get API key securely
api_key = st.secrets["GEMINI_API_KEY"]

# Create Gemini client
client = genai.Client(api_key=api_key)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input(
    "Ask about stocks, ML predictions, indicators..."
)

if prompt:
    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # System instruction
    system_prompt = """
    You are an AI Stock Market Assistant for a college AI/ML project.

    Help users understand:
    - Stock markets
    - NSE and S&P 500
    - Stock price prediction
    - Machine learning
    - Linear Regression
    - Random Forest
    - Technical indicators
    - Moving averages
    - MAE, RMSE, R2 and MAPE

    Give educational explanations.
    Do not provide guaranteed investment advice.
    """

    try:
        response = client.models.generate_content(
        model="gemini-3.6-flash",
            contents=system_prompt + "\n\nUser question: " + prompt
        )

        answer = response.text

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as e:
        st.error(f"Unable to connect to Gemini API: {e}")