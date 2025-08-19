import streamlit as st

st.set_page_config(page_title="Stock RiskLens Dashboard", layout="wide")

st.header("Stock RiskLens Prediction Dashboard")

with st.form(key='prediction_form'):
    topic = st.text_input("News Topic")
    sentiment = st.selectbox("News Sentiment", ["Positive", "Negative", "Neutral"])
    close_price = st.number_input("Yesterday's Closing Price", min_value=0.0)

    submit_button = st.form_submit_button("Predict")

if submit_button:
    st.write("Prediction button clicked")