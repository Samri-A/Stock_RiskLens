import streamlit as st
import requests

port = 8000

st.set_page_config(page_title="Stock RiskLens Dashboard", layout="wide")

st.header("Stock RiskLens Prediction Dashboard")

with st.form(key='prediction_form'):
    topic = st.selectbox("News Topic" , ["FDA Approval" , "Price Target" , "Earnings Report" , "Stock Movement" , "Mergers & Acquisitions" , "Product Launch" , "Legal or Regulatory" , "Partnerships / Collaborations" , "Operations / Production" ,"Other"])
    sentiment = st.selectbox("News Sentiment", ["Positive", "Negative", "Neutral"])
    close_price = st.number_input("Yesterday's Closing Price", min_value=0.0)
    submit_button = st.form_submit_button("Predict")


if submit_button:
    try:
        response = requests.post(f"http://localhost:{port}/predict/", json={
            "topic": topic,
            "sentiment_category": sentiment,
            "close_price": close_price
        })
        if response.status_code == 200:
             prediction = response.json().get("Predicted Price")
             st.write(f"Predicted Price: {prediction}")
        else:
             st.write(f"Error occurred while fetching prediction.")
             st.write(f"Status Code: {response.status_code}, Message: {response.text}")
    except Exception as e:
        st.write(f"An error occurred: {e}")