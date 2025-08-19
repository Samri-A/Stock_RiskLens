import fastapi
import pickle
from pydantic import BaseModel

app = fastapi.FastAPI( title="Stock Price Prediction API", description="API for predicting stock price from news data")
model = pickle.load(open("../src/model/stock_predict_model.pkl", "rb"))
encoder = pickle.load(open("../src/model/encoder.pkl", "rb"))
scaler = pickle.load(open("../src/model/scaler.pkl", "rb"))


class data(BaseModel):
    topic: str
    sentiment_category: str
    close_price: float


@app.post("/predict/")
def predict(data: data):
    topic = encoder.transform([data.topic])[0]
    sentiment_category = encoder.transform([data.sentiment_category])[0]
    close_price = scaler.transform([[data.close_price]])[0][0]
    processed_data = [topic, sentiment_category, close_price]
    prediction = model.predict([processed_data])
    return {"Predicted Price": prediction}

