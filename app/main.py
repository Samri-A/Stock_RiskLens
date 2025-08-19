import fastapi
import pickle
from sklearn.preprocessing import LabelEncoder , MinMaxScaler

app = fastapi.FastAPI( title="Stock Price Prediction API", description="API for predicting stock price from news data")
model = pickle.load(open("model.pkl", "rb"))
encoder = LabelEncoder()
scaler = MinMaxScaler()

@app.post("/predict/")
def predict(topic , sentiment_category):
    topic = encoder.fit_transform([topic])
    sentiment_category = encoder.fit_transform([sentiment_category])
    close_price = scaler.fit_transform([[close_price]])
    processed_data = [topic, sentiment_category, close_price]
    prediction = model.predict([processed_data])
    return {"Predicted Price": prediction}

