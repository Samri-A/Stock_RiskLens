import mlflow
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import pickle

mlflow.tensorflow.autolog()

df = pd.read_csv('src/data/processed/cleaned_data.csv')

X = df[["publisher", "topic", "sentiment_category"]]
y = df['Close']

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
batch_size = 32
model = keras.Sequential([
    layers.Dense(units=1 , input_shape=[X.shape[1]])
])
model.compile(optimizer='adam', loss='mean_squared_error' )

with mlflow.start_run():
    model.fit(X_train, y_train, epochs=120 , batch_size=batch_size)
    y_pred = model.predict(X_test)
    mlflow.log_metric("test_loss", model.evaluate(X_test, y_test))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_r2", r2)


with open('src/model/stock_predict_model.pkl', 'wb') as f:
    pickle.dump(model, f)