from tensorflow.keras import  optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM , Dense
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
import pickle
import pandas as pd
import mlflow
import numpy as np

mlflow.tensorflow.autolog()

df = pd.read_csv('src/data/processed/cleaned_data.csv')

Data = df["Close"]

def Create_Sequences(data, window=60):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data.iloc[i:i+window])
        y.append(data.iloc[i+window])
    return np.array(X), np.array(y)

X, y = Create_Sequences(Data, window=60)

train_length = int(len(X) * 0.8)
X_train, y_train = X[:train_length], y[:train_length]
X_test, y_test = X[train_length:], y[train_length:]

epochs = 100
batch = 32
lr = 0.0001
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=( 60, 1), return_sequences=True))
lstm_model.add(LSTM(25, activation='tanh', return_sequences=True))
lstm_model.add(LSTM(10, activation='tanh'))
lstm_model.add(Dense(1))
lstm_model.summary()


with mlflow.start_run():
    lstm_model.compile(loss=MeanSquaredError(), optimizer=optimizers.Adam(learning_rate=lr), metrics=[RootMeanSquaredError()])
    lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_split=0.1 , verbose=1 )
    eval_loss, eval_rmse = lstm_model.evaluate(X_test, y_test, verbose=1)
    y_pred = lstm_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mlflow.log_metric("eval_loss", eval_loss)
    mlflow.log_metric("eval_rmse", eval_rmse)
    mlflow.log_metric("mae", mae)

