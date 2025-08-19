from sklearn.preprocessing import LabelEncoder, MinMaxScaler  
import pickle
def preprocess_data(df):
    scaler = MinMaxScaler()
    encoder = LabelEncoder()
    df.fillna(method='ffill', inplace=True)
    df.drop_duplicates(inplace=True)
    df["Close"] = scaler.fit_transform(df[["Close"]])
    df["publisher"] = encoder.fit_transform(df["publisher"])
    df["topic"] = encoder.fit_transform(df["topic"])
    with open('../src/model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('../src/model/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    df["sentiment_category"] = encoder.fit_transform(df["sentiment_category"])
    return df

