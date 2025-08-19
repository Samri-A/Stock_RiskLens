
# Stock RiskLens

Stock RiskLens is an end-to-end toolkit for stock market risk analysis and price prediction, combining financial data, technical indicators, news sentiment analytics, and deep learning. It features a data pipeline, model training, API, and dashboard for interactive prediction.

## Project Structure

- `app/`
  - `main.py`: FastAPI backend for stock price prediction
- `dashboard/`
  - `app.py`: Streamlit dashboard for user-friendly prediction interface
- `notebooks/`
  - `EDA.ipynb`: Exploratory data analysis
  - `feature_eng.ipynb`: Feature engineering and preprocessing
- `scripts/`
  - `yfinance_data_analysis.py`: Stock data analysis and visualization (technical indicators, risk metrics, merging with news)
  - `news_data_analysis.py`: News sentiment, topic, and publication analytics
  - `feature.py`: Feature engineering and visualization utilities
  - `preprocess.py`: Data preprocessing, encoding, and scaling
  - `forcast.py`: Model-based forecasting and plotting
- `src/`
  - `data/`: Raw and processed data storage
    - `raw/`: Input CSVs (e.g., `NVDA_historical_data.csv`, `raw_analyst_ratings.csv`)
    - `processed/`: Cleaned, merged, and feature-engineered datasets
  - `model/`: Trained model and encoders (e.g., `stock_predict_model.pkl`)
  - `train/`: Model training scripts (LSTM pipeline)
  - `predict/`: (Reserved for prediction utilities)
- `tests/`: Unit tests
- `requirements.txt`: Python dependencies

## Data Flow

1. **Raw Data**: Stock and news CSVs in `src/data/raw/`
2. **Analysis & Feature Engineering**: Performed in `notebooks/` and `scripts/`, results saved to `src/data/processed/`
3. **Preprocessing**: Scaling and encoding with `scripts/preprocess.py`, outputting cleaned data and encoders
4. **Model Training**: LSTM model trained with `src/train/train.py`, model saved to `src/model/`
5. **API & Dashboard**: FastAPI (`app/main.py`) serves predictions, Streamlit dashboard (`dashboard/app.py`) provides UI

## Features

- **Stock Data Analysis**: Technical indicators (SMA, EMA, RSI, MACD), risk metrics (Sharpe, VaR), visualization
- **News Data Analysis**: Sentiment scoring, topic classification, publication trends, publisher analytics
- **Feature Engineering**: Correlation, publisher/return analysis, topic/sentiment impact
- **Deep Learning Model**: LSTM-based price prediction using merged features
- **API**: FastAPI endpoint for real-time prediction
- **Dashboard**: Streamlit app for interactive prediction
- **Forecasting**: Multi-day prediction and plotting

## Example Usage

### 1. Data Analysis & Feature Engineering (Jupyter Notebooks)
See `notebooks/EDA.ipynb` and `notebooks/feature_eng.ipynb` for step-by-step data exploration and feature creation.

### 2. Model Training
```bash
python src/train/train.py
```

### 3. Run API
```bash
uvicorn app.main:app --reload
```

### 4. Run Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. API Example
POST to `/predict/` with JSON:
```json
{
  "topic": "Earnings Report",
  "sentiment_category": "positive",
  "close_price": 500.0
}
```

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, tensorflow, pydantic, fastapi, uvicorn
- textblob, nltk, talib, PyPortfolioOpt

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

- **Stock data CSV:** Columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **News data CSV:** Columns like `date`, `headline`, `publisher`, `stock`, etc.

