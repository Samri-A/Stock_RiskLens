# Stock RiskLens

A toolkit for stock market risk analysis, combining financial data, technical indicators, and news sentiment analytics. Designed for data-driven investors, analysts, and researchers.

## Project Structure

- `scripts/`
  - `yfinance_data_analysis.py`: Stock data analysis and visualization (technical indicators, risk metrics, etc.)
  - `news_data_analysis.py`: News headline/article sentiment, topic, and publication analytics
- `notebooks/`: Jupyter notebooks for EDA and feature engineering
- `src/data/`: Data storage (raw and processed)
- `tests/`: Unit tests
- `requirements.txt`: Python dependencies

## Features

### Stock Data Analysis (`scripts/yfinance_data_analysis.py`)
- Load and preprocess historical stock data (CSV)
- Calculate technical indicators: SMA, EMA, RSI, MACD (using TA-Lib)
- Visualize price, indicators, and trends
- Compute Sharpe Ratio and Value at Risk (VaR)
- Merge stock data with news data for combined analysis

### News Data Analysis (`scripts/news_data_analysis.py`)
- Load and preprocess financial news data (CSV)
- Sentiment analysis of headlines (TextBlob)
- Topic classification using keyword matching
- Publication time series (hour, day, month, year)
- Publisher and organization analytics
- Save processed data for downstream tasks

## Example Usage

### Stock Analysis
```python
from scripts.yfinance_data_analysis import FinanceAnalysis
fa = FinanceAnalysis('src/data/raw/NVDA_historical_data.csv')
fa.calculateTechnicalIndicator()
fa.analysisClosingPrice()
fa.FinancialCalculation()
```

### News Analysis
```python
from scripts.news_data_analysis import NewsDataAnalysis
na = NewsDataAnalysis('src/data/raw/news_data.csv')
na.sentiment_analysis()
na.topic_analysis()
na.publication_timeseries_analysis()
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- talib
- pypfopt
- textblob
- nltk

Install dependencies:
```bash
pip install -r requirements.txt
```

For NLTK, the following resources are downloaded at runtime:
- punkt
- stopwords

TA-Lib may require system-specific installation steps. See [TA-Lib installation guide](https://mrjbq7.github.io/ta-lib/install.html).

## Data Format

- **Stock data CSV:** Must include columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- **News data CSV:** Should include `date`, `headline`, `publisher`, `stock`, etc.

## Notebooks

- `notebooks/EDA.ipynb`: Exploratory data analysis
- `notebooks/feature_eng.ipynb`: Feature engineering for modeling

## Testing

Unit tests are in the `tests/` directory. Run with your preferred test runner.

## License

MIT License (see LICENSE file if present)

## Authors

- Samri-A (see repository for contributors)
