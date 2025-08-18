import matplotlib.pyplot as plt
import pandas as pd 
import talib as tl
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import numpy as np


class FinanceAnalysis:
    """
    A class for processing, visualizing, and analyzing stock market data.
    """

    def __init__(self, path: str):
        try:
            self.df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading the CSV file: {e}")

        required_cols = {"Date", "Close"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain {required_cols}")

        # Convert date column
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        self.df.dropna(subset=["Date"], inplace=True)

        # Calculate daily percentage change
        self.df["Daily_Change"] = self.df["Close"].pct_change() * 100

    def calculateTechnicalIndicator(self):
        """Compute SMA, EMA, RSI, and MACD indicators."""
        self.df["SMA"] = tl.SMA(self.df["Close"], timeperiod=20)
        self.df["RSI"] = tl.RSI(self.df["Close"], timeperiod=14)
        self.df["EMA"] = tl.EMA(self.df["Close"], timeperiod=20)

        macd, macd_signal, _ = tl.MACD(self.df["Close"])
        self.df["MACD"] = macd
        self.df["MACD_Signal"] = macd_signal

        self.df.dropna(inplace=True)

    def analysisClosingPrice(self):
        plt.plot(self.df["Date"], self.df["Close"], label="Close")
        plt.title("Closing Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.show()

    def plotTechnicalIndicators(self):
        plt.plot(self.df["Date"], self.df["SMA"], label="SMA", color="blue")
        plt.plot(self.df["Date"], self.df["EMA"], label="EMA", color="orange")
        plt.title("Technical Indicators Over Time")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    def closingPriceRelativeStrengthIndex(self):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Date")
        ax1.set_ylabel("RSI", color="purple")
        ax1.plot(self.df["Date"], self.df["RSI"], label="RSI", color="purple")
        ax1.tick_params(axis="y", labelcolor="purple")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Close", color="blue")
        ax2.plot(self.df["Date"], self.df["Close"], label="Close", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        plt.title("RSI and Closing Price")
        plt.show()

    def closingPriceMovingAverageConvergenceDivergence(self):
        plt.plot(self.df["Date"], self.df["MACD"], label="MACD", color="green")
        plt.plot(self.df["Date"], self.df["MACD_Signal"], label="MACD Signal", color="red")
        plt.title("MACD and Signal Line")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    @staticmethod
    def getSharpe(returns: pd.Series, annual_rf: float = 0.01) -> float:
        """Calculate annualized Sharpe Ratio."""
        daily_rf = annual_rf / 252
        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std(ddof=0)

        if std_excess == 0:
            return np.nan 

        daily_sharpe = mean_excess / std_excess
        return np.sqrt(252) * daily_sharpe

    def FinancialCalculation(self):
        returns = self.df["Close"].pct_change().dropna()
        self.sharpe_ratio = self.getSharpe(returns, 0)
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")

        VaR_95 = self.df["Daily_Change"].quantile(0.05)
        print(f"Value at Risk (VaR) at 95% confidence level: {VaR_95:.2f}%")

    def merge_with_news_data(self, newspath: str) -> pd.DataFrame:
        """Merge stock data with news data on Date."""
        try:
            news_data = pd.read_csv(newspath)
        except Exception as e:
            raise ValueError(f"Error reading the news CSV: {e}")

        if "date" not in news_data.columns:
            raise ValueError("News CSV must contain a 'date' column")

        news_data["date"] = pd.to_datetime(news_data["date"], errors="coerce")
        news_data.dropna(subset=["date"], inplace=True)

        self.df["Date"] = pd.to_datetime(self.df["Date"]).dt.tz_localize(None)
        news_data.rename(columns={"date": "Date"}, inplace=True)
        news_data["Date"] = news_data["Date"].dt.tz_localize(None)

        return pd.merge(self.df, news_data, on="Date", how="inner")
