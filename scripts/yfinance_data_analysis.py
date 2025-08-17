import matplotlib.pyplot as plt
import pandas as pd 
import talib as tl
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models , expected_returns
import numpy as np

class FinanceAnalysis:
    """
    A class for  processing, visualizing, and analyzing stock market data.
    other than this the code speak it self
    """
    def __init__(self, path):
        try:
            self.df  = pd.read_csv(path)
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
        self.df["Date"] = pd.to_datetime(self.df["Date"], format='ISO8601')
        self.df["Daily_Change"] = self.df['Close'].pct_change() * 100 
    
    def calculateTechnicalIndicator(self):
        self.df['SMA'] = tl.SMA(self.df['Close'], timeperiod=20)
        self.df['RSI'] = tl.RSI(self.df['Close'], timeperiod=14)
        self.df['EMA'] = tl.EMA(self.df['Close'], timeperiod=20)

        macd_signal ,macd , _=  tl.MACD(self.df['Close'])
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = macd_signal
        self.df.dropna(inplace=True)
    
    def analysisClosingPrice(self):
        plt.plot(self.df['Date'], self.df['Close'])
        plt.title('Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.show()

    def plotTechnicalIndicators(self):
        plt.plot(self.df['Date'], self.df['SMA'], label='SMA', color='blue')
        plt.plot(self.df['Date'], self.df['EMA'], label='EMA', color='orange')
        plt.title('Technical Indicators Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def closingPriceRelativeStrengthIndex(self):
        plt.plot(self.df['Date'], self.df['RSI'], label='RSI', color='purple')
        plt.plot(self.df["Date"] , self.df['Close'], label='Close', color='blue')
        plt.title('RSI and Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    # A MACD crossover (when the signal line crosses the MACD line) can indicate a potential trend change.
    def closingPriceMovingAverageConvergenceDivergence(self):
        plt.plot(self.df['Date'], self.df['MACD'], label='MACD', color='green')
        plt.plot(self.df['Date'], self.df['MACD_Signal'], label='MACD Signal', color='red')
        plt.title('MACD and Signal Line')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    

    
    @staticmethod
    def getSharpe(returns , annual_rf = 0.01):         
        daily_rf  = annual_rf / 252
        excess_returns = returns - daily_rf 
        mean_excess = excess_returns.mean()       
        std_excess  = excess_returns.std(ddof=0)  
        
        daily_sharpe = mean_excess / std_excess
        annual_sharpe = np.sqrt(252) * daily_sharpe
        return annual_sharpe
    
    def FinancialCalculation(self):
        returns = self.df['Close'].pct_change().dropna()
        self.sharpe_ratio = self.getSharpe(returns , 0)
        print(f'Sharpe Ratio: {self.sharpe_ratio:.2f}')
        VaR_95 = self.df["Daily_Change"].quantile(0.05)
        print(f'Value at Risk (VaR) at 95% confidence level: {VaR_95:.2f}%')

    def merge_with_news_data(self , newspath):
        news_data = pd.read_csv(newspath)
        news_data['date'] = pd.to_datetime(news_data['date'], format='ISO8601')
        news_data.rename(columns={'date': 'Date'}, inplace=True)
        merged_data = self.df.merge(news_data, how='inner', on='Date')
        return merged_data



