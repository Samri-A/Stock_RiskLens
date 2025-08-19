import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np                     
from xverse.ensemble import VotingSelector
import warnings                        
warnings.filterwarnings("ignore")

"""This file consists of plots needed to visualize the data relationships for stock market analysis"""


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.rename(columns={"Daily_Change": "Daily_Return"}, inplace=True)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None
    


def comparsion_plot(dataframe , X , Y):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=X, y=Y, data=dataframe, estimator='mean', ci=None)
    plt.title('Correlation between {} and {}'.format(X, Y))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[["Daily_Return", "sentiment_score"]].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


def corelation_plot(dataframe, X, Y):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=X, y=Y, data=dataframe)
    plt.title('Correlation between {} and {}'.format(X, Y))
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.tight_layout()
    plt.show()

def plot_publisher_vs_return(dataframe):
    plt.figure(figsize=(12, 18))
    sns.barplot(x='Daily_Return', y='publisher', data=dataframe, estimator='mean', ci=None)
    plt.title(f' Stock Return by Publisher')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()






