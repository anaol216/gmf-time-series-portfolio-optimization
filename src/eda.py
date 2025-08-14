import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller

def plot_prices(df, save_path=None):
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title('Adjusted Close Prices')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    
def plot_correlation_heatmap(df, save_path=None):
    plt.figure(figsize=(6, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# -------------------- New Functions Added Below --------------------

def plot_rolling_volatility(df, window=21, save_path=None):
    """
    Plots the rolling standard deviation (volatility) of daily returns.
    """
    plt.figure(figsize=(14, 7))
    rolling_std = df.rolling(window=window).std()
    for column in rolling_std.columns:
        plt.plot(rolling_std.index, rolling_std[column], label=f'{column} Rolling Volatility')
    plt.title(f'{window}-Day Rolling Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Standard Deviation)')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def detect_outliers(returns_df, threshold=3):
    """
    Detects outliers in daily returns using the z-score method.
    
    Args:
        returns_df (pd.DataFrame): DataFrame containing daily returns for each ticker.
        threshold (int): The number of standard deviations to use as a threshold.
    Returns:
        pd.DataFrame: A DataFrame of identified outliers.
    """
    outliers = pd.DataFrame()
    for ticker in returns_df.columns:
        mean_return = returns_df[ticker].mean()
        std_return = returns_df[ticker].std()
        is_outlier = (returns_df[ticker] > mean_return + threshold * std_return) | \
                     (returns_df[ticker] < mean_return - threshold * std_return)
        ticker_outliers = returns_df[ticker][is_outlier]
        if not ticker_outliers.empty:
            outliers = pd.concat([outliers, ticker_outliers.to_frame(name=ticker)])
    return outliers

def perform_adf_test(series, ticker):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity.
    """
    print(f"--- ADF Test for {ticker} ---")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        print("Conclusion: The series is likely stationary (p-value <= 0.05).\n")
    else:
        print("Conclusion: The series is likely non-stationary (p-value > 0.05).\n")

def calculate_var(returns_df, confidence_level=0.95):
    """
    Calculates the historical Value at Risk (VaR) for each asset.
    """
    alpha = 1 - confidence_level
    var = returns_df.quantile(alpha)
    return var

def calculate_sharpe_ratio(returns_df, risk_free_rate=0.02, trading_days=252):
    """
    Calculates the annualized Sharpe Ratio for each asset.
    """
    avg_daily_return = returns_df.mean()
    std_daily_return = returns_df.std()
    annualized_return = avg_daily_return * trading_days
    annualized_std = std_daily_return * np.sqrt(trading_days)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
    return sharpe_ratio