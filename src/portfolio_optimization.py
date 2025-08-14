import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import plotting # Add this import
def calculate_portfolio_stats(df, forecast_return_tsla=None):
    """
    Calculates historical returns, covariance matrix, and updates TSLA's expected return with the forecast.

    Args:
        df (pd.DataFrame): DataFrame of historical asset prices.
        forecast_return_tsla (float, optional): Forecasted annualized return for TSLA.

    Returns:
        tuple: A tuple containing expected returns (pd.Series), and covariance matrix (pd.DataFrame).
    """
    # Calculate daily returns
    returns = df.pct_change().dropna()

    # Calculate expected returns (annualized) and covariance matrix
    mu = expected_returns.mean_historical_return(df)
    sigma = risk_models.sample_cov(df)

    # Use the forecasted TSLA return if provided
    if forecast_return_tsla is not None:
        mu['TSLA'] = forecast_return_tsla

    return mu, sigma

def optimize_portfolio(mu, sigma, risk_free_rate=0.02):
    """
    Finds the Minimum Volatility and Maximum Sharpe Ratio portfolios and their performance.
    """
    # Create the first Efficient Frontier object for max Sharpe
    ef_max_sharpe = EfficientFrontier(mu, sigma)
    ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights_max_sharpe = ef_max_sharpe.clean_weights()
    max_sharpe_performance = ef_max_sharpe.portfolio_performance(verbose=False)

    # Create a second Efficient Frontier object for min volatility
    ef_min_vol = EfficientFrontier(mu, sigma)
    ef_min_vol.min_volatility()
    cleaned_weights_min_vol = ef_min_vol.clean_weights()
    min_vol_performance = ef_min_vol.portfolio_performance(verbose=False)

    return ef_max_sharpe, cleaned_weights_max_sharpe, max_sharpe_performance, cleaned_weights_min_vol, min_vol_performance

def plot_efficient_frontier(ef, max_sharpe_weights, min_vol_weights, max_sharpe_performance, min_vol_performance):
    """
    Plots the Efficient Frontier with key portfolios marked using their performance data.
    """
    max_sharpe_ret, max_sharpe_vol, _ = max_sharpe_performance
    min_vol_ret, min_vol_vol, _ = min_vol_performance
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the efficient frontier curve using pypfopt's built-in plotting function
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

    # Plot the Minimum Volatility portfolio
    ax.scatter(min_vol_vol, min_vol_ret, marker='o', s=100, c='b', label='Minimum Volatility')

    # Plot the Maximum Sharpe Ratio portfolio
    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', s=200, c='r', label='Maximum Sharpe Ratio')
    
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Volatility (Risk)')
    plt.ylabel('Portfolio Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/efficient_frontier.png')
    plt.show()

def get_portfolio_performance(weights, mu, sigma, risk_free_rate=0.02):
    """
    Calculates the expected return, volatility, and Sharpe Ratio for a given set of weights.
    
    Args:
        weights (dict): A dictionary of asset weights.
        mu (pd.Series): Expected returns.
        sigma (pd.DataFrame): Covariance matrix.
        risk_free_rate (float): The risk-free rate.
        
    Returns:
        tuple: (Expected Return, Volatility, Sharpe Ratio).
    """
    portfolio_return = np.dot(np.array(list(weights.values())), mu)
    portfolio_volatility = np.sqrt(np.dot(np.array(list(weights.values())).T, np.dot(sigma, np.array(list(weights.values())))))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def backtest_strategy(df, strategy_weights, benchmark_weights, start_date, end_date):
    """
    Performs a backtest of the portfolio strategy against a benchmark.
    """
    print("\n--- 1. Backtesting Strategy vs. Benchmark ---")
    backtest_data = df.loc[start_date:end_date]
    if backtest_data.empty:
        print(f"No data available for the period {start_date} to {end_date}.")
        return

    backtest_returns = backtest_data.pct_change().dropna()
    risk_free_rate = 0.02

    strategy_weights_series = pd.Series(strategy_weights).reindex(backtest_returns.columns, fill_value=0)
    benchmark_weights_series = pd.Series(benchmark_weights).reindex(backtest_returns.columns, fill_value=0)

    strategy_portfolio_returns = backtest_returns.dot(strategy_weights_series)
    benchmark_portfolio_returns = backtest_returns.dot(benchmark_weights_series)

    strategy_cumulative_returns = (1 + strategy_portfolio_returns).cumprod()
    benchmark_cumulative_returns = (1 + benchmark_portfolio_returns).cumprod()

    plt.figure(figsize=(12, 8))
    plt.plot(strategy_cumulative_returns, label="Optimized Portfolio Strategy")
    plt.plot(benchmark_cumulative_returns, label="60/40 SPY/BND Benchmark")
    plt.title("Backtest of Portfolio Strategy vs. Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    strategy_total_return = strategy_cumulative_returns.iloc[-1] - 1
    benchmark_total_return = benchmark_cumulative_returns.iloc[-1] - 1

    num_trading_days = len(strategy_portfolio_returns)
    if num_trading_days > 0:
        strategy_annualized_return = (1 + strategy_portfolio_returns.mean())**252 - 1
        strategy_annualized_volatility = strategy_portfolio_returns.std() * np.sqrt(252)
        strategy_sharpe = (strategy_annualized_return - risk_free_rate) / strategy_annualized_volatility if strategy_annualized_volatility > 0 else 0

        benchmark_annualized_return = (1 + benchmark_portfolio_returns.mean())**252 - 1
        benchmark_annualized_volatility = benchmark_portfolio_returns.std() * np.sqrt(252)
        benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_annualized_volatility if benchmark_annualized_volatility > 0 else 0
    else:
        strategy_annualized_return, strategy_annualized_volatility, strategy_sharpe = 0, 0, 0
        benchmark_annualized_return, benchmark_annualized_volatility, benchmark_sharpe = 0, 0, 0

    print("\n### Backtest Results ###")
    print("-" * 30)
    print("Final Performance (Backtest Period):")
    print(f"Strategy Total Return: {strategy_total_return:.2%}")
    print(f"Benchmark Total Return: {benchmark_total_return:.2%}")
    print("\nRisk-Adjusted Performance (Annualized):")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.2f}")
