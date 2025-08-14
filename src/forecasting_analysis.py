import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from pmdarima.arima import ARIMA
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

def generate_arima_forecast(model, steps=252):
    """
    Generates an ARIMA forecast with confidence intervals.
    
    Args:
        model (pmdarima.ARIMA): The fitted ARIMA model.
        steps (int): The number of steps to forecast.
        
    Returns:
        tuple: A tuple containing the forecast (pd.Series) and confidence intervals (pd.DataFrame).
    """
    print(f"Generating ARIMA forecast for {steps} steps...")
    forecast_results = model.get_forecast(steps)
    forecast = forecast_results.predicted_mean
    confidence_interval = forecast_results.conf_int()
    confidence_interval.columns = ['lower', 'upper']
    return forecast, confidence_interval

def generate_lstm_forecast(model, train_scaled_data, scaler, steps=252, n_steps=60):
    """
    Generates an LSTM forecast using a walk-forward approach.
    
    Args:
        model (tensorflow.keras.Model): The trained LSTM model.
        train_scaled_data (np.array): The scaled training data.
        scaler (MinMaxScaler): The scaler used for the data.
        steps (int): The number of steps to forecast.
        n_steps (int): The number of time steps to look back.
        
    Returns:
        pd.Series: The inverse-transformed forecast values.
    """
    print(f"Generating LSTM forecast for {steps} steps...")
    forecast = []
    current_batch = train_scaled_data[-n_steps:].reshape(1, n_steps, 1)

    for i in range(steps):
        predicted_value = model.predict(current_batch, verbose=0)[0]
        forecast.append(predicted_value[0])
        current_batch = np.append(current_batch[:, 1:, :], [[predicted_value]], axis=1)

    forecast = np.array(forecast).reshape(-1, 1)
    return scaler.inverse_transform(forecast).flatten()

def plot_forecast_results(historical_data, forecast_data, confidence_interval=None, title='Stock Price Forecast'):
    """
    Visualizes the stock price forecast with historical data and confidence intervals.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(historical_data.index, historical_data.values, label='Historical Data', color='blue')
    plt.plot(forecast_data.index, forecast_data.values, label='Forecast', color='red')
    
    if confidence_interval is not None:
        plt.fill_between(
            confidence_interval.index,
            confidence_interval['lower'],
            confidence_interval['upper'],
            color='pink',
            alpha=0.5,
            label='Confidence Interval (95%)'
        )
        
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/final_forecast.png', dpi=300)
    plt.show()

def interpret_forecast_results(forecast_data, confidence_interval):
    """
    Analyzes the forecast and provides insights on trends, risks, and opportunities.
    """
    print("\n--- Forecast Interpretation ---")
    
    # 1. Trend Analysis
    start_price = forecast_data.iloc[0]
    end_price = forecast_data.iloc[-1]
    price_change = ((end_price - start_price) / start_price) * 100
    
    trend_direction = "upward" if price_change > 0 else "downward"
    print(f" Trend Analysis: The forecast shows a long-term {trend_direction} trend, with an expected change of {price_change:.2f}% over the forecast period.")
    
    # 2. Volatility and Risk
    if confidence_interval is not None:
        initial_ci_width = confidence_interval.iloc[0]['upper'] - confidence_interval.iloc[0]['lower']
        final_ci_width = confidence_interval.iloc[-1]['upper'] - confidence_interval.iloc[-1]['lower']
        ci_growth = (final_ci_width - initial_ci_width) / initial_ci_width * 100
        
        print(f" Volatility Analysis: The forecast's confidence interval widens significantly over time. Its width increases by {ci_growth:.2f}%, implying that the reliability and certainty of the prediction decrease as the forecast horizon extends.")
    
    # 3. Opportunities and Risks
    if price_change > 0:
        opportunity = "Potential opportunity for growth due to the expected upward trend."
        risk = "Risk of high volatility, as indicated by the widening confidence intervals, and the possibility of unforeseen market shocks."
    else:
        opportunity = "Potential shorting opportunity or a chance to enter the market at a lower price point later."
        risk = "Risk of further declines in value beyond the forecast, and the high uncertainty of the long-term prediction."
        
    print(f"\n Opportunities: {opportunity}")
    print(f" Risks: {risk}")