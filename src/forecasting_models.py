import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pmdarima import auto_arima
import warnings

# Suppress harmless warnings for cleaner output
warnings.filterwarnings("ignore")

# --- ARIMA Model Functions ---

def train_arima_model(train_series):
    """
    Trains an ARIMA model on the provided series using auto_arima to find optimal parameters.
    
    Args:
        train_series (pd.Series): The training data.
        
    Returns:
        The fitted ARIMA model.
    """
    print("Finding optimal ARIMA parameters with auto_arima...")
    model = auto_arima(train_series, 
                       seasonal=False, 
                       stepwise=True, 
                       suppress_warnings=True, 
                       trace=True)
    return model

def forecast_arima(model, steps):
    """
    Generates a forecast from a fitted ARIMA model.
    
    Args:
        model (ARIMA model): The fitted model.
        steps (int): The number of steps to forecast.
        
    Returns:
        pd.Series: The forecast values.
    """
    forecast = model.predict(n_periods=steps)
    return pd.Series(forecast, index=model.data.index[-steps:])

# --- LSTM Model Functions ---

def create_sequences(data, n_steps=60):
    """
    Creates sequences of data for LSTM training.
    
    Args:
        data (np.array): Scaled price data.
        n_steps (int): The number of time steps to look back.
    
    Returns:
        tuple: A tuple containing the feature (X) and target (y) sequences.
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, n_steps, epochs=50, batch_size=32):
    """
    Builds and trains a simple LSTM model.
    
    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        n_steps (int): The number of time steps.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        
    Returns:
        The trained Keras LSTM model.
    """
    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Training LSTM model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model

def forecast_lstm(model, train_data, scaler, steps=None, n_steps=60):
    """
    Generates a forecast from a trained LSTM model.
    
    Args:
        model (Keras model): The trained LSTM model.
        train_data (np.array): The scaled training data.
        scaler (MinMaxScaler): The scaler used for the data.
        steps (int): The number of steps to forecast.
        n_steps (int): The number of time steps to look back.
        
    Returns:
        np.array: The forecast values (inverse-transformed).
    """
    forecast = []
    current_batch = train_data[-n_steps:].reshape(1, n_steps, 1)

    for i in range(steps):
        # Predict the next value
        predicted_value = model.predict(current_batch)[0]
        forecast.append(predicted_value)
        
        # Update the sequence for the next prediction
        current_batch = np.append(current_batch[:, 1:, :], [[predicted_value]], axis=1)

    # Inverse transform the forecast to the original scale
    forecast = np.array(forecast).reshape(-1, 1)
    return scaler.inverse_transform(forecast)

# --- Evaluation Functions ---

def evaluate_forecast(y_true, y_pred):
    """
    Calculates and prints evaluation metrics for a forecast.
    
    Args:
        y_true (pd.Series or np.array): The actual values.
        y_pred (pd.Series or np.array): The predicted values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}%")
    return mae, rmse, mape