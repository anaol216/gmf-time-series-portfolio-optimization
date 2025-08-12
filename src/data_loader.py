import  os 
import pandas as pd
import yfinance as yf

def download_data(tickers, start_date, end_date, save_dir='data/raw'):
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters:
    tickers (list): List of stock tickers to download.
    start_date (str): Start date for the data in 'YYYY-MM-DD' format.
    end_date (str): End date for the data in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: DataFrame containing the historical stock data.
    """
    os.makedirs(save_dir, exist_ok=True)
    data={}
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        file_path = os.path.join(save_dir, f"{ticker}.csv")
        df.to_csv(file_path)
        data[ticker] = df
    print("Data download complete.")
    return data

def load_data_from_csv(filepath):
    """
    Load CSV data and ensure columns are named consistently.
    Handles files where ticker names are repeated for each OHLCV field.
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # If duplicate column names exist, try to rename them
    if df.columns.duplicated().any():
        # Assign proper names assuming standard Yahoo format
        col_names = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        ticker = os.path.splitext(os.path.basename(filepath))[0]  # file name as ticker
        df.columns = pd.MultiIndex.from_product([[col_names[i] for i in range(len(df.columns)//len(col_names))], col_names])

    return df
