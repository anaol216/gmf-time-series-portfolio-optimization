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
def load_data_from_csv(ticker, data_dir='data/raw'):
    """
    Load historical stock data from a CSV file.
    """
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for {ticker} not found in {data_dir}.")
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return df