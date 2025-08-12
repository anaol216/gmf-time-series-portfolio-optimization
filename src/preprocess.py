import pandas as pd

def align_and_merge(data_dict):
    """
    Align and merge multiple DataFrames in a dictionary based on their indices.
    """
    merge= pd.concat([df["Adj Close"].rename(ticker) 
                    for ticker, df in data_dict.items()], axis=1)
    return merge.dropna()
def compute_daily_returns(df):
    """
    Compute daily returns from adjusted close prices.
    """
    daily_returns = df.pct_change().dropna()
    return daily_returns