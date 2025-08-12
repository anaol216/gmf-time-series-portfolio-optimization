import pandas as pd

import pandas as pd

def align_and_merge(data_dict):
    """
    Align and merge 'Adj Close' columns from multiple tickers,
    even if CSVs have duplicate column names from Yahoo Finance.
    """
    merged = []
    for ticker, df in data_dict.items():
        # If duplicate columns exist (MultiIndex case)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                series = df[('Close', ticker)]
            except KeyError:
                # Fallback if the ticker is not in second level
                series = df.xs('Close', axis=1, level=0).iloc[:, 0]
        else:
            # If columns are flat, try to locate 'Adj Close'
            if 'Close' in df.columns:
                series = df['Close']
            else:
                # If no 'Adj Close', take last numeric column as fallback
                series = df.iloc[:, 4]
        merged.append(series.rename(ticker))
    return pd.concat(merged, axis=1).dropna()


def compute_daily_returns(df):
    """
    Compute daily returns from adjusted close prices.
    """
    daily_returns = df.pct_change().dropna()
    return daily_returns
