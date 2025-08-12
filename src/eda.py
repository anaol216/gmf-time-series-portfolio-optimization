import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()        