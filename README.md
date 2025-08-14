# GMF Time Series Portfolio Optimization

An end-to-end pipeline for data-driven investment strategies. This project elegantly fuses advanced time series forecasting with the principles of Modern Portfolio Theory to construct, optimize, and backtest a sophisticated portfolio.

-----

### Key Features

This project is a complete and modular financial analysis tool. Its architecture ensures both powerful functionality and easy extensibility.

  - **Data Ingestion & Preparation:** A robust process for loading and cleaning historical financial data, ensuring it's ready for high-fidelity modeling. We use **Pandas** to process daily price data for SPY, BND, and TSLA.
  - **ARIMA Forecasting:** We apply a statistical time series model, **ARIMA**, to forecast the future annualized return of a highly volatile asset, TSLA. This forward-looking insight is crucial for making informed investment decisions.
  - **Portfolio Optimization:** Leveraging the powerful **PyPortfolioOpt** library, we perform Mean-Variance Optimization (MVO) to calculate the most efficient asset allocation based on our forecast and historical data.
  - **Performance Visualization:** We generate insightful plots using **Matplotlib** and PyPortfolioOpt's built-in tools. These visualizations make complex data digestible, showing the optimal risk-return trade-offs and backtesting results.
  - **Strategy Backtesting:** Our custom backtesting engine simulates the performance of the optimized portfolio against a static **60/40 SPY/BND benchmark**. This is the ultimate test of the strategy's real-world viability.
  - **Modular Codebase:** The project's structure separates core logic into reusable Python scripts (`.py`), keeping the analysis notebook clean, focused, and reproducible.

-----

### Core Concepts

#### Modern Portfolio Theory (MPT) & The Efficient Frontier

MPT is an investment framework that seeks to maximize portfolio returns for a given level of risk. The **Efficient Frontier** is the cornerstone of this theory—a curve representing all the optimal portfolios that provide the highest expected return for a specific amount of risk. Our project identifies two key points on this curve:

  - **Maximum Sharpe Ratio Portfolio:** The portfolio that offers the best return for every unit of risk taken.
  - **Minimum Volatility Portfolio:** The portfolio that has the lowest possible risk.

#### ARIMA Forecasting

The **ARIMA (AutoRegressive Integrated Moving Average)** model is a popular statistical method for time series analysis. It is designed to capture the dependencies within a time series data set to predict future points. Our pipeline uses an ARIMA(1,1,1) model to project TSLA's future price, converting this forecast into an expected return that guides our portfolio optimization.

-----

### Project Structure

```
gmf-time-series-portfolio-optimization/
├── data/
│   └── merged_prices.csv           # A CSV file containing historical daily prices for SPY, BND, and TSLA.
├── notebooks/
│   └── task4_portfolio_optimization.ipynb # The central Jupyter Notebook for running the entire pipeline, from data loading to backtesting.
├── src/
│   ├── __init__.py                  # Marks the directory as a Python package.
│   ├── forecasting_analysis.py      # Contains the function for generating ARIMA forecasts.
│   └── portfolio_optimization.py    # Houses all the core functions for calculating portfolio stats, optimization, and backtesting.
└── README.md
```

-----

### Installation & Usage

To get started with this project, follow these simple steps.

1.  **Clone the Repository:**

    ```sh
    git clone https://github.com/your-username/gmf-time-series-portfolio-optimization.git
    cd gmf-time-series-portfolio-optimization
    ```

2.  **Install Dependencies:**
    Make sure you have `pip` installed. The project's dependencies are listed in `requirements.txt`.

    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the Analysis:**
    Launch the Jupyter Notebook to run the full pipeline.

    ```sh
    jupyter notebook notebooks/task4_portfolio_optimization.ipynb
    ```

    Simply execute the cells in the notebook sequentially to see the data processing, forecasting, optimization, and backtesting in action.

-----

### Results & Visualizations

Our project generates several key visualizations that provide deep insights into the strategy's performance.

#### 1\. Time Series Forecasting

This plot displays the **ARIMA forecast for TSLA's price** over the next year, including its confidence intervals. This visualization confirms the model's forward-looking projection and its associated uncertainty.

#### 2\. The Efficient Frontier

This chart, the heart of the optimization process, plots the Efficient Frontier. You can visually identify where the **Maximum Sharpe Ratio** and **Minimum Volatility** portfolios are located.

#### 3\. Backtest: Strategy vs. Benchmark

This powerful visualization compares the **cumulative returns** of our optimized portfolio against a simple 60/40 benchmark. The graph clearly shows the difference in performance over the backtesting period.

-----

### Future Improvements

This project serves as a strong foundation for more advanced work. Here are some potential extensions:

  - **Dynamic Rebalancing:** Enhance the backtest to simulate a real-world scenario where the portfolio is rebalanced at regular intervals (e.g., monthly or quarterly), adapting to new forecasts.
  - **Multivariate Forecasting:** Explore more sophisticated time series models like VAR (Vector Autoregression) or advanced deep learning models to capture the relationships and dependencies between all assets in the portfolio.
  - **Interactive Dashboard:** Build a real-time dashboard using a tool like Streamlit or Dash to allow users to interact with the model, update parameters, and visualize results on-the-fly.

-----

### Author

Authored by **Anaol Atinafu** | `atinafuanaol@gmail.com`
Organization: **10 Academy**
Date: August 14, 2025