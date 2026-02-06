import yfinance as yf
import pandas as pd
from datetime import datetime


def get_data(ticker: str = "AAPL", start: str = "2015-01-01", end: str | None = None):
    """
    Download historical price data from Yahoo Finance.
    
    Args:
        ticker: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format (defaults to today)
    
    Returns:
        tuple: (dates, prices) as numpy arrays
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    if isinstance(df.columns, pd.MultiIndex):
        prices_series = df.xs('Close', level='Price', axis=1)[ticker]
    else:
        prices_series = df['Close']

    prices = prices_series.values.flatten()
    prices = pd.Series(prices).ffill().bfill().values
    dates = df.index
    
    return dates, prices
