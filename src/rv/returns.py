"""Daily log returns (see monte-carlo-stock-sim for log vs simple and annualising)."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """ln(P_t / P_{t-1}); the first row has no previous price and is dropped."""
    if (prices <= 0).any(axis=None):
        raise ValueError("prices must be positive for log returns")
    return np.log(prices / prices.shift(1)).iloc[1:]
