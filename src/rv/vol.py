"""Volatility estimators. All return annualised volatility (a 0.20 means 20% a year)."""

import numpy as np
import pandas as pd

from rv.returns import TRADING_DAYS


def rolling_vol(returns: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Standard deviation of the last `window` daily returns, times sqrt(252).

    Every day in the window counts equally, and a day drops out all at once `window` days later.
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
