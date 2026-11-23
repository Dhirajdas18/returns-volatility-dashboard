"""Risk-adjusted performance: Sharpe, Sortino, Calmar, and how uncertain a Sharpe ratio is.

Inputs are daily simple returns. `rf` is an annual risk-free rate (e.g. 0.065 for 6.5%).
"""

import numpy as np
import pandas as pd
from scipy import stats

from rv.returns import TRADING_DAYS


def _excess(r: pd.Series, rf: float) -> pd.Series:
    return r - rf / TRADING_DAYS


def sharpe(r: pd.Series, rf: float = 0.0) -> float:
    """Average excess return per unit of volatility, annualised."""
    x = _excess(r, rf)
    return float(x.mean() / x.std() * np.sqrt(TRADING_DAYS))


def sharpe_se(r: pd.Series, rf: float = 0.0) -> float:
    """Standard error of the annualised Sharpe ratio (Mertens 2002: allows for skew and fat tails).

    Rule of thumb: roughly 1 / sqrt(years). Ten years of data still leaves about +/-0.3.
    """
    x = _excess(r, rf).dropna()
    s = x.mean() / x.std()
    skew, kurt = stats.skew(x), stats.kurtosis(x, fisher=False)
    var = (1 + 0.5 * s**2 - skew * s + (kurt - 3) / 4 * s**2) / len(x)
    return float(np.sqrt(var * TRADING_DAYS))


def sortino(r: pd.Series, rf: float = 0.0) -> float:
    """Like Sharpe, but only days below the risk-free rate count as risk."""
    x = _excess(r, rf)
    downside = np.sqrt((np.minimum(x, 0) ** 2).mean())
    return float(x.mean() / downside * np.sqrt(TRADING_DAYS))


def cagr(r: pd.Series) -> float:
    """Compound annual growth rate. Uses calendar time when the index has dates (NSE data has
    about 246 rows a year, so assuming 252 would overstate growth); otherwise 252 rows a year."""
    growth = float((1 + r).prod())
    if isinstance(r.index, pd.DatetimeIndex) and len(r) > 1:
        years = (r.index[-1] - r.index[0]).days / 365.25
        return growth ** (1 / years) - 1
    return growth ** (TRADING_DAYS / len(r)) - 1


def max_drawdown(r: pd.Series) -> float:
    """Worst fall from a running peak, as a negative fraction (see monte-carlo-stock-sim)."""
    wealth = (1 + r).cumprod()
    peak = np.maximum(wealth.cummax(), 1.0)
    return float((wealth / peak - 1).min())


def calmar(r: pd.Series) -> float:
    """CAGR divided by the size of the worst drawdown."""
    return cagr(r) / abs(max_drawdown(r))


def rolling_sharpe(r: pd.Series, window: int = 252, rf: float = 0.0) -> pd.Series:
    x = _excess(r, rf)
    return x.rolling(window).mean() / x.rolling(window).std() * np.sqrt(TRADING_DAYS)


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Compounded return for each year (rows) and month 1-12 (columns), plus a 'Year' column."""
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    table = monthly.unstack().reindex(columns=range(1, 13))
    table["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    table.index.name = None
    return table


def summary(returns: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "CAGR": cagr(r),
            "Vol": float(r.std() * np.sqrt(TRADING_DAYS)),
            "Sharpe": sharpe(r, rf),
            "Sharpe SE": sharpe_se(r, rf),
            "Sortino": sortino(r, rf),
            "Max DD": max_drawdown(r),
            "Calmar": calmar(r),
        }
    return pd.DataFrame(rows).T
