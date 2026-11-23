"""Volatility targeting: hold more when forecast risk is low, less when it is high.

The weight is decided at each close from that day's forecast and applied from the next day, so
there is no look-ahead. The uninvested part earns the risk-free rate. Trading costs are charged
on every change in weight.
"""

import numpy as np
import pandas as pd

from rv.perf import cagr, max_drawdown, sharpe
from rv.returns import TRADING_DAYS


def vol_target_weights(vol_forecast: pd.Series, target: float, cap: float = 1.0) -> pd.Series:
    """target / forecast vol, capped (cap 1.0 = never more than fully invested, no leverage)."""
    return (target / vol_forecast).clip(upper=cap)


def apply_weights(
    r: pd.Series, weights: pd.Series, rf: float = 0.0, cost: float = 0.0,
    start_weight: float = 1.0,
) -> pd.DataFrame:
    """Daily returns of holding `weights` (decided at each close) of the asset, rest in cash.

    `cost` is the fraction lost per unit of weight traded (0.001 = 0.1% of the amount traded).
    """
    held = weights.reindex(r.index).shift(1).fillna(start_weight)
    traded = held.diff().abs().fillna(0.0)
    ret = held * r + (1 - held) * rf / TRADING_DAYS - traded * cost
    return pd.DataFrame({"return": ret, "weight": held})


def static_match(r: pd.Series, used: pd.Series, rf: float = 0.0, cost: float = 0.0) -> pd.DataFrame:
    """A constant weight equal to the average of `used`: same average exposure, no timing.

    The fair benchmark. Scaling exposure up or down uniformly never changes the excess Sharpe
    ratio, so any Sharpe difference vs this comes from timing alone.
    """
    w = pd.Series(float(used.mean()), index=r.index)
    return apply_weights(r, w, rf, cost, start_weight=float(used.mean()))


def compare(strategies: dict[str, pd.DataFrame], rf: float = 0.0) -> pd.DataFrame:
    rows = {}
    for name, df in strategies.items():
        x = df["return"]
        roll = x.rolling(TRADING_DAYS).std().dropna() * np.sqrt(TRADING_DAYS)
        rows[name] = {
            "CAGR": cagr(x),
            "Vol": float(x.std() * np.sqrt(TRADING_DAYS)),
            "Sharpe": sharpe(x, rf),
            "Max DD": max_drawdown(x),
            "1y vol low": float(roll.min()) if len(roll) else np.nan,
            "1y vol high": float(roll.max()) if len(roll) else np.nan,
            "Avg weight": float(df["weight"].mean()),
            "Turnover / yr": float(df["weight"].diff().abs().sum() / (len(x) / TRADING_DAYS)),
        }
    return pd.DataFrame(rows).T
