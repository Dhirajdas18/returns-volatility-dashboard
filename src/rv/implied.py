"""Implied volatility (India VIX) vs realised volatility, and what high VIX says about returns."""

import numpy as np
import pandas as pd


def premium(implied: pd.Series, realised: pd.Series) -> pd.DataFrame:
    """Implied vol at t, realised vol over the following month, and the gap (implied - realised).

    A positive gap is the volatility risk premium: option buyers pay for more risk than shows up.
    """
    out = pd.concat([implied, realised], axis=1, keys=["implied", "realised"], sort=True).dropna()
    out["spread"] = out["implied"] - out["realised"]
    return out


def rolling_percentile(x: pd.Series, window: int = 504) -> pd.Series:
    """Where today's value ranks among the last `window` values, today included (1.0 = highest).

    Uses only the past, so it can drive a trading rule without look-ahead. 504 days = 2 years.
    """
    return x.rolling(window).apply(lambda w: (w <= w[-1]).mean(), raw=True)


def regime_returns(
    prices: pd.Series, pct: pd.Series, horizon: int = 21, threshold: float = 0.8, offset: int = 0
) -> pd.DataFrame:
    """Next-`horizon`-day log return after high-VIX days (pct > threshold) vs the rest.

    Samples every `horizon`-th day so the months don't overlap: each row is an independent month.
    `offset` picks which day of the month to start from; results that flip with it are noise.
    """
    fwd = np.log(prices.shift(-horizon) / prices)
    d = pd.concat([pct, fwd], axis=1, keys=["pct", "fwd"], sort=True).dropna().iloc[offset::horizon]
    d["regime"] = np.where(d["pct"] > threshold, "high", "normal")
    g = d.groupby("regime")["fwd"]
    out = pd.DataFrame({
        "months": g.size(),
        "mean return": g.mean(),
        "share positive": g.apply(lambda s: (s > 0).mean()),
        "worst month": g.min(),
        "monthly vol": g.std(),
    })
    out["return / vol"] = out["mean return"] / out["monthly vol"]
    return out
