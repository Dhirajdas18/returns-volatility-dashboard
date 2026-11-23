"""How stocks move with the market (beta, correlation) and volatility regimes."""

import numpy as np
import pandas as pd

from rv.implied import rolling_percentile

REGIMES = ["calm", "normal", "stressed"]


def beta(r: pd.Series, m: pd.Series) -> float:
    """Sensitivity to the market: a beta of 1.3 moves 1.3% on average when the market moves 1%.

    beta = correlation x (stock vol / market vol), so a volatile stock can still have a low beta.
    """
    return float(r.cov(m) / m.var())


def rolling_beta(r: pd.Series, m: pd.Series, window: int = 252) -> pd.Series:
    return r.rolling(window).cov(m) / m.rolling(window).var()


def rolling_corr(r: pd.Series, m: pd.Series, window: int = 252) -> pd.Series:
    return r.rolling(window).corr(m)


def blume(b, weight: float = 2 / 3):
    """Shrink a measured beta towards 1 (Blume 1975): extreme estimates are partly noise."""
    return weight * b + (1 - weight)


def beta_forecast_errors(
    r: pd.Series, m: pd.Series, windows=(63, 126, 252, 504, 756), horizon: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """Mean absolute error when a beta measured over each look-back window is used as the forecast
    of the beta over the next `horizon` days. Checked every `step` days, raw and shrunk to 1.
    """
    x = pd.concat([r, m], axis=1, keys=["r", "m"]).dropna()
    rows = {}
    longest = max(windows)
    starts = range(longest, len(x) - horizon, step)  # same dates for every window: a fair race
    for w in windows:
        raw, shrunk = [], []
        for i in starts:
            past, future = x.iloc[i - w + 1:i + 1], x.iloc[i + 1:i + 1 + horizon]
            est, true = beta(past.r, past.m), beta(future.r, future.m)
            raw.append(abs(est - true))
            shrunk.append(abs(blume(est) - true))
        rows[w] = {"raw": float(np.mean(raw)), "shrunk to 1": float(np.mean(shrunk))}
    return pd.DataFrame(rows).T


def label_regimes(vol: pd.Series, window: int = 504) -> pd.Series:
    """calm / normal / stressed by where today's volatility ranks within the past `window` days
    (bottom, middle, top third). Uses only the past."""
    pct = rolling_percentile(vol, window)
    lab = pd.cut(pct, [0, 1 / 3, 2 / 3, 1.0], labels=REGIMES, include_lowest=True)
    return lab.astype(object).where(pct.notna())


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    """Chance of each regime tomorrow (columns) given today's regime (rows)."""
    lab = labels.dropna()
    t = pd.crosstab(lab.shift(1).iloc[1:], lab.iloc[1:], normalize="index")
    order = [g for g in REGIMES if g in t.index] or list(t.index)
    return t.reindex(index=order, columns=order).fillna(0.0)


def run_lengths(labels: pd.Series) -> dict[str, list[int]]:
    """Length in days of each unbroken stretch of every regime."""
    lab = labels.dropna().reset_index(drop=True)
    runs = (lab != lab.shift()).cumsum()
    out: dict[str, list[int]] = {}
    for _, g in lab.groupby(runs):
        out.setdefault(g.iloc[0], []).append(len(g))
    return out


def regime_corr(returns: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Average pairwise correlation between the columns, within each regime."""
    out = {}
    lab = labels.reindex(returns.index)
    k = returns.shape[1]
    for g in pd.unique(lab.dropna()):
        c = returns[lab == g].corr().to_numpy()
        out[g] = float(c[np.triu_indices(k, 1)].mean())
    return pd.Series(out)
