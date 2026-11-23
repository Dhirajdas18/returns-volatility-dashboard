"""Volatility estimators. All return annualised volatility (a 0.20 means 20% a year)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rv.returns import TRADING_DAYS


def rolling_vol(returns: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Standard deviation of the last `window` daily returns, times sqrt(252).

    Every day in the window counts equally, and a day drops out all at once `window` days later.
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def ewma_vol(returns: pd.Series | pd.DataFrame, lam: float = 0.94):
    """Exponentially weighted volatility (RiskMetrics). Value on day t uses returns up to t.

    variance_t = lam * variance_{t-1} + (1 - lam) * r_t^2. Each day's weight shrinks by `lam`
    per day, so old shocks fade out gradually instead of dropping off a window edge.
    Mean return is taken as 0.
    """
    var = (returns**2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var * TRADING_DAYS)


@dataclass
class GarchFit:
    """GARCH(1,1): variance_{t+1} = omega + alpha * r_t^2 + beta * variance_t (daily %^2 units)."""

    omega: float
    alpha: float
    beta: float
    vol: pd.Series  # annualised one-step-ahead forecast made at each day's close

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        return float(np.sqrt(self.omega / (1 - self.persistence) * TRADING_DAYS) / 100)

    @property
    def half_life(self) -> float:
        """Days for a volatility shock's excess variance to halve."""
        return float(np.log(0.5) / np.log(self.persistence))

    def forecast(self, horizon: int, asof=None) -> pd.Series:
        """Annualised daily vol forecasts for the next `horizon` days, made at `asof`'s close."""
        start = self.vol.iloc[-1] if asof is None else self.vol.loc[asof]
        lr_var = self.omega / (1 - self.persistence)
        var1 = (start * 100) ** 2 / TRADING_DAYS
        k = np.arange(horizon)
        var = lr_var + self.persistence**k * (var1 - lr_var)
        return pd.Series(np.sqrt(var * TRADING_DAYS) / 100, index=k + 1, name="vol")

    def horizon_vol(self, horizon: int) -> pd.Series:
        """Annualised vol expected over the next `horizon` days, made at each day's close."""
        p = self.persistence
        lr_var = self.omega / (1 - p)
        var1 = (self.vol * 100) ** 2 / TRADING_DAYS
        avg = lr_var + (var1 - lr_var) * (1 - p**horizon) / ((1 - p) * horizon)
        return np.sqrt(avg * TRADING_DAYS) / 100

    def apply(self, returns: pd.Series) -> "GarchFit":
        """Run these fitted parameters over `returns` (e.g. a later test period) without refitting.

        Starts from the long-run variance; each day's forecast uses only returns up to that day.
        """
        pct = (returns * 100).to_numpy()
        var = np.empty(len(pct))
        v = self.omega / (1 - self.persistence)
        for t, x in enumerate(pct):
            v = self.omega + self.alpha * x**2 + self.beta * v
            var[t] = v
        vol = pd.Series(np.sqrt(var * TRADING_DAYS) / 100, index=returns.index, name=returns.name)
        return GarchFit(self.omega, self.alpha, self.beta, vol)


def garch_fit(returns: pd.Series) -> GarchFit:
    """Fit GARCH(1,1) with zero mean and normal errors by maximum likelihood (`arch` package)."""
    from arch import arch_model

    pct = returns * 100  # arch's optimiser is happiest with returns in percent
    res = arch_model(pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
        disp="off"
    )
    omega, alpha, beta = (float(res.params[k]) for k in ("omega", "alpha[1]", "beta[1]"))
    cond = res.conditional_volatility  # sigma_t, known at the close of t-1
    next_var = omega + alpha * pct**2 + beta * cond**2  # sigma_{t+1}, known at the close of t
    vol = np.sqrt(next_var * TRADING_DAYS) / 100
    return GarchFit(omega, alpha, beta, vol.rename(returns.name))


def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Range-based volatility from each day's high/low (Parkinson 1980).

    Uses the whole intraday path, not just the close, so it is less noisy for the same window.
    Ignores the overnight gap (open vs previous close), so it reads low when gaps are large.
    """
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)) * TRADING_DAYS)


def garman_klass(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Garman-Klass (1980): adds the open-to-close move to Parkinson. Ignores overnight gaps too."""
    daily = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    return np.sqrt(daily.rolling(window).mean() * TRADING_DAYS)


def yang_zhang(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """Yang-Zhang (2000): overnight-gap variance + open-to-close + Rogers-Satchell range term.

    Keeps most of the range estimators' low noise but no longer misses the overnight gap.
    """
    overnight = np.log(open_ / close.shift(1))
    intraday = np.log(close / open_)
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = (
        overnight.rolling(window).var()
        + k * intraday.rolling(window).var()
        + (1 - k) * rs.rolling(window).mean()
    )
    return np.sqrt(var * TRADING_DAYS)
