"""Out-of-sample volatility forecast comparison.

Each day's forecast uses only data up to that day's close. GARCH parameters are fitted on the
training period only, then run forward over the test period without refitting.
"""

import numpy as np
import pandas as pd

from rv.returns import TRADING_DAYS
from rv.vol import ewma_vol, garch_fit, rolling_vol, yang_zhang

HORIZON = 21  # one trading month


def realised_forward(returns: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Annualised volatility actually delivered over the next `horizon` days (t+1 .. t+horizon).

    Zero-mean, like the estimators. NaN for the last `horizon` days (the future isn't known yet).
    """
    future = (returns**2).rolling(horizon).sum().shift(-horizon)
    return np.sqrt(future / horizon * TRADING_DAYS)


def forecasts(
    returns: pd.Series,
    train_end,
    ohlc: pd.DataFrame | None = None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """Each estimator's annualised vol forecast for the next `horizon` days, made at each close."""
    train = returns[returns.index < train_end]
    out = {
        "20-day rolling": rolling_vol(returns, 20),
        "60-day rolling": rolling_vol(returns, 60),
        "252-day rolling": rolling_vol(returns, 252),
        "EWMA": ewma_vol(returns),
        "GARCH": garch_fit(train).apply(returns).horizon_vol(horizon),
        "Long-run (train)": pd.Series(train.std() * np.sqrt(TRADING_DAYS), index=returns.index),
    }
    if ohlc is not None:
        o = ohlc.reindex(returns.index)
        out["Yang-Zhang"] = yang_zhang(o["open"], o["high"], o["low"], o["close"], 20)
    return pd.DataFrame(out)


def qlike(real: pd.Series, forecast: pd.Series) -> float:
    """Mean QLIKE loss on variances: x - ln(x) - 1 with x = real/forecast. 0 = perfect.

    Punishes under-forecasting risk more than over-forecasting, and is robust to a noisy target
    (Patton 2011), which is why it is the standard score for volatility forecasts.
    """
    x = (real / forecast) ** 2
    return float((x - np.log(x) - 1).mean())


def evaluate(fc: pd.DataFrame, real: pd.Series, start, end=None) -> pd.DataFrame:
    """Score every forecast column against `real` on [start, end]. Best (lowest QLIKE) first.

    RMSE and bias are in annualised volatility points (0.05 = 5 points).
    """
    rows = {}
    for name in fc.columns:
        both = pd.concat([fc[name], real], axis=1, keys=["f", "r"]).dropna().loc[start:end]
        rows[name] = {
            "RMSE": float(np.sqrt(((both["f"] - both["r"]) ** 2).mean())),
            "QLIKE": qlike(both["r"], both["f"]),
            "bias": float((both["f"] - both["r"]).mean()),
        }
    return pd.DataFrame(rows).T.sort_values("QLIKE")


def scoreboard(table: pd.DataFrame, train_end, horizon: int = HORIZON) -> pd.DataFrame:
    """Test-period QLIKE for each estimator (rows) and symbol (columns), divided by that symbol's
    best score, so 1.00 = best and 1.25 = 25% worse. Rows sorted by average rank, best first.

    `table` is the (field, symbol) price table; OHLC fields are used for Yang-Zhang when present.
    """
    from rv.returns import log_returns

    fields = set(table.columns.get_level_values("field"))
    cols = {}
    for sym in table["adj_close"].columns:
        r = log_returns(table["adj_close"][sym].dropna())
        ohlc = None
        if {"open", "high", "low", "close"} <= fields:
            ohlc = pd.concat({f: table[f][sym] for f in ("open", "high", "low", "close")}, axis=1)
        fc = forecasts(r, train_end, ohlc, horizon)
        cols[sym] = evaluate(fc, realised_forward(r, horizon), start=train_end)["QLIKE"]
    q = pd.DataFrame(cols)
    rel = q / q.min()
    return rel.loc[q.rank().mean(axis=1).sort_values().index]
