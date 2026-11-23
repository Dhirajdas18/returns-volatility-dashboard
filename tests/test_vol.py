import numpy as np
import pandas as pd
import pytest

from rv.returns import log_returns
from rv.vol import rolling_vol


def _gbm_returns(sigma=0.2, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(rng.normal(0, sigma / np.sqrt(252), n), index=idx)


def test_log_returns_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        log_returns(pd.Series([1.0, 0.0, 2.0]))


def test_log_returns_first_row_dropped():
    out = log_returns(pd.Series([100.0, 110.0, 99.0]))
    assert np.allclose(out, np.log([1.1, 0.9]))


def test_rolling_vol_recovers_constant_sigma_annualised():
    vol = rolling_vol(_gbm_returns(0.2), 252)
    assert vol.iloc[:251].isna().all()
    assert abs(vol.dropna().mean() - 0.2) < 0.005


def test_rolling_vol_short_window_is_noisier():
    r = _gbm_returns(0.2)
    assert rolling_vol(r, 20).std() > 2 * rolling_vol(r, 252).std()


def _intraday_ohlc(sigma=0.2, days=1000, steps=390, seed=1):
    """GBM sampled `steps` times a day, no overnight gap: open = previous close."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, sigma / np.sqrt(252 * steps), (days, steps))
    logp = np.cumsum(z.ravel()).reshape(days, steps)
    opens = np.concatenate([[0.0], logp[:-1, -1]])
    path = np.column_stack([opens, logp])
    idx = pd.bdate_range("2010-01-04", periods=days)
    return pd.DataFrame({"open": np.exp(opens), "high": np.exp(path.max(axis=1)),
                         "low": np.exp(path.min(axis=1)), "close": np.exp(logp[:, -1])}, index=idx)


def test_ewma_recovers_constant_sigma():
    from rv.vol import ewma_vol

    vol = ewma_vol(_gbm_returns(0.2), lam=0.94)
    assert abs(vol.iloc[100:].mean() - 0.2) < 0.01


def test_ewma_has_no_ghost_cliff_after_a_shock():
    from rv.vol import ewma_vol

    r = _gbm_returns(0.2, n=400)
    r.iloc[200] = -0.12
    after = slice(201, 260)
    ewma_drop = -ewma_vol(r).diff().iloc[after].min()
    rolling_drop = -rolling_vol(r, 20).diff().iloc[after].min()
    assert ewma_drop < rolling_drop / 3


def test_garch_recovers_persistence_of_a_garch_process():
    from rv.vol import garch_fit

    rng = np.random.default_rng(3)
    omega, alpha, beta, n = 0.02, 0.08, 0.90, 6000  # daily %-return units
    var, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = rng.normal() * np.sqrt(var)
        var = omega + alpha * r[t] ** 2 + beta * var
    fit = garch_fit(pd.Series(r / 100, index=pd.bdate_range("2000-01-03", periods=n)))
    assert abs(fit.persistence - 0.98) < 0.02
    assert len(fit.vol) == n and fit.vol.index[-1] == pd.bdate_range("2000-01-03", periods=n)[-1]
    assert abs(fit.long_run_vol - np.sqrt(252 * 1.0) / 100) < 0.04  # long-run var 1 (%^2/day)


def test_garch_forecast_reverts_to_long_run():
    from rv.vol import garch_fit

    fit = garch_fit(_gbm_returns(0.2, n=3000))
    path = fit.forecast(60)
    assert len(path) == 60
    assert abs(path.iloc[-1] - fit.long_run_vol) <= abs(path.iloc[0] - fit.long_run_vol) + 1e-9


def test_range_estimators_recover_sigma_and_are_less_noisy():
    from rv.vol import garman_klass, parkinson

    ohlc = _intraday_ohlc(0.2)
    pk = parkinson(ohlc["high"], ohlc["low"], 20)
    gk = garman_klass(ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], 20)
    cc = rolling_vol(np.log(ohlc["close"]).diff(), 20)
    assert abs(pk.mean() - 0.2) < 0.015 and abs(gk.mean() - 0.2) < 0.015
    assert pk.std() < 0.7 * cc.std() and gk.std() < 0.7 * cc.std()


def test_yang_zhang_captures_overnight_gaps_that_garman_klass_misses():
    from rv.vol import garman_klass, yang_zhang

    rng = np.random.default_rng(5)
    ohlc = _intraday_ohlc(np.sqrt(0.7) * 0.2, days=1500)  # 70% of variance intraday
    gaps = np.exp(np.cumsum(rng.normal(0, np.sqrt(0.3) * 0.2 / np.sqrt(252), len(ohlc))))
    ohlc = ohlc.mul(gaps, axis=0)  # 30% overnight: each day's prices shifted by a cumulative gap
    args = (ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], 60)
    assert abs(yang_zhang(*args).mean() - 0.2) < 0.01
    assert garman_klass(*args).mean() < 0.18
