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
