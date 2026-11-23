import numpy as np
import pandas as pd

from rv.implied import premium, regime_returns, rolling_percentile


def _idx(n):
    return pd.bdate_range("2015-01-01", periods=n)


def test_rolling_percentile_is_rank_within_past_window_only():
    x = pd.Series(np.arange(1.0, 11.0), index=_idx(10))
    pct = rolling_percentile(x, 5)
    assert pct.iloc[:4].isna().all()
    assert pct.iloc[-1] == 1.0
    y = x.copy()
    y.iloc[8:] = 0.0  # change the future
    assert rolling_percentile(y, 5).iloc[7] == pct.iloc[7]


def test_premium_is_implied_minus_realised():
    vix = pd.Series([0.20, 0.30], index=_idx(2))
    real = pd.Series([0.15, 0.40], index=_idx(2))
    out = premium(vix, real)
    assert np.allclose(out["spread"], [0.05, -0.10])
    assert list(out.columns) == ["implied", "realised", "spread"]


def test_regime_returns_uses_non_overlapping_months_and_splits_on_threshold():
    n = 21 * 40
    idx = _idx(n)
    vix = pd.Series(np.where(np.arange(n) // 210 % 2 == 0, 0.15, 0.30), index=idx)
    pct = pd.Series(np.where(vix > 0.2, 0.9, 0.3), index=idx)
    daily = np.where(vix > 0.2, 0.002, 0.0)  # high-VIX stretches rise, others flat
    prices = pd.Series(100 * np.exp(np.cumsum(daily)), index=idx)
    out = regime_returns(prices, pct, horizon=21, threshold=0.8)
    assert out.loc["high", "months"] + out.loc["normal", "months"] == 39  # last month has no future
    assert out.loc["high", "mean return"] > out.loc["normal", "mean return"]


def test_regime_returns_offset_shifts_the_sampled_days():
    idx = _idx(21 * 10)
    prices = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    pct = pd.Series(0.5, index=idx)
    a = regime_returns(prices, pct, offset=0)
    b = regime_returns(prices, pct, offset=5)
    assert a.loc["normal", "mean return"] != b.loc["normal", "mean return"]
