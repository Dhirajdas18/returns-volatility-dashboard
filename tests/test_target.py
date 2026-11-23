import numpy as np
import pandas as pd

from rv.perf import sharpe
from rv.target import apply_weights, compare, static_match, vol_target_weights


def _idx(n):
    return pd.bdate_range("2010-01-01", periods=n)


def test_weights_are_target_over_vol_capped():
    vol = pd.Series([0.2, 0.05, 0.4], index=_idx(3))
    w = vol_target_weights(vol, target=0.1, cap=1.0)
    assert np.allclose(w, [0.5, 1.0, 0.25])


def test_apply_weights_uses_yesterdays_weight_and_charges_turnover():
    r = pd.Series([0.01, 0.02, -0.01], index=_idx(3))
    w = pd.Series([0.5, 1.0, 0.0], index=_idx(3))
    out = apply_weights(r, w, rf=0.0, cost=0.001, start_weight=1.0)
    # day 1 runs on the start weight 1.0; day 2 on 0.5 (decided day 1); day 3 on 1.0
    expected = [0.01, 0.5 * 0.02 - 0.5 * 0.001, 1.0 * -0.01 - 0.5 * 0.001]
    assert np.allclose(out["return"], expected)
    assert np.allclose(out["weight"], [1.0, 0.5, 1.0])


def test_tomorrows_weight_cannot_change_todays_return():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 50), index=_idx(50))
    w = pd.Series(0.7, index=r.index)
    w2 = w.copy()
    w2.iloc[-1] = 0.0
    a, b = apply_weights(r, w), apply_weights(r, w2)
    assert np.isclose(a["return"].iloc[-1], b["return"].iloc[-1])


def test_static_mix_keeps_the_same_excess_sharpe_as_full_exposure():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000), index=_idx(2000))
    used = pd.Series(np.linspace(0.3, 0.9, 2000), index=r.index)
    s = static_match(r, used, rf=0.05)
    assert np.isclose(s["weight"].mean(), used.mean())
    assert np.isclose(sharpe(s["return"], 0.05), sharpe(r, 0.05))


def test_vol_targeting_steadies_risk_on_a_clustered_series():
    rng = np.random.default_rng(2)
    n = 3000
    sig = np.where((np.arange(n) // 250) % 2 == 0, 0.005, 0.02)  # calm and wild years
    r = pd.Series(rng.normal(0, 1, n) * sig, index=_idx(n))
    vol = r.rolling(20).std() * np.sqrt(252)
    vt = apply_weights(r, vol_target_weights(vol, 0.08, cap=5.0))
    roll = lambda x: x.rolling(250).std().dropna()  # noqa: E731
    assert roll(vt["return"]).std() < 0.5 * roll(r).std()


def test_compare_table():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0004, 0.01, 800), index=_idx(800))
    table = compare({"hold": apply_weights(r, pd.Series(1.0, index=r.index))}, rf=0.0)
    assert {"CAGR", "Vol", "Sharpe", "Max DD", "Avg weight", "Turnover / yr"} <= set(table.columns)


def test_vol_target_with_no_leverage_never_holds_more_than_everything():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 500), index=_idx(500))
    vol = r.rolling(20).std() * np.sqrt(252)
    out = apply_weights(r, vol_target_weights(vol, 0.30, cap=1.0))
    assert out["weight"].max() <= 1.0 and out["weight"].min() > 0
