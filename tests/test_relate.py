import numpy as np
import pandas as pd

from rv.relate import (
    beta,
    beta_forecast_errors,
    blume,
    label_regimes,
    regime_corr,
    rolling_beta,
    run_lengths,
    transition_matrix,
)


def _pair(b=1.5, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    m = pd.Series(rng.normal(0, 0.01, n), index=idx)
    r = b * m + pd.Series(rng.normal(0, 0.01, n), index=idx)
    return r, m


def test_beta_recovers_slope_and_equals_corr_times_vol_ratio():
    r, m = _pair(1.5)
    assert abs(beta(r, m) - 1.5) < 0.05
    assert np.isclose(beta(r, m), r.corr(m) * r.std() / m.std())


def test_rolling_beta_and_blume():
    r, m = _pair()
    assert rolling_beta(r, m, 60).iloc[:59].isna().all()
    assert np.isclose(blume(1.6), 1.4) and np.isclose(blume(0.4), 0.6)


def test_beta_forecast_errors_longer_windows_win_when_beta_is_constant():
    r, m = _pair(1.5)
    err = beta_forecast_errors(r, m, windows=(63, 504))
    assert list(err.columns) == ["raw", "shrunk to 1"]
    assert err.loc[504, "raw"] < err.loc[63, "raw"]


def test_label_regimes_by_past_percentile_only():
    idx = pd.bdate_range("2015-01-01", periods=60)
    vol = pd.Series(np.r_[np.full(40, 0.10), np.linspace(0.1, 0.5, 20)], index=idx)
    lab = label_regimes(vol, window=20)
    assert lab.iloc[-1] == "stressed"
    changed = vol.copy()
    changed.iloc[50:] = 0.01  # change the future
    assert label_regimes(changed, window=20).iloc[45] == lab.iloc[45]


def test_transition_matrix_rows_sum_to_one_and_run_lengths():
    lab = pd.Series(["calm", "calm", "stressed", "stressed", "stressed", "calm"])
    t = transition_matrix(lab)
    assert np.allclose(t.sum(axis=1), 1.0)
    runs = run_lengths(lab)
    assert sorted(runs["calm"]) == [1, 2] and runs["stressed"] == [3]


def test_regime_corr_average_pairwise():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=400)
    common = rng.normal(0, 1, 400)
    df = pd.DataFrame({k: common + rng.normal(0, 1, 400) for k in "ABC"}, index=idx)
    lab = pd.Series("calm", index=idx)
    assert abs(regime_corr(df, lab)["calm"] - 0.5) < 0.1
