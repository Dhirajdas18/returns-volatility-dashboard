import numpy as np
import pandas as pd

from rv.forecast import evaluate, forecasts, qlike, realised_forward


def _garch_returns(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.02, 0.08, 0.90
    var, r = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        r[t] = rng.normal() * np.sqrt(var)
        var = omega + alpha * r[t] ** 2 + beta * var
    return pd.Series(r / 100, index=pd.bdate_range("2005-01-03", periods=n))


def test_realised_forward_uses_only_the_next_h_days():
    r = pd.Series([0.0, 0.01, -0.01, 0.02, 0.0], index=pd.bdate_range("2020-01-01", periods=5))
    out = realised_forward(r, 2)
    assert np.isclose(out.iloc[0], np.sqrt((0.01**2 + 0.01**2) / 2 * 252))
    assert out.iloc[-2:].isna().all()


def test_qlike_is_zero_when_perfect_and_punishes_under_forecasts_more():
    real = pd.Series([0.2, 0.2])
    assert np.isclose(qlike(real, real), 0)
    under = qlike(pd.Series([0.2]), pd.Series([0.2 / np.sqrt(2)]))  # variance half of real
    over = qlike(pd.Series([0.2]), pd.Series([0.2 * np.sqrt(2)]))  # variance double
    assert under > over > 0


def test_forecasts_have_no_look_ahead():
    r = _garch_returns()
    split = r.index[2500]
    base = forecasts(r, train_end=split)
    shocked = r.copy()
    shocked.iloc[3000:] *= 3  # change only the future
    after = forecasts(shocked, train_end=split)
    day = r.index[2999]
    pd.testing.assert_series_equal(base.loc[day], after.loc[day])


def test_garch_beats_slow_window_on_a_garch_process():
    r = _garch_returns()
    split = r.index[2000]
    scores = evaluate(forecasts(r, train_end=split), realised_forward(r, 21), start=split)
    assert scores.loc["GARCH", "QLIKE"] < scores.loc["252-day rolling", "QLIKE"]
    assert list(scores.columns) == ["RMSE", "QLIKE", "bias"]


def test_scoreboard_is_symbols_by_estimators_relative_to_best():
    from rv.forecast import scoreboard

    r = _garch_returns(3000)
    p = 100 * np.exp(r.cumsum())
    table = pd.concat({("adj_close", "A"): p, ("adj_close", "B"): p * 2}, axis=1)
    table.columns.names = ["field", "symbol"]
    board = scoreboard(table, train_end=r.index[2000])
    assert list(board.columns) == ["A", "B"]
    assert np.allclose(board.min(), 1.0)  # each symbol's best estimator scores 1.00
