import numpy as np
import pandas as pd

from rv.perf import (
    cagr,
    calmar,
    max_drawdown,
    monthly_table,
    rolling_sharpe,
    sharpe,
    sharpe_se,
    sortino,
    summary,
)


def _r(mu=0.10, sigma=0.20, n=252 * 40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu / 252, sigma / np.sqrt(252), n),
                     index=pd.bdate_range("1990-01-01", periods=n))


def test_sharpe_recovers_mu_over_sigma_and_subtracts_risk_free():
    r = _r(0.10, 0.20)
    assert abs(sharpe(r) - 0.5) < 0.2
    assert np.isclose(sharpe(r, rf=0.04), sharpe(r - 0.04 / 252))


def test_sharpe_se_matches_simulation_spread():
    est = [sharpe(_r(n=252 * 10, seed=s)) for s in range(300)]
    se = np.mean([sharpe_se(_r(n=252 * 10, seed=s)) for s in range(20)])
    assert abs(np.std(est) - se) / se < 0.15  # about 1/sqrt(10 years) = 0.32


def test_sortino_ignores_upside_moves():
    r = pd.Series([0.012, -0.01, 0.012, -0.01] * 100)
    bigger_up = r.where(r < 0, 0.03)
    assert sortino(bigger_up) > sortino(r)
    assert np.isclose(sortino(bigger_up) / sortino(r), bigger_up.mean() / r.mean())


def test_max_drawdown_and_calmar():
    prices_r = pd.Series([0.10, -0.50, 0.20], index=pd.bdate_range("2020-01-01", periods=3))
    assert np.isclose(max_drawdown(prices_r), -0.5)
    assert np.isclose(calmar(prices_r), cagr(prices_r) / 0.5)


def test_cagr_without_dates_assumes_252_rows_a_year():
    r = pd.Series(2 ** (1 / 504) - 1, index=range(505))  # no dates: 505 rows ~ 2 years
    r.iloc[0] = 0.0
    assert abs(cagr(r) - (2 ** (252 / 505) - 1)) < 1e-9


def test_rolling_sharpe_and_monthly_table_shapes():
    r = _r(n=600)
    assert rolling_sharpe(r, 252).iloc[:251].isna().all()
    table = monthly_table(r)
    assert list(table.columns[:12]) == list(range(1, 13)) and "Year" in table.columns
    jan = r[(r.index.year == 1990) & (r.index.month == 1)]
    assert np.isclose(table.loc[1990, 1], (1 + jan).prod() - 1)


def test_summary_has_one_row_per_column():
    df = pd.DataFrame({"A": _r(seed=1), "B": _r(seed=2)})
    out = summary(df, rf=0.0)
    assert list(out.index) == ["A", "B"]
    assert {"CAGR", "Vol", "Sharpe", "Sharpe SE", "Sortino", "Max DD", "Calmar"} <= set(out.columns)


def test_cagr_uses_calendar_time_when_dates_are_known():
    idx = pd.bdate_range("2020-01-01", "2022-01-01")  # about 2 calendar years, ~523 rows
    r = pd.Series(0.0, index=idx)
    r.iloc[-1] = 1.0  # wealth doubles once
    years = (idx[-1] - idx[0]).days / 365.25
    assert abs(cagr(r) - (2 ** (1 / years) - 1)) < 1e-9
