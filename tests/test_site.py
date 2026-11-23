import numpy as np
import pandas as pd

from rv.site import (
    build_html,
    estimators_figure,
    forecast_figure,
    implied_figure,
    monthly_figure,
    perf_table,
    relate_figure,
    rolling_figure,
    target_figure,
)


def _table():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=600)
    frames = {}
    for sym in ["NIFTY", "ITC"]:
        close = 100 * np.exp(rng.normal(0, 0.01, 600).cumsum())
        open_ = close * np.exp(rng.normal(0, 0.003, 600))
        hi = np.maximum(open_, close) * 1.004
        lo = np.minimum(open_, close) * 0.996
        frames[sym] = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                                    "adj_close": close}, index=idx)
    table = pd.concat(frames, axis=1).swaplevel(axis=1).sort_index(axis=1)
    table.columns.names = ["field", "symbol"]
    return table


def test_rolling_figure_one_trace_per_symbol_window_and_dropdown():
    fig = rolling_figure(_table(), windows=(20, 60))
    assert len(fig.data) == 4
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert labels == ["ITC", "NIFTY"]
    assert [t.visible for t in fig.data] == [True, True, False, False]


def test_estimators_figure_has_four_estimators_per_symbol():
    fig = estimators_figure(_table())
    assert len(fig.data) == 8
    assert {t.name for t in fig.data} == {"20-day rolling", "EWMA", "GARCH", "Yang-Zhang"}


def test_build_html_is_one_page_loading_plotly_once():
    html = build_html(_table())
    assert html.startswith("<!doctype html>")
    assert html.count('src="https://cdn.plot.ly') == 1
    assert "<title>" in html


def test_forecast_figure_is_one_heatmap_symbols_by_estimators():
    fig = forecast_figure(_table(), train_end="2016-06-01")
    assert len(fig.data) == 1 and fig.data[0].type == "heatmap"
    assert list(fig.data[0].x) == ["ITC", "NIFTY"]


def _with_vix():
    table = _table()
    vix = pd.Series(15.0, index=table.index)
    extra = pd.concat({("close", "INDIAVIX"): vix}, axis=1)
    extra.columns.names = ["field", "symbol"]
    return pd.concat([table, extra], axis=1)


def test_implied_figure_has_vix_and_realised_lines():
    fig = implied_figure(_with_vix())
    assert [t.name for t in fig.data] == ["India VIX (implied)", "NIFTY vol over the next month"]


def test_build_html_keeps_vix_out_of_the_stock_menus():
    html = build_html(_with_vix())
    assert "Implied vs realised volatility" in html
    assert html.count('"label":"INDIAVIX"') == 0


def test_perf_table_is_html_with_a_row_per_symbol():
    html = perf_table(_table())
    assert html.count("<tr>") == 3  # header + 2 symbols
    assert '<th scope="row">ITC</th>' in html and '<th scope="row">NIFTY</th>' in html


def test_build_html_has_nav_links_matching_section_ids():
    html = build_html(_table())
    assert 'href="#rolling-volatility"' in html and 'id="rolling-volatility"' in html


def test_monthly_figure_one_heatmap_per_symbol_with_dropdown():
    fig = monthly_figure(_table())
    assert [t.type for t in fig.data] == ["heatmap", "heatmap"]
    assert [t.visible for t in fig.data] == [True, False]


def test_relate_figure_beta_and_corr_per_stock_excluding_the_index():
    fig = relate_figure(_table())
    assert [t.name for t in fig.data] == ["beta (1 year)", "correlation (1 year)"]
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["ITC"]


def test_target_figure_vol_target_vs_static_per_symbol():
    fig = target_figure(_table())
    names = [t.name for t in fig.data[:2]]
    assert names == ["vol target (0.9× own vol)", "fixed mix, same exposure"]
    assert len(fig.data) == 4


def test_dashboard_numbers_match_reports_when_another_symbol_has_gaps():
    from rv.perf import summary

    table = _table()
    col = ("adj_close", "NIFTY")
    table.loc[table.index[105:], col] *= 1.5  # NIFTY jumps 50% across a gap...
    table.loc[table.index[100:105], col] = np.nan  # ...of days it didn't trade but stocks did
    html = perf_table(table)
    nifty = table["adj_close"]["NIFTY"].dropna().pct_change().dropna()
    expected = summary(nifty.to_frame(), 0.065).loc["NIFTY", "CAGR"]
    row = html.split('<th scope="row">NIFTY</th>')[1].split("</tr>")[0].replace("\u2212", "-")
    assert f"{expected:.1%}" in row
