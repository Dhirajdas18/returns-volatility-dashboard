import numpy as np
import pandas as pd

from rv.site import build_html, rolling_figure


def _prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=600)
    steps = rng.normal(0, 0.01, (600, 2))
    return pd.DataFrame(100 * np.exp(steps.cumsum(axis=0)), index=idx, columns=["NIFTY", "ITC"])


def test_rolling_figure_one_trace_per_symbol_window_and_dropdown():
    fig = rolling_figure(_prices(), windows=(20, 60))
    assert len(fig.data) == 4
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert labels == ["NIFTY", "ITC"]
    # only the first symbol's traces start visible
    assert [t.visible for t in fig.data] == [True, True, False, False]


def test_build_html_is_one_page_loading_plotly_once():
    html = build_html(_prices())
    assert html.startswith("<!doctype html>")
    assert html.count("cdn.plot.ly") == 1
    assert "<title>" in html
