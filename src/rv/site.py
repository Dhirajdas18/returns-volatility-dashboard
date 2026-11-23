"""Interactive dashboard: one static HTML page of Plotly charts, built from the price table."""

import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rv import plots
from rv.data import INDEX, VIX
from rv.forecast import realised_forward, scoreboard
from rv.implied import premium
from rv.perf import monthly_table, summary
from rv.relate import rolling_beta, rolling_corr
from rv.returns import log_returns
from rv.target import apply_weights, static_match, vol_target_weights
from rv.vol import ewma_vol, garch_fit, rolling_vol, yang_zhang

FONT = "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 12 5 7l3 3 7-8%22 fill=%22none%22 stroke=%22%232a78d6%22"
    " stroke-width=%222%22/%3E"
    "%3C/svg%3E"
)
WINDOW_COLORS = {20: plots.BLUE_RAMP[2], 60: plots.BLUE_RAMP[4], 252: plots.BLUE_RAMP[6]}


def _layout(fig: go.Figure, ytitle: str) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=plots.SURFACE,
        plot_bgcolor=plots.SURFACE,
        font=dict(family=FONT, color=plots.TEXT_2, size=13),
        hovermode="x unified",
        margin=dict(l=56, r=24, t=96, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
        yaxis=dict(title=ytitle, gridcolor=plots.GRID, zeroline=False, automargin=True),
        xaxis=dict(
            gridcolor=plots.GRID,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[dict(count=n, label=f"{n}y", step="year", stepmode="backward")
                         for n in (1, 3, 5)] + [dict(step="all", label="All")],
                x=1, xanchor="right", y=1.08, yanchor="bottom", bgcolor="white",
            ),
        ),
    )
    end = max(pd.Timestamp(max(t.x)) for t in fig.data)
    fig.update_xaxes(range=[end - pd.DateOffset(years=3), end])  # open on the last 3 years
    return fig


def _symbol_dropdown(fig: go.Figure, symbols: list[str], per_symbol: int) -> None:
    """Dropdown that shows one symbol's group of traces at a time."""
    buttons = []
    for i, sym in enumerate(symbols):
        visible = [j // per_symbol == i for j in range(len(symbols) * per_symbol)]
        buttons.append(dict(label=sym, method="update", args=[{"visible": visible}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.14,
                                        yanchor="bottom", bgcolor="white")])


def _line(fig: go.Figure, v: pd.Series, name: str, color: str, visible: bool) -> None:
    v = v.dropna() * 100
    # plain lists (not binary arrays) so the page script can read y; short dates keep it small
    fig.add_scatter(x=list(v.index.strftime("%Y-%m-%d")), y=v.round(1).tolist(), name=name,
                    visible=visible,
                    line=dict(color=color, width=2), hovertemplate="%{y:.1f}%")


def rolling_figure(table: pd.DataFrame, windows=(20, 60, 252)) -> go.Figure:
    """Rolling annualised volatility for each symbol and window, picked by dropdown."""
    fig = go.Figure()
    prices = table["adj_close"]
    r = log_returns(prices)
    symbols = list(prices.columns)
    for i, sym in enumerate(symbols):
        for w in windows:
            _line(fig, rolling_vol(r[sym].dropna(), w), f"{w}-day",
                  WINDOW_COLORS.get(w, plots.MUTED), i == 0)
    _symbol_dropdown(fig, symbols, len(windows))
    return _layout(fig, "annualised volatility, %")


def estimators_figure(table: pd.DataFrame) -> go.Figure:
    """20-day rolling vs EWMA vs GARCH vs 20-day Yang-Zhang for each symbol."""
    fig = go.Figure()
    c = plots.ESTIMATOR_COLORS
    symbols = list(table["adj_close"].columns)
    for i, sym in enumerate(symbols):
        r = log_returns(table["adj_close"][sym].dropna())
        o = pd.concat({f: table[f][sym] for f in ("open", "high", "low", "close")}, axis=1).dropna()
        lines = {
            "20-day rolling": rolling_vol(r, 20),
            "EWMA": ewma_vol(r),
            "GARCH": garch_fit(r).vol,
            "Yang-Zhang": yang_zhang(o["open"], o["high"], o["low"], o["close"], 20),
        }
        for name, v in lines.items():
            _line(fig, v, name, c[name], i == 0)
    _symbol_dropdown(fig, symbols, 4)
    return _layout(fig, "annualised volatility, %")


def forecast_figure(table: pd.DataFrame, train_end="2020-01-01") -> go.Figure:
    """Out-of-sample forecast scoreboard: QLIKE relative to each symbol's best estimator."""
    board = scoreboard(table, train_end)
    fig = go.Figure(go.Heatmap(
        z=board.to_numpy(), x=list(board.columns), y=list(board.index),
        text=board.round(2).to_numpy(), texttemplate="%{text:.2f}", showscale=False,
        colorscale=[[i / (len(plots.BLUE_RAMP) - 1), c] for i, c in enumerate(plots.BLUE_RAMP)],
        zmin=1.0, zmax=max(2.0, float(board.max().max())),
        hovertemplate="%{y} on %{x}: %{z:.2f}× the best error<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=plots.SURFACE, plot_bgcolor=plots.SURFACE, height=420,
        font=dict(family=FONT, color=plots.TEXT_2, size=13),
        margin=dict(l=140, r=24, t=24, b=40), yaxis=dict(autorange="reversed"),
    )
    return fig


def implied_figure(table: pd.DataFrame) -> go.Figure:
    """India VIX at each date vs the NIFTY volatility that followed over the next month."""
    vix = table["close"][VIX].dropna() / 100
    real = realised_forward(log_returns(table["adj_close"][INDEX].dropna()))
    prem = premium(vix, real)
    fig = go.Figure()
    _line(fig, prem["implied"], "India VIX (implied)", plots.SERIES[0], True)
    _line(fig, prem["realised"], "NIFTY vol over the next month", plots.TEXT, True)
    return _layout(fig, "annualised volatility, %")


RF = 0.065  # assumed risk-free rate (India 91-day T-bills ran roughly 3-7% over 2013-2026)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def perf_table(table: pd.DataFrame) -> str:
    """Risk-adjusted summary as an HTML table, one row per symbol."""
    r = table["adj_close"].pct_change().iloc[1:]
    s = summary(r, RF)
    lo, hi = s["Sharpe"] - 1.96 * s["Sharpe SE"], s["Sharpe"] + 1.96 * s["Sharpe SE"]
    head = ["", "CAGR", "Volatility", "Sharpe", "Sharpe 95% range", "Sortino", "Max drawdown",
            "Calmar"]
    rows = []
    for sym in s.index:
        v = s.loc[sym]
        cells = [f"{v['CAGR']:.1%}", f"{v['Vol']:.1%}", f"{v['Sharpe']:.2f}",
                 f"{lo[sym]:+.2f} to {hi[sym]:+.2f}", f"{v['Sortino']:.2f}",
                 f"{v['Max DD']:.0%}", f"{v['Calmar']:.2f}"]
        tds = "".join(f"<td>{c}</td>" for c in cells).replace("-", "\u2212")
        rows.append(f'<tr><th scope="row">{sym}</th>{tds}</tr>')
    ths = "".join(f'<th scope="col">{h}</th>' for h in head)
    return (f'<div class="table-scroll"><table class="data"><thead><tr>{ths}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def monthly_figure(table: pd.DataFrame) -> go.Figure:
    """Calendar heatmap of monthly returns, one symbol at a time."""
    fig = go.Figure()
    symbols = list(table["adj_close"].columns)
    scale = [[0, plots.SERIES[7]], [0.5, "#f1f0ec"], [1, plots.SERIES[0]]]
    for i, sym in enumerate(symbols):
        m = monthly_table(table["adj_close"][sym].dropna().pct_change().iloc[1:]).iloc[:, :12]
        z = (m * 100).round(1)
        fig.add_heatmap(z=z.to_numpy(), x=MONTHS, y=[str(y) for y in m.index], visible=i == 0,
                        colorscale=scale, zmid=0, zmin=-15, zmax=15, showscale=False,
                        texttemplate="%{z:.0f}", name=sym,
                        hovertemplate="%{x} %{y}: %{z:.1f}%<extra></extra>")
    _symbol_dropdown(fig, symbols, 1)
    fig.update_layout(paper_bgcolor=plots.SURFACE, plot_bgcolor=plots.SURFACE, height=520,
                      font=dict(family=FONT, color=plots.TEXT_2, size=12),
                      margin=dict(l=56, r=24, t=64, b=40), yaxis=dict(autorange="reversed"))
    return fig


def relate_figure(table: pd.DataFrame) -> go.Figure:
    """One-year rolling beta and correlation of each stock against NIFTY."""
    r = table["adj_close"].pct_change().iloc[1:]
    stocks = [c for c in r.columns if c != INDEX]
    fig = go.Figure()
    for i, sym in enumerate(stocks):
        pair = r[[sym, INDEX]].dropna()
        for name, series, color in (
            ("beta (1 year)", rolling_beta(pair[sym], pair[INDEX]), plots.SERIES[0]),
            ("correlation (1 year)", rolling_corr(pair[sym], pair[INDEX]), plots.MUTED),
        ):
            v = series.dropna()
            fig.add_scatter(x=list(v.index.strftime("%Y-%m-%d")), y=v.round(2).tolist(),
                            name=name, visible=i == 0, line=dict(color=color, width=2),
                            hovertemplate="%{y:.2f}")
    _symbol_dropdown(fig, stocks, 2)
    return _layout(fig, "vs NIFTY")


COST = 0.001  # 0.1% of the amount traded (assumption)


def target_figure(table: pd.DataFrame) -> go.Figure:
    """Growth of ₹1 lakh: vol targeting (no leverage) vs a fixed mix with the same exposure."""
    r = table["adj_close"].pct_change().iloc[1:]
    symbols = list(r.columns)
    fig = go.Figure()
    for i, sym in enumerate(symbols):
        x = r[sym].dropna()
        target = float(x.std() * (252**0.5) * 0.9)
        vt = apply_weights(x, vol_target_weights(ewma_vol(np.log1p(x)), target), RF, COST)
        st = static_match(x, vt["weight"], RF, COST)
        for name, df, color in (("vol target (0.9× own vol)", vt, plots.SERIES[0]),
                                ("fixed mix, same exposure", st, plots.TEXT)):
            w = (1 + df["return"]).cumprod()
            fig.add_scatter(x=list(w.index.strftime("%Y-%m-%d")), y=w.round(3).tolist(),
                            name=name, visible=i == 0, line=dict(color=color, width=2),
                            hovertemplate="₹%{y:.2f} lakh")
    _symbol_dropdown(fig, symbols, 2)
    return _layout(fig, "₹ lakh, started with 1 (after costs)")


SECTIONS = [
    ("Rolling volatility",
     "Standard deviation of the last N daily returns, annualised. Short windows react fast but are "
     "noisy; long windows are smooth but lag, and every window drops a big day all at once N days "
     "later, so the line falls off a cliff on an ordinary day.",
     rolling_figure),
    ("Volatility estimators",
     "EWMA weights recent days more and fades old shocks smoothly (decay 0.94 a day). GARCH(1,1) "
     "adds a pull back to the stock's long-run level. Yang-Zhang uses each day's open, high, low "
     "and close, including the overnight gap that pure high/low estimators miss.",
     estimators_figure),
    ("Which estimator forecasts best?",
     "Each estimator predicts the next 21 trading days' volatility every day. Models are fitted on "
     "2013–2019 only and scored on 2020 onwards with QLIKE, the standard loss for volatility "
     "forecasts, which punishes under-forecasting risk hardest. 1.00 = best for that symbol.",
     forecast_figure),
    ("Implied vs realised volatility",
     "India VIX is the volatility NIFTY options price in for the next 30 days. It sat above the "
     "volatility that actually followed on about 4 days in 5: option buyers pay a premium for "
     "protection. It still forecast NIFTY's next month better than any model here, but it was "
     "calm on the eve of the 2020 crash.",
     implied_figure),
    ("Risk-adjusted returns",
     "Sharpe = excess return over a 6.5% risk-free rate per unit of volatility; Sortino counts "
     "only downside volatility; Calmar = growth rate over the worst drawdown. After 14 years each "
     "Sharpe ratio is still uncertain by about ±0.5, so small differences between stocks mean "
     "nothing.",
     perf_table),
    ("Monthly returns",
     "Every calendar month, for the chosen symbol. Patterns like ‘January is weak’ come from only "
     "13–14 observations per month, which is too few to trade on.",
     monthly_figure),
    ("Beta and correlation to NIFTY",
     "Beta: how much the stock moves, on average, when NIFTY moves 1%. It equals correlation times "
     "the stock's volatility relative to NIFTY, so a volatile stock can still have a low beta. "
     "One-year betas wander a lot; a three-year window predicts next year's beta best.",
     relate_figure),
    ("Volatility targeting",
     "Hold less when forecast volatility (EWMA) is high and more when it is low, never above 100%. "
     "Compared with a fixed mix holding the same average amount, it softened the 2020 crash but "
     "earned less and had deeper ordinary drawdowns in most years: crash insurance, not a free "
     "lunch.",
     target_figure),
]

REPO = "https://github.com/Dhirajdas18/returns-volatility-dashboard"

PAGE = Template((Path(__file__).with_name("page.html")).read_text(encoding="utf-8"))


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


NAV_LABELS = {
    "Which estimator forecasts best?": "Forecast test",
    "Implied vs realised volatility": "India VIX",
    "Beta and correlation to NIFTY": "Beta",
}


def build_html(table: pd.DataFrame) -> str:
    """`table`: the (field, symbol) price table from rv.data."""
    parts, nav = [], []
    has_vix = VIX in table.columns.get_level_values("symbol")
    stocks = table.drop(columns=VIX, level="symbol") if has_vix else table
    plotly_loaded = False
    for title, blurb, make in SECTIONS:
        if make is implied_figure and not has_vix:
            continue
        out = make(table) if make is implied_figure else make(stocks)
        if isinstance(out, str):
            body = out
        else:
            body = out.to_html(full_html=False, include_plotlyjs=False if plotly_loaded else "cdn",
                               config={"displaylogo": False, "responsive": True})
            plotly_loaded = True
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{NAV_LABELS.get(title, title)}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    dark = {k.lower(): v for k, v in plots.DARK.items()}
    return PAGE.substitute(
        icon=ICON, font=FONT, repo=REPO, updated=f"{table.index[-1]:%d %b %Y}",
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=plots.SERIES[0], d_surface=plots.DARK[plots.SURFACE], d_text=plots.DARK[plots.TEXT],
        d_text2=plots.DARK[plots.TEXT_2], d_grid=plots.DARK[plots.GRID],
        d_accent=plots.DARK[plots.SERIES[0]], dark=json.dumps(dark),
        dark_lines=json.dumps({k.lower(): v for k, v in plots.DARK_LINES.items()}),
        nav="\n".join(nav), sections="\n".join(parts),
    )
