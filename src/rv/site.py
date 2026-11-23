"""Interactive dashboard: one static HTML page of Plotly charts, built from the price table."""

import pandas as pd
import plotly.graph_objects as go

from rv import plots
from rv.data import INDEX, VIX
from rv.forecast import realised_forward, scoreboard
from rv.implied import premium
from rv.perf import monthly_table, summary
from rv.returns import log_returns
from rv.vol import ewma_vol, garch_fit, rolling_vol, yang_zhang

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
        font=dict(family="Inter, system-ui, sans-serif", color=plots.TEXT_2, size=13),
        hovermode="x unified",
        margin=dict(l=56, r=24, t=96, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left"),
        yaxis=dict(title=ytitle, gridcolor=plots.GRID, zeroline=False),
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
        font=dict(family="Inter, system-ui, sans-serif", color=plots.TEXT_2, size=13),
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


def perf_figure(table: pd.DataFrame) -> go.Figure:
    """Risk-adjusted summary table, one row per symbol."""
    r = table["adj_close"].pct_change().iloc[1:]
    s = summary(r, RF)
    lo, hi = s["Sharpe"] - 1.96 * s["Sharpe SE"], s["Sharpe"] + 1.96 * s["Sharpe SE"]
    cols = {
        "": list(s.index),
        "CAGR": [f"{v:.1%}" for v in s["CAGR"]],
        "Volatility": [f"{v:.1%}" for v in s["Vol"]],
        "Sharpe": [f"{v:.2f}" for v in s["Sharpe"]],
        "Sharpe 95% range": [f"{a:+.2f} to {b:+.2f}" for a, b in zip(lo, hi, strict=True)],
        "Sortino": [f"{v:.2f}" for v in s["Sortino"]],
        "Max drawdown": [f"{v:.0%}" for v in s["Max DD"]],
        "Calmar": [f"{v:.2f}" for v in s["Calmar"]],
    }
    fig = go.Figure(go.Table(
        header=dict(values=list(cols), fill_color=plots.SURFACE, align="left",
                    font=dict(color=plots.TEXT, size=13), line_color=plots.GRID),
        cells=dict(values=list(cols.values()), fill_color=plots.SURFACE, align="left",
                   font=dict(color=plots.TEXT_2, size=13), line_color=plots.GRID, height=30),
    ))
    fig.update_layout(paper_bgcolor=plots.SURFACE, margin=dict(l=0, r=0, t=8, b=0),
                      height=80 + 30 * len(s))
    return fig


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
                      font=dict(family="Inter, system-ui, sans-serif", color=plots.TEXT_2, size=12),
                      margin=dict(l=56, r=24, t=64, b=40), yaxis=dict(autorange="reversed"))
    return fig


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
     perf_figure),
    ("Monthly returns",
     "Every calendar month, for the chosen symbol. Patterns like 'January is weak' come from only "
     "13–14 observations per month, which is too few to trade on.",
     monthly_figure),
]

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Returns &amp; Volatility</title>
<link rel="icon" href="{icon}">
<style>
  :root {{ --surface: {surface}; --text: {text}; --text-2: {text2}; --grid: {grid}; }}
  body {{ margin: 0; background: var(--surface); color: var(--text);
         font: 16px/1.55 Inter, system-ui, sans-serif; }}
  main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-size: 28px; margin: 0 0 4px; }}
  .meta {{ color: var(--text-2); margin: 0 0 32px; }}
  section {{ border-top: 1px solid var(--grid); padding-top: 24px; margin-top: 24px; }}
  h2 {{ font-size: 20px; margin: 0 0 6px; }}
  section p {{ color: var(--text-2); max-width: 70ch; margin: 0 0 8px; }}
</style>
</head>
<body>
<main>
<h1>Returns &amp; Volatility</h1>
<p class="meta">NSE large caps and NIFTY 50, daily data from Yahoo Finance.
Updated {updated}. Pick a symbol from each chart's menu.</p>
{sections}
</main>
<script>
// Fit each chart's y-axis to the visible traces and date range (Plotly keeps the full range).
document.querySelectorAll(".plotly-graph-div").forEach((gd) => {{
  let busy = false;
  const fit = () => {{
    if (busy) return;
    const [x0, x1] = (gd.layout.xaxis.range || []).map(String);
    let lo = Infinity, hi = -Infinity;
    gd.data.forEach((t) => {{
      if (t.visible === false || t.type !== "scatter") return;
      for (let i = 0; i < t.x.length; i++) {{
        if (x0 && (t.x[i] < x0.slice(0, 10) || t.x[i] > x1.slice(0, 10))) continue;
        lo = Math.min(lo, t.y[i]); hi = Math.max(hi, t.y[i]);
      }}
    }});
    if (!isFinite(lo)) return;
    const pad = (hi - lo) * 0.08 || 1;
    busy = true;
    Plotly.relayout(gd, {{"yaxis.range": [Math.max(0, lo - pad), hi + pad]}})
      .then(() => {{ busy = false; }});
  }};
  if (!gd.data.some((t) => t.type === "scatter")) return;
  gd.on("plotly_relayout", (e) => {{ if (!("yaxis.range" in e)) fit(); }});
  gd.on("plotly_update", fit);
  fit();
}});
</script>
</body>
</html>
"""


def build_html(table: pd.DataFrame) -> str:
    """`table`: the (field, symbol) price table from rv.data."""
    parts = []
    has_vix = VIX in table.columns.get_level_values("symbol")
    stocks = table.drop(columns=VIX, level="symbol") if has_vix else table
    for i, (title, blurb, make) in enumerate(SECTIONS):
        if make is implied_figure and not has_vix:
            continue
        fig = make(table) if make is implied_figure else make(stocks)
        div = fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                          config={"displaylogo": False, "responsive": True})
        parts.append(f"<section>\n<h2>{title}</h2>\n<p>{blurb}</p>\n{div}\n</section>")
    return PAGE.format(icon=ICON, surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2,
                       grid=plots.GRID, updated=f"{table.index[-1]:%d %b %Y}",
                       sections="\n".join(parts))
