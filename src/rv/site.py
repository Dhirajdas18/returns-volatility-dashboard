"""Interactive dashboard: one static HTML page of Plotly charts, built from the price table."""

import pandas as pd
import plotly.graph_objects as go

from rv import plots
from rv.forecast import scoreboard
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
    for i, (title, blurb, make) in enumerate(SECTIONS):
        div = make(table).to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                                   config={"displaylogo": False, "responsive": True})
        parts.append(f"<section>\n<h2>{title}</h2>\n<p>{blurb}</p>\n{div}\n</section>")
    return PAGE.format(icon=ICON, surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2,
                       grid=plots.GRID, updated=f"{table.index[-1]:%d %b %Y}",
                       sections="\n".join(parts))
