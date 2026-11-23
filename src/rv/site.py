"""Interactive dashboard: one static HTML page of Plotly charts, built from the price table."""

import pandas as pd
import plotly.graph_objects as go

from rv import plots
from rv.returns import log_returns
from rv.vol import rolling_vol

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
        margin=dict(l=56, r=24, t=64, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
        yaxis=dict(title=ytitle, gridcolor=plots.GRID, zeroline=False),
        xaxis=dict(gridcolor=plots.GRID, rangeslider=dict(visible=False)),
    )
    return fig


def _symbol_dropdown(fig: go.Figure, symbols: list[str], per_symbol: int) -> None:
    """Dropdown that shows one symbol's group of traces at a time."""
    buttons = []
    for i, sym in enumerate(symbols):
        visible = [j // per_symbol == i for j in range(len(symbols) * per_symbol)]
        buttons.append(dict(label=sym, method="update", args=[{"visible": visible}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.08,
                                        yanchor="bottom", bgcolor="white")])


def rolling_figure(prices: pd.DataFrame, windows=(20, 60, 252)) -> go.Figure:
    """Rolling annualised volatility for each symbol and window, picked by dropdown."""
    fig = go.Figure()
    r = log_returns(prices)
    symbols = list(prices.columns)
    for i, sym in enumerate(symbols):
        for w in windows:
            v = rolling_vol(r[sym], w).dropna() * 100
            fig.add_scatter(x=v.index, y=v.round(1), name=f"{w}-day", visible=i == 0,
                            line=dict(color=WINDOW_COLORS.get(w, plots.MUTED), width=2),
                            hovertemplate="%{y:.1f}%")
    _symbol_dropdown(fig, symbols, len(windows))
    return _layout(fig, "annualised volatility, %")


SECTIONS = [
    ("Rolling volatility",
     "Standard deviation of the last N daily returns, annualised. Short windows react fast but are "
     "noisy; long windows are smooth but lag, and every window drops a big day all at once N days "
     "later, so the line falls off a cliff on an ordinary day.",
     rolling_figure),
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
</body>
</html>
"""


def build_html(prices: pd.DataFrame) -> str:
    parts = []
    for i, (title, blurb, make) in enumerate(SECTIONS):
        div = make(prices).to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False,
                                   config={"displaylogo": False, "responsive": True})
        parts.append(f"<section>\n<h2>{title}</h2>\n<p>{blurb}</p>\n{div}\n</section>")
    return PAGE.format(icon=ICON, surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2,
                       grid=plots.GRID, updated=f"{prices.index[-1]:%d %b %Y}",
                       sections="\n".join(parts))
