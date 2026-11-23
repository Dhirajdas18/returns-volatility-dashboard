# Returns & Volatility Dashboard

[![CI](https://github.com/Dhirajdas18/returns-volatility-dashboard/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/returns-volatility-dashboard/actions/workflows/ci.yml)
[![Dashboard](https://github.com/Dhirajdas18/returns-volatility-dashboard/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/returns-volatility-dashboard/)

Volatility measurement and forecasting for NSE large caps (RELIANCE, HDFCBANK, INFY, TCS, ITC) and the
NIFTY 50, on daily data from 2013. Every forecast is scored out of sample.

**Live dashboard: [dhirajdas18.github.io/returns-volatility-dashboard](https://dhirajdas18.github.io/returns-volatility-dashboard/)**,
a static interactive site rebuilt every week from fresh data by GitHub Actions.

## Features

- [x] Rolling volatility across windows, with the window-length trade-off measured
- [x] EWMA, GARCH(1,1) and range estimators (Parkinson, Garman-Klass, Yang-Zhang)
- [ ] Out-of-sample volatility forecast comparison (RMSE, QLIKE)
- [ ] Implied (India VIX) vs realised volatility and the variance risk premium
- [ ] Risk-adjusted returns: Sharpe, Sortino, Calmar, rolling Sharpe, monthly heatmap
- [ ] Rolling beta and correlation, volatility regimes
- [ ] Volatility targeting

## Findings

### 1. Window length trades speed for noise, and every window has a ghost

<table><tr>
<td><a href="docs/img/rolling_windows.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/rolling_windows.png"><img src="docs/img/rolling_windows.gif" alt="NIFTY rolling volatility with 20, 60 and 252-day windows through 2020"></picture></a></td>
<td><a href="docs/img/rolling_ghost.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/rolling_ghost.png"><img src="docs/img/rolling_ghost.gif" alt="20-day volatility dropping 21 points on 24 April 2020"></picture></a></td>
</tr></table>

- In the 2020 crash, NIFTY's 20-day volatility peaked on 9 Apr 2020 at 90%. The 60-day window peaked
  seven weeks later (1 Jun, 59%). The 252-day window peaked on 3 Mar 2021 (32%), when markets had been
  calm for months.
- The 20-day estimate is about 15× noisier: its median daily change is 0.25 volatility points against
  0.02 for the 252-day window.
- **Ghost effect:** on 24 Apr 2020 the 20-day volatility fell from 75% to 54% even though NIFTY dropped
  1.7% that day. The cause was not that day's move: the 23 Mar −14% day had simply left the window.
  Equal-weight windows forget a shock all at once. Weighting recent days more heavily fixes this (next
  section).

### 2. EWMA and GARCH fix the ghost; high/low estimators need the overnight gap

<table><tr>
<td><a href="docs/img/estimators_2020.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/estimators_2020.png"><img src="docs/img/estimators_2020.gif" alt="NIFTY 2020: 20-day rolling vs EWMA vs GARCH volatility"></picture></a></td>
<td><a href="docs/img/garch_forecast.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/garch_forecast.png"><img src="docs/img/garch_forecast.gif" alt="GARCH volatility forecasts made on 23 Mar 2020 and 22 Jun 2017 converging to 16%"></picture></a></td>
</tr></table>

- **EWMA** (each day's weight shrinks 6% a day) peaked on 25 Mar 2020, two weeks before the 20-day
  window. Its largest one-day fall was 2.3 volatility points, against 21.1 for the 20-day window. Old
  shocks fade out; they don't fall off an edge.
- **GARCH(1,1)** adds a pull back to a long-run level. For NIFTY: persistence 0.977, long-run
  volatility 16%, and half of any shock gone in about 30 trading days. On 23 Mar 2020 it forecast 89%
  for the next day, 72% a month out, 46% three months out and 26% after six months. Shocks fade much
  faster in INFY and ITC (half-life about 4–5 days) than in NIFTY or HDFCBANK (25–30 days).
- Parameters here are fitted on the whole history, so this section describes behaviour; forecasting
  skill is tested out of sample in the next section.

<a href="docs/img/range_vs_close.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/range_vs_close.png"><img src="docs/img/range_vs_close.gif" alt="Close-to-close vs Garman-Klass vs Yang-Zhang volatility for six NSE series" width="720"></picture></a>

- High/low **range estimators** (Parkinson, Garman-Klass) use the whole day's path, so on simulated data
  they are much less noisy than close-to-close for the same window. On NSE they read 6–22% too low:
  21–37% of daily variance happens overnight, between one close and the next open, and they never see it.
  NIFTY loses the most (37% overnight, 12.6% vs 16.2%), because global news lands while India sleeps.
- **Yang-Zhang** adds the overnight gap back and lands within about ±10% of close-to-close on every
  series, keeping most of the noise reduction.

## Quick start

```bash
uv sync
uv run python scripts/fetch_data.py   # Yahoo Finance -> data/prices.parquet
uv run rv-report rolling              # or: estimators. Charts -> out/, numbers printed
uv run python scripts/build_site.py   # dashboard -> site/index.html
uv run pytest
```

## Design

- `src/rv/data.py`: Yahoo OHLC + adjusted closes in one (field, symbol) table; flags suspicious
  adjusted-price drops.
- `src/rv/vol.py`: volatility estimators, all annualised.
- `src/rv/site.py`: the dashboard, one static Plotly page; no server, no committed data.
- `src/rv/reports.py`: README findings, reproducible with `rv-report <topic>`.
- Charts: one validated palette; README charts are animated GIFs with a static fallback for
  reduced-motion users.

MIT licence.
