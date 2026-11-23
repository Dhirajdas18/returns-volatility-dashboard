"""`rv-report <topic>`: build a topic's charts into out/ and print its key numbers."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rv import plots
from rv.data import INDEX, SYMBOLS, VIX, load, load_table
from rv.forecast import evaluate, forecasts, realised_forward, scoreboard
from rv.implied import premium, regime_returns, rolling_percentile
from rv.perf import cagr, max_drawdown, monthly_table, rolling_sharpe, sharpe, summary
from rv.relate import (
    REGIMES,
    beta,
    beta_forecast_errors,
    label_regimes,
    regime_corr,
    rolling_beta,
    run_lengths,
    transition_matrix,
)
from rv.returns import log_returns
from rv.target import apply_weights, compare, static_match, vol_target_weights
from rv.vol import ewma_vol, garch_fit, garman_klass, parkinson, rolling_vol, yang_zhang

OUT = Path("out")
WINDOWS = {20: plots.BLUE_RAMP[2], 60: plots.BLUE_RAMP[4], 252: plots.BLUE_RAMP[6]}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _label_ends(ax, lines: dict, colors: dict, gap: float = 4) -> None:
    """Direct labels at each line's right end, nudged apart so none overlap."""
    floor = -1e9
    for name in sorted(lines, key=lambda k: lines[k].iloc[-1]):
        floor = max(lines[name].iloc[-1], floor + gap)
        plots.label_line_end(ax, lines[name].index[-1], floor, name, colors[name])


def report_rolling(out_dir: Path) -> dict:
    """Rolling volatility with 20/60/252-day windows on NIFTY around the 2020 crash."""
    r = log_returns(load("adj_close", [INDEX])[INDEX])
    vols = {w: rolling_vol(r, w) for w in WINDOWS}
    span = slice("2019-10-01", "2021-09-30")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    shown = {f"{w}-day": vols[w][span] * 100 for w in WINDOWS}
    for v, color in zip(shown.values(), WINDOWS.values(), strict=True):
        ax.plot(v.index, v, color=color)
    _label_ends(ax, shown, dict(zip(shown, WINDOWS.values(), strict=True)))
    ax.set_title("Longer windows are smoother but react months late (NIFTY, 2020)")
    ax.set_ylabel("annualised volatility, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plots.save(fig, out_dir, "rolling_windows")

    v20 = vols[20]
    crash = r.idxmin()
    exit_day = r.index[r.index.get_loc(crash) + 20]
    zoom = v20["2020-03-01":"2020-05-15"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(zoom.index, zoom * 100, color=WINDOWS[20])
    ax.axvline(crash, color=plots.MUTED, lw=1, ls="--")
    ax.axvline(exit_day, color=plots.MUTED, lw=1, ls="--")
    top = zoom.max() * 100
    ax.annotate(f"{crash:%d %b}: NIFTY {r.min():.0%}", (crash, top * 1.02), color=plots.TEXT_2,
                ha="right", fontsize=9)
    ax.annotate(f"{exit_day:%d %b}: that day leaves\nthe window, vol drops "
                f"{(v20[exit_day] - v20.shift(1)[exit_day]) * 100:+.0f} points",
                (exit_day, top * 0.9), color=plots.TEXT_2, xytext=(6, 0),
                textcoords="offset points", fontsize=9)
    ax.set_ylim(0, top * 1.12)
    ax.set_title("20-day vol falls off a cliff on an ordinary day (ghost effect)")
    ax.set_ylabel("annualised volatility, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plots.save(fig, out_dir, "rolling_ghost")

    out = {}
    for w in WINDOWS:
        v = vols[w][span]
        out[f"{w}d peak"] = f"{v.idxmax():%Y-%m-%d} at {_pct(v.max())}"
        out[f"{w}d median daily change"] = f"{vols[w].diff().abs().median() * 100:.2f} vol pts"
    out["ghost drop"] = (f"{exit_day:%Y-%m-%d}: {_pct(v20.shift(1)[exit_day])} -> "
                         f"{_pct(v20[exit_day])}, NIFTY moved {r[exit_day]:+.1%} that day")
    out["latest"] = {w: _pct(vols[w].iloc[-1]) for w in WINDOWS}
    return out


def _ohlc(sym: str) -> pd.DataFrame:
    fields = ["open", "high", "low", "close"]
    return pd.concat([load(f, [sym])[sym] for f in fields], axis=1, keys=fields).dropna()


def report_estimators(out_dir: Path) -> dict:
    """EWMA and GARCH vs a rolling window in 2020; GARCH forecasts; range vs close-to-close."""
    c = plots.ESTIMATOR_COLORS
    r = log_returns(load("adj_close", [INDEX])[INDEX])
    fit = garch_fit(r)
    lines = {"20-day rolling": rolling_vol(r, 20), "EWMA": ewma_vol(r), "GARCH": fit.vol}
    span = slice("2020-02-15", "2020-07-31")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    shown = {name: v[span] * 100 for name, v in lines.items()}
    for name, v in shown.items():
        ax.plot(v.index, v, color=c[name])
    _label_ends(ax, shown, c)
    ax.set_title("EWMA and GARCH react sooner and fade smoothly (NIFTY, 2020)")
    ax.set_ylabel("annualised volatility, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plots.save(fig, out_dir, "estimators_2020")

    crash = r.idxmin()
    calm = fit.vol["2017-01-01":"2019-12-31"].idxmin()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for day, color in ((crash, plots.SERIES[7]), (calm, plots.SERIES[0])):
        path = fit.forecast(126, day) * 100
        ax.plot(path.index, path, color=color)
        plots.label_line_end(ax, path.index[-1], path.iloc[-1], f"made {day:%d %b %Y}", color)
    ax.axhline(fit.long_run_vol * 100, color=plots.MUTED, lw=1, ls="--")
    ax.annotate(f"long-run {fit.long_run_vol:.0%}", (1, fit.long_run_vol * 100), xytext=(0, 4),
                textcoords="offset points", color=plots.TEXT_2, fontsize=9)
    ax.set_title(f"GARCH forecasts drift back to normal; shocks halve in ~{fit.half_life:.0f} days")
    ax.set_xlabel("trading days ahead")
    ax.set_ylabel("forecast volatility, %")
    plots.save(fig, out_dir, "garch_forecast")

    rows = {}
    for sym in [INDEX] + SYMBOLS:
        o = _ohlc(sym)
        n = len(o) - 1  # whole sample (the first day has no previous close)
        rows[sym] = {
            "Close-to-close": float(np.log(o["close"]).diff().std() * np.sqrt(252)),
            "Garman-Klass": float(garman_klass(o["open"], o["high"], o["low"], o["close"], n)
                                  .iloc[-1]),
            "Yang-Zhang": float(yang_zhang(o["open"], o["high"], o["low"], o["close"], n)
                                .iloc[-1]),
            "Parkinson": float(parkinson(o["high"], o["low"], n).iloc[-1]),
            "overnight share": float(np.log(o["open"] / o["close"].shift(1)).var()
                                     / np.log(o["close"]).diff().var()),
        }
    tab = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(tab))
    bars = {"Close-to-close": plots.MUTED, "Garman-Klass": c["Garman-Klass"],
            "Yang-Zhang": c["Yang-Zhang"]}
    for i, (name, color) in enumerate(bars.items()):
        ax.barh(y + (i - 1) * 0.26, tab[name] * 100, height=0.24, color=color, label=name)
    ax.set_yticks(y, tab.index)
    ax.invert_yaxis()
    ax.set_xlabel("annualised volatility since 2013, %")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    ax.grid(axis="y", visible=False)
    ax.set_title("High/low estimators miss overnight gaps; Yang-Zhang puts them back", pad=28)
    plots.save(fig, out_dir, "range_vs_close")

    e, v20 = lines["EWMA"][span], lines["20-day rolling"][span]
    return {
        "EWMA peak": f"{e.idxmax():%Y-%m-%d} at {_pct(e.max())} (20-day: {v20.idxmax():%Y-%m-%d})",
        "largest 1-day drop": f"EWMA {-e.diff().min() * 100:.1f} pts, 20-day "
                              f"{-v20.diff().min() * 100:.1f} pts",
        "NIFTY GARCH": f"alpha {fit.alpha:.3f}, beta {fit.beta:.3f}, persistence "
                       f"{fit.persistence:.3f}, half-life {fit.half_life:.0f}d, long-run "
                       f"{_pct(fit.long_run_vol)}",
        "half-lives": {s: round(garch_fit(log_returns(load("adj_close", [s])[s])).half_life, 1)
                       for s in SYMBOLS},
        f"forecast made {crash:%Y-%m-%d}": [_pct(x) for x in fit.forecast(126, crash).iloc[
            [0, 20, 62, 125]]],
        "range table": tab.round(3).to_dict("index"),
    }


SPLIT = "2020-01-01"


def report_forecast(out_dir: Path) -> dict:
    """Which estimator predicts the next month's volatility best, out of sample (2020 onwards)."""
    board = scoreboard(load_table(), SPLIT)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vals = board.to_numpy()
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", plots.BLUE_RAMP)
    ax.imshow(vals, cmap=cmap, vmin=1.0, vmax=max(2.0, vals.max()), aspect="auto")
    for (i, j), v in np.ndenumerate(vals):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="white" if v > 1.5 else plots.TEXT)
    ax.set_xticks(range(board.shape[1]), board.columns)
    ax.set_yticks(range(board.shape[0]), board.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Forecast error vs the best estimator, 2020–2026 (1.00 = best)")
    plots.save(fig, out_dir, "forecast_scoreboard")

    r = log_returns(load("adj_close", [INDEX])[INDEX])
    fc = forecasts(r, SPLIT, _ohlc(INDEX))
    real = realised_forward(r)
    span = slice("2020-01-01", "2020-12-31")
    shown = {"realised next month": real[span] * 100}
    colors = {"realised next month": plots.TEXT, "GARCH": plots.ESTIMATOR_COLORS["GARCH"],
              "20-day rolling": plots.MUTED}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name in ("20-day rolling", "GARCH"):
        shown[name] = fc[name][span] * 100
    for name, v in shown.items():
        ax.plot(v.index, v, color=colors[name], lw=2.6 if name.startswith("realised") else 2)
    _label_ends(ax, shown, colors)
    ax.set_title("Every forecast missed the start of the 2020 crash, then over-shot the calm")
    ax.set_ylabel("annualised volatility, %")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plots.save(fig, out_dir, "forecast_2020")

    wins = {}
    table = load_table()
    for sym in table["adj_close"].columns:
        rs = log_returns(table["adj_close"][sym].dropna())
        d = forecasts(rs, SPLIT, _ohlc(sym)).join(realised_forward(rs).rename("real")).dropna()
        d = d.loc[SPLIT:]
        loss = pd.DataFrame({c: (d.real / d[c]) ** 2 - np.log((d.real / d[c]) ** 2) - 1
                             for c in d.columns if c != "real"})
        yearly = loss.groupby(loss.index.year).mean()
        for c in yearly.columns.drop("GARCH"):
            wins.setdefault(c, []).extend((yearly["GARCH"] < yearly[c]).tolist())
    board5 = scoreboard(table, SPLIT, horizon=5)
    best_2020 = {}
    for sym in table["adj_close"].columns:
        rs = log_returns(table["adj_close"][sym].dropna())
        sc = evaluate(forecasts(rs, SPLIT, _ohlc(sym)), realised_forward(rs), SPLIT, "2020-12-31")
        best_2020[sym] = sc.index[0]
    feb = pd.Timestamp("2020-02-25")
    return {
        "5-day horizon, mean vs best": board5.mean(axis=1).round(3).to_dict(),
        "best in 2020 alone": best_2020,
        "NIFTY forecasts on 2020-02-25": {k: _pct(v) for k, v in fc.loc[feb].items()},
        "NIFTY realised next month from 2020-02-25": _pct(real.loc[feb]),
        "average rank order": list(board.index),
        "scoreboard": board.round(2).to_dict("index"),
        "GARCH wins (symbol-years)": {c: f"{sum(w)}/{len(w)}" for c, w in wins.items()},
        "test months (independent)": f"~{len(real.loc[SPLIT:].dropna()) // 21} per symbol",
    }


def _vix_halves(prices: pd.Series, pct: pd.Series) -> dict:
    out = {}
    for name, sl in (("2015-19", slice(None, "2019-12-31")),
                     ("2020-26", slice("2020-01-01", None))):
        runs = [regime_returns(prices.loc[sl], pct.loc[sl], offset=o) for o in range(21)]
        hi = [x.loc["high", "mean return"] for x in runs]
        no = [x.loc["normal", "mean return"] for x in runs]
        wins = sum(h > n for h, n in zip(hi, no, strict=True))
        out[name] = f"{np.median(hi):+.2%} / {np.median(no):+.2%}, high ahead {wins}/21"
    return out


def report_implied(out_dir: Path) -> dict:
    """India VIX vs realised NIFTY vol; VIX as a forecast; what high VIX says about returns."""
    prices = load("adj_close", [INDEX])[INDEX]
    r = log_returns(prices)
    vix = load("close", [VIX])[VIX] / 100
    real = realised_forward(r)
    prem = premium(vix, real)

    weekly = prem.resample("W").last() * 100
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(weekly.index, weekly["realised"], weekly["implied"],
                    where=weekly["implied"] >= weekly["realised"], color=plots.BLUE_RAMP[0],
                    interpolate=True, lw=0)
    ax.fill_between(weekly.index, weekly["realised"], weekly["implied"],
                    where=weekly["implied"] < weekly["realised"], color=plots.SERIES[7],
                    alpha=0.35, interpolate=True, lw=0)
    ax.plot(weekly.index, weekly["implied"], color=plots.SERIES[0], lw=1.8,
            label="India VIX (implied, at the time)")
    ax.plot(weekly.index, weekly["realised"], color=plots.TEXT, lw=1.2,
            label="NIFTY volatility over the next month")
    ax.legend(loc="upper left")
    worst = prem["spread"].idxmin()
    ax.annotate(f"{worst:%d %b %Y}: VIX {prem.loc[worst, 'implied']:.0%}, next month "
                f"{prem.loc[worst, 'realised']:.0%}", (worst, prem.loc[worst, "realised"] * 100),
                xytext=(10, -4), textcoords="offset points", color=plots.TEXT_2, fontsize=9)
    ax.set_ylim(0, None)
    ax.set_ylabel("annualised volatility, %")
    ax.set_title(f"VIX sat above the volatility that followed on {(prem.spread > 0).mean():.0%} "
                 "of days (NIFTY)")
    plots.save(fig, out_dir, "implied_vs_realised")

    train = prem.loc[: "2019-12-31"]
    scale = float((train["realised"] / train["implied"]).mean())
    fc = forecasts(r, SPLIT, _ohlc(INDEX))
    fc["India VIX"] = vix.reindex(fc.index)
    fc["VIX x train ratio"] = fc["India VIX"] * scale
    scores = evaluate(fc, real, SPLIT)
    keep = ["India VIX", "VIX x train ratio", "GARCH", "Yang-Zhang", "EWMA", "20-day rolling"]
    shown = scores.loc[[k for k in scores.index if k in keep], "QLIKE"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [plots.SERIES[0] if "VIX" in k else plots.ESTIMATOR_COLORS.get(k, plots.MUTED)
              for k in shown.index]
    ax.barh(shown.index, shown, color=colors, height=0.6)
    for i, v in enumerate(shown):
        ax.text(v, i, f" {v:.2f}", va="center", color=plots.TEXT_2, fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("QLIKE, 2020–2026 (lower is better)")
    ax.set_title("The options market forecast NIFTY's next month better than any model")
    plots.save(fig, out_dir, "vix_forecast")

    pct = rolling_percentile(vix)
    runs = [regime_returns(prices, pct, offset=o) for o in range(21)]
    med = {g: float(np.median([x.loc[g, "mean return"] for x in runs])) for g in ("high", "normal")}
    lo = {g: min(x.loc[g, "mean return"] for x in runs) for g in med}
    hi = {g: max(x.loc[g, "mean return"] for x in runs) for g in med}
    months = {g: int(runs[0].loc[g, "months"]) for g in med}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [f"VIX above its 2-year 80th percentile\n({months['high']} months)",
              f"all other days\n({months['normal']} months)"]
    vals = [med["high"] * 100, med["normal"] * 100]
    errs = [[(med[g] - lo[g]) * 100 for g in med], [(hi[g] - med[g]) * 100 for g in med]]
    ax.bar(labels, vals, yerr=errs, color=[plots.SERIES[0], plots.MUTED], width=0.5,
           capsize=6, ecolor=plots.TEXT_2)
    for i, v in enumerate(vals):
        ax.text(i + 0.27, v, f"{v:+.1f}%", va="center", color=plots.TEXT, fontsize=10)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("NIFTY return over the next month, %")
    ax.set_xlabel("bars: median across all 21 possible start days; whiskers: lowest to highest",
                  fontsize=9, color=plots.TEXT_2)
    ax.set_title("High VIX was followed by better months, not worse (2015–2026)")
    plots.save(fig, out_dir, "vix_regime")

    hi_wins = sum(x.loc["high", "mean return"] > x.loc["normal", "mean return"] for x in runs)
    worst_start = np.log(prices.shift(-21) / prices).idxmin()
    return {
        "VIX above realised": f"{(prem.spread > 0).mean():.0%} of days",
        "mean / median premium": f"{prem.spread.mean() * 100:.1f} / "
                                 f"{prem.spread.median() * 100:.1f} vol pts",
        "worst miss": f"{worst:%Y-%m-%d}: VIX {prem.loc[worst, 'implied']:.0%} vs "
                      f"{prem.loc[worst, 'realised']:.0%}",
        "train ratio realised/VIX": round(scale, 3),
        "QLIKE 2020-26": scores["QLIKE"].round(3).to_dict(),
        "bias 2020-26 (vol pts)": (scores["bias"] * 100).round(1).to_dict(),
        "average next-month return (median across 21 start days)": {
            g: f"{med[g]:+.2%} (range {lo[g]:+.2%} to {hi[g]:+.2%})" for g in med},
        "high beat normal": f"{hi_wins}/21 offsets",
        "high-VIX months across offsets": f"{min(x.loc['high', 'months'] for x in runs)}–"
                                          f"{max(x.loc['high', 'months'] for x in runs)}",
        "by half (median avg high / normal, wins)": _vix_halves(prices, pct),
        "worst month start": f"{worst_start:%Y-%m-%d}, VIX percentile then "
                             f"{pct.asof(worst_start):.2f}",
    }


RF = 0.065  # assumed risk-free rate: India 91-day T-bills ran roughly 3-7% over 2013-2026


def report_perf(out_dir: Path) -> dict:
    """Sharpe (with its uncertainty), Sortino, Calmar; rolling Sharpe; NIFTY monthly heatmap."""
    r = load("adj_close", [INDEX] + SYMBOLS).pct_change().iloc[1:]
    at_rf, at_zero = summary(r, RF), summary(r, 0.0)
    order = at_rf["Sharpe"].sort_values().index

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(order))
    ci = 1.96 * at_rf.loc[order, "Sharpe SE"]
    ax.errorbar(at_rf.loc[order, "Sharpe"], y, xerr=ci, fmt="o", color=plots.SERIES[0],
                ecolor=plots.BLUE_RAMP[2], elinewidth=6, capsize=0, ms=8,
                label=f"risk-free {RF:.1%} (95% range)")
    ax.scatter(at_zero.loc[order, "Sharpe"], y, color=plots.MUTED, s=40, zorder=3,
               label="risk-free 0% (common shortcut)")
    ax.axvline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("annualised Sharpe ratio, 2013–2026")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title("Every Sharpe ratio's 95% range includes zero after 14 years", pad=28)
    plots.save(fig, out_dir, "sharpe_ci")

    rs = rolling_sharpe(r[INDEX], 252, RF).dropna()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.fill_between(rs.index, rs, 0, where=rs >= 0, color=plots.BLUE_RAMP[1], lw=0)
    ax.fill_between(rs.index, rs, 0, where=rs < 0, color=plots.SERIES[7], alpha=0.35, lw=0)
    ax.plot(rs.index, rs, color=plots.TEXT, lw=1)
    ax.axhline(at_rf.loc[INDEX, "Sharpe"], color=plots.SERIES[0], lw=1.2, ls="--")
    plots.label_line_end(ax, rs.index[-1], at_rf.loc[INDEX, "Sharpe"],
                         f"full period {at_rf.loc[INDEX, 'Sharpe']:.2f}", plots.SERIES[0])
    ax.set_ylabel("Sharpe ratio over the past year")
    ax.set_title(f"NIFTY's one-year Sharpe swung from {rs.min():.1f} to {rs.max():.1f}")
    plots.save(fig, out_dir, "rolling_sharpe")

    m = monthly_table(r[INDEX])
    months = m.iloc[:, :12] * 100
    lim = float(np.nanmax(np.abs(months.to_numpy())))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", [plots.SERIES[7], "#f1f0ec", plots.SERIES[0]])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(months.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(months.to_numpy()):
        if not np.isnan(v):
            ax.text(j, i, f"{round(v) + 0:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > lim * 0.6 else plots.TEXT)
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                              "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(m)), m.index)
    ax.tick_params(length=0)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("NIFTY monthly returns, % (every month has losing years)")
    plots.save(fig, out_dir, "monthly_heatmap")

    up = (m.iloc[:, :12] > 0).mean()
    counts = (m.iloc[:, :12] > 0).sum()
    n_years = int(m.iloc[:, :12].notna().sum().max())
    p_up = float((m.iloc[:, :12] > 0).sum().sum() / m.iloc[:, :12].notna().sum().sum())
    extreme = int(counts.min())  # as lopsided as the weakest month (January: 4 up of 14)
    sims = np.random.default_rng(0).binomial(n_years, p_up, (100_000, 12))
    chance = float(((sims <= extreme) | (sims >= n_years - extreme)).any(axis=1).mean())
    return {
        "coin-flip chance some month is as lopsided": f"{chance:.0%} (≤{extreme} or "
                                                       f"≥{n_years - extreme} up of {n_years})",
        f"summary at rf {RF:.1%}": at_rf.round(3).to_dict("index"),
        "Sharpe at rf 0%": at_zero["Sharpe"].round(2).to_dict(),
        "95% Sharpe range": {k: f"{v.Sharpe - 1.96 * v['Sharpe SE']:+.2f} to "
                                f"{v.Sharpe + 1.96 * v['Sharpe SE']:+.2f}"
                             for k, v in at_rf.iterrows()},
        "rank by": {c: list(at_rf[c].sort_values(ascending=False).index)
                    for c in ("CAGR", "Sharpe", "Sortino", "Calmar")},
        "rolling 1y Sharpe": f"{rs.min():.2f} ({rs.idxmin():%Y-%m}) to {rs.max():.2f} "
                             f"({rs.idxmax():%Y-%m}); below 0 {(rs < 0).mean():.0%} of the time",
        "share of years up, by month": up.round(2).to_dict(),
        "years up": f"{(m['Year'] > 0).sum()}/{len(m)}",
    }


REGIME_COLORS = {"calm": plots.BLUE_RAMP[0], "normal": "#ecebe6", "stressed": "#f6c9c4"}


def report_relate(out_dir: Path) -> dict:
    """Beta vs volatility, rolling beta, forecasting next year's beta, volatility regimes."""
    r = load("adj_close", [INDEX] + SYMBOLS).pct_change().iloc[1:]
    m = r[INDEX]
    stats = pd.DataFrame({s: {"beta": beta(r[s], m), "corr": r[s].corr(m),
                              "vol ratio": r[s].std() / m.std()} for s in SYMBOLS}).T

    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.linspace(1.3, 1.85, 50)
    for c in (0.4, 0.6, 0.8):
        ax.plot(xs, c * xs, color=plots.GRID, lw=1.2, zorder=1)
        ax.annotate(f"correlation {c}", (xs[-1], c * xs[-1]), xytext=(4, 0),
                    textcoords="offset points", color=plots.TEXT_2, fontsize=8, va="center")
    for sym, row in stats.iterrows():
        ax.scatter(row["vol ratio"], row["beta"], s=90, color=plots.COLORS[sym], zorder=3,
                   edgecolor=plots.SURFACE, linewidth=2)
        ax.annotate(sym, (row["vol ratio"], row["beta"]), xytext=(8, -3),
                    textcoords="offset points", color=plots.TEXT, fontsize=9)
    ax.set_xlabel("volatility relative to NIFTY (×)")
    ax.set_ylabel("beta to NIFTY")
    ax.set_title("A volatile stock isn't a high-beta stock: beta = correlation × vol ratio")
    plots.save(fig, out_dir, "beta_vs_vol")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    lines = {}
    for sym in SYMBOLS:
        rb = rolling_beta(r[sym], m, 252).dropna()
        ax.plot(rb.index, rb, color=plots.COLORS[sym], lw=1.6)
        lines[sym] = rb
    ax.axhline(1, color=plots.TEXT_2, lw=0.8, ls="--")
    _label_ends(ax, lines, plots.COLORS, gap=0.07)
    ax.set_ylabel("beta over the past year")
    ax.set_title("One-year betas wander: TCS ranged from 0.22 to 1.05")
    plots.save(fig, out_dir, "rolling_beta")

    errs = pd.concat({s: beta_forecast_errors(r[s], m) for s in SYMBOLS}).groupby(level=1).mean()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for col, color in (("raw", plots.MUTED), ("shrunk to 1", plots.SERIES[0])):
        ax.plot(errs.index, errs[col], marker="o", color=color, ms=7)
        plots.label_line_end(ax, errs.index[-1], errs[col].iloc[-1], col, color)
    ax.set_xscale("log")
    ax.set_xticks(errs.index, [f"{w // 21}m" if w < 252 else f"{w // 252}y" for w in errs.index])
    ax.minorticks_off()
    ax.set_xlabel("look-back window used to measure beta")
    ax.set_ylabel("average error vs next year's beta")
    ax.set_title("Predicting next year's beta: use 3 years, or shrink a short estimate to 1")
    plots.save(fig, out_dir, "beta_forecast")

    vol = ewma_vol(np.log(1 + m))
    lab = label_regimes(vol)
    known = lab.shift(1)  # the regime as known at yesterday's close: a day can't label itself
    corr = regime_corr(r[SYMBOLS], known).reindex(REGIMES)
    nx = pd.concat([m, known], axis=1, keys=["r", "lab"]).dropna()
    ann = nx.groupby("lab")["r"].agg(["mean", "std"]).reindex(REGIMES)
    ret_per_risk = ann["mean"] * 252 / (ann["std"] * np.sqrt(252))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    panels = ((axes[0], corr, "average correlation between the 5 stocks"),
              (axes[1], ret_per_risk, "NIFTY return per unit of risk (regime known a day ahead)"))
    for ax, vals, title in panels:
        ax.bar(REGIMES, vals, color=[REGIME_COLORS[g] for g in REGIMES], edgecolor=plots.TEXT_2,
               linewidth=0.6, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", color=plots.TEXT, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="normal", color=plots.TEXT_2)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Stressed regimes: stocks move together, but returns per risk hold up", x=0.06,
                 ha="left", fontsize=13, fontweight="semibold", color=plots.TEXT)
    plots.save(fig, out_dir, "regimes")

    runs = run_lengths(lab)
    shares = lab.value_counts(normalize=True).reindex(REGIMES)
    return {
        "beta / corr / vol ratio": stats.round(2).to_dict("index"),
        "1y beta range": {s: f"{rolling_beta(r[s], m).min():.2f} to "
                             f"{rolling_beta(r[s], m).max():.2f}" for s in SYMBOLS},
        "beta forecast error (MAE)": errs.round(3).to_dict("index"),
        "regime share of days": shares.round(2).to_dict(),
        "stay-in-regime next day": {g: round(float(transition_matrix(lab).loc[g, g]), 3)
                                    for g in REGIMES},
        "run length median/mean/max (days)": {g: (int(np.median(v)), round(float(np.mean(v)), 1),
                                                  int(max(v))) for g, v in runs.items()},
        "pairwise corr by regime": corr.round(2).to_dict(),
        "NIFTY ann. return / vol by regime": {g: f"{ann.loc[g, 'mean'] * 252:.1%} / "
                                                 f"{ann.loc[g, 'std'] * np.sqrt(252):.1%}"
                                              for g in REGIMES},
        "return per risk by regime": ret_per_risk.round(2).to_dict(),
    }


COST = 0.001  # 0.1% of the amount traded each time the weight changes (assumption)
TARGET = 0.15


def _vt(r: pd.Series, target: float):
    """Vol-target r with EWMA forecasts (no leverage) and its same-exposure static benchmark."""
    vol = ewma_vol(np.log(1 + r))
    vt = apply_weights(r, vol_target_weights(vol, target), RF, COST)
    return vt, static_match(r, vt["weight"], RF, COST)


def report_target(out_dir: Path) -> dict:
    """Vol targeting on NIFTY (no leverage) vs buy-and-hold and a same-exposure static mix."""
    r = load("adj_close", [INDEX]).pct_change().dropna()[INDEX]
    hold = apply_weights(r, pd.Series(1.0, index=r.index), RF, COST)
    vt, static = _vt(r, TARGET)
    vix = load("close", [VIX])[VIX].reindex(r.index).ffill() / 100
    rule_w = pd.Series(np.where(rolling_percentile(vix) > 0.8, 0.5, 1.0), index=r.index)
    rule = apply_weights(r, rule_w, RF, COST)
    table = compare({"buy and hold": hold, f"vol target {TARGET:.0%}": vt,
                     "static, same exposure": static, "VIX rule (halve > 80th pct)": rule}, RF)
    colors = {"buy and hold": plots.MUTED, f"vol target {TARGET:.0%}": plots.SERIES[0],
              "static, same exposure": plots.TEXT}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    curves = {}
    for name, df in (("buy and hold", hold), ("static, same exposure", static),
                     (f"vol target {TARGET:.0%}", vt)):  # vol target drawn last, on top
        wealth = (1 + df["return"]).cumprod() * 1e5
        ax.plot(wealth.index, wealth / 1e5, color=colors[name], lw=1.8)
        curves[name] = wealth / 1e5
        dd = wealth / wealth.cummax() - 1
        ax2.plot(dd.index, dd * 100, color=colors[name], lw=1.2)
    _label_ends(ax, curves, colors, gap=0.25)
    ax.set_ylabel("₹ lakh (started with 1)")
    ax2.set_ylabel("drawdown, %")
    ax.set_title("Vol targeting cut the 2020 drawdown but lagged on growth (NIFTY, after costs)")
    plots.save(fig, out_dir, "vt_equity")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    mix = []
    for w in np.linspace(0, 1, 21):
        m = apply_weights(r, pd.Series(w, index=r.index), RF, COST, start_weight=w)["return"]
        mix.append((m.std() * np.sqrt(252) * 100, (cagr(m)) * 100))
    mx = np.array(mix)
    ax.plot(mx[:, 0], mx[:, 1], color=plots.TEXT, lw=1.6)
    ax.annotate("NIFTY + cash, fixed mix", (mx[-4, 0], mx[-4, 1]), xytext=(-10, 10),
                textcoords="offset points", color=plots.TEXT_2, fontsize=9, ha="right")
    pts = []
    for tgt in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20):
        v, _ = _vt(r, tgt)
        x = v["return"]
        pts.append((x.std() * np.sqrt(252) * 100, cagr(x) * 100, tgt))
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], "o-", color=plots.SERIES[0], ms=7)
    for vx, vy, tgt in pts:
        ax.annotate(f"{tgt:.0%}", (vx, vy), xytext=(0, -14), textcoords="offset points",
                    ha="center", color=plots.SERIES[0], fontsize=8)
    ax.annotate("vol targets", (pts[0, 0], pts[0, 1]), xytext=(-8, 6), textcoords="offset points",
                color=plots.SERIES[0], fontsize=9, ha="right")
    ax.set_xlabel("volatility, % a year")
    ax.set_ylabel("growth, % a year (after costs)")
    ax.set_title("At every risk level, a fixed mix earned more than vol targeting")
    plots.save(fig, out_dir, "vt_frontier")

    rows = {}
    for sym in [INDEX] + SYMBOLS:
        rs = load("adj_close", [sym]).pct_change().dropna()[sym]
        v, st = _vt(rs, float(rs.std() * np.sqrt(252) * 0.9))
        rows[sym] = {"vt Sharpe": sharpe(v["return"], RF),
                     "static Sharpe": sharpe(st["return"], RF),
                     "vt DD": max_drawdown(v["return"]),
                     "static DD": max_drawdown(st["return"])}
    per = pd.DataFrame(rows).T
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    y = np.arange(len(per))
    for ax, (a, b, title) in zip(axes, (("vt Sharpe", "static Sharpe", "Sharpe ratio"),
                                        ("vt DD", "static DD", "max drawdown, %")), strict=True):
        scale = 100 if "DD" in a else 1
        ax.barh(y - 0.2, per[a] * scale, height=0.38, color=plots.SERIES[0], label="vol target")
        ax.barh(y + 0.2, per[b] * scale, height=0.38, color=plots.MUTED,
                label="static, same exposure")
        ax.set_title(title, fontsize=10, fontweight="normal", color=plots.TEXT_2)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, per.index)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Every series: smaller drawdowns; Sharpe lower in 4 of 6", x=0.06, ha="left",
                 fontsize=13, fontweight="semibold", color=plots.TEXT)
    plots.save(fig, out_dir, "vt_stocks")

    halves = {}
    for name, sl in (("2013-19", slice(None, "2019-12-31")),
                     ("2020-26", slice("2020-01-01", None))):
        halves[name] = (f"{sharpe(vt['return'].loc[sl], RF):.2f} vs "
                        f"{sharpe(static['return'].loc[sl], RF):.2f}")
    vol0 = ewma_vol(np.log(1 + r))
    vt0 = apply_weights(r, vol_target_weights(vol0, TARGET), RF, 0.0)
    st0 = static_match(r, vt0["weight"], RF, 0.0)

    def dd(x):
        w = (1 + x).cumprod()
        return w / w.cummax() - 1

    dv, ds = dd(vt["return"]), dd(static["return"])
    yv, ys = dv.groupby(dv.index.year).min(), ds.groupby(ds.index.year).min()
    return {
        "Sharpe by half (vt vs static)": halves,
        "Sharpe at zero cost (vt vs static)": f"{sharpe(vt0['return'], RF):.3f} vs "
                                              f"{sharpe(st0['return'], RF):.3f}",
        "years vt worst drawdown same or deeper": f"{int((yv <= ys + 1e-9).sum())}/{len(yv)}",
        "mean drawdown vt / static": f"{dv.mean():.1%} / {ds.mean():.1%}",
        "turnover VIX rule / vt": round(
            table.loc["VIX rule (halve > 80th pct)", "Turnover / yr"]
            / table.loc[f"vol target {TARGET:.0%}", "Turnover / yr"], 2),
        "NIFTY table": table.round(3).to_dict("index"),
        "per series (target = 0.9 x own vol)": per.round(3).to_dict("index"),
        "frontier vol-target points (vol, cagr, target)": pts.round(2).tolist(),
    }


REPORTS = {"rolling": report_rolling, "estimators": report_estimators, "forecast": report_forecast,
           "implied": report_implied, "perf": report_perf,
           "relate": report_relate, "target": report_target}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in REPORTS:
        print(f"usage: rv-report <topic>   (topics: {', '.join(REPORTS)})")
        return 2
    plots.style()
    result = REPORTS[args[0]](OUT)
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"charts -> {OUT.resolve()}")
    return 0
