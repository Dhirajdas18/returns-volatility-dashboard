"""`rv-report <topic>`: build a topic's charts into out/ and print its key numbers."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rv import plots
from rv.data import INDEX, SYMBOLS, load, load_table
from rv.forecast import forecasts, realised_forward, scoreboard
from rv.returns import log_returns
from rv.vol import ewma_vol, garch_fit, garman_klass, rolling_vol, yang_zhang

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
    return {
        "average rank order": list(board.index),
        "scoreboard": board.round(2).to_dict("index"),
        "GARCH wins (symbol-years)": {c: f"{sum(w)}/{len(w)}" for c, w in wins.items()},
        "test months (independent)": f"~{len(real.loc[SPLIT:].dropna()) // 21} per symbol",
    }


REPORTS = {"rolling": report_rolling, "estimators": report_estimators, "forecast": report_forecast}


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
