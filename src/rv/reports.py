"""`rv-report <topic>`: build a topic's charts into out/ and print its key numbers."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from rv import plots
from rv.data import INDEX, load
from rv.returns import log_returns
from rv.vol import rolling_vol

OUT = Path("out")
WINDOWS = {20: plots.BLUE_RAMP[2], 60: plots.BLUE_RAMP[4], 252: plots.BLUE_RAMP[6]}


def _pct(x: float) -> str:
    return f"{x:.0%}"


def report_rolling(out_dir: Path) -> dict:
    """Rolling volatility with 20/60/252-day windows on NIFTY around the 2020 crash."""
    r = log_returns(load("adj_close", [INDEX])[INDEX])
    vols = {w: rolling_vol(r, w) for w in WINDOWS}
    span = slice("2019-10-01", "2021-09-30")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ends = {}
    for w, color in WINDOWS.items():
        v = vols[w][span]
        ax.plot(v.index, v * 100, color=color)
        ends[w] = v.iloc[-1] * 100
    floor = -1e9
    for w in sorted(ends, key=ends.get):  # nudge labels apart so none overlap
        floor = max(ends[w], floor + 4)
        plots.label_line_end(ax, vols[w][span].index[-1], floor, f"{w}-day", WINDOWS[w])
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


REPORTS = {"rolling": report_rolling}


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
