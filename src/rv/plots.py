"""Shared chart style. Colours follow the entity (a stock keeps its colour in every chart)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
# Categorical slots in fixed order (validated: CVD + normal-vision separation pass).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLORS = {
    "RELIANCE": SERIES[0],
    "HDFCBANK": SERIES[1],
    "INFY": SERIES[2],
    "TCS": SERIES[3],
    "ITC": SERIES[4],
    "NIFTY": SERIES[5],
}
MUTED = "#a3a29c"
# Estimators (validated 4-slot set; contrast WARN, so always direct-label or legend these lines).
ESTIMATOR_COLORS = {
    "20-day rolling": MUTED,
    "EWMA": SERIES[1],
    "GARCH": SERIES[6],
    "Yang-Zhang": SERIES[2],
    "Garman-Klass": SERIES[3],
}
# Sequential blue ramp (light to dark) for magnitudes such as correlation.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# Dark mode: the same hues stepped for a dark surface (reference palette's dark column).
DARK = {
    SURFACE: "#1a1a19",
    TEXT: "#f0efec",
    TEXT_2: "#c3c2b7",
    GRID: "#2e2d2b",
    "#f1f0ec": "#2e2d2b",  # diverging midpoint
    "white": "#242422",
    SERIES[0]: "#3987e5",
    SERIES[1]: "#d95926",
    SERIES[2]: "#199e70",
    SERIES[3]: "#c98500",
    SERIES[4]: "#d55181",
    SERIES[5]: "#0ca30c",
    SERIES[6]: "#9085e9",
    SERIES[7]: "#e66767",
}
# Line colours only: on dark, longer rolling windows get lighter (still ordered by length).
# Kept out of DARK so heatmap colour scales built from the same ramp stay in order.
DARK_LINES = {
    BLUE_RAMP[2]: "#256abf",
    BLUE_RAMP[4]: "#6da7ec",
    BLUE_RAMP[6]: "#cde2fb",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_2,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def label_line_end(ax, x, y, text: str, color: str) -> None:
    """Direct label at a line's right end (identity never by colour alone)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 1},
    )


# Boomerang reveal for README GIFs: data sweeps in left to right, holds long enough to read,
# sweeps back out faster, short pause, loop. Strong ease-out both ways (never ease-in).
FPS = 25
IN_FRAMES = 35  # 1.4 s
OUT_FRAMES = 20  # 0.8 s
HOLD_MS = 8000
PAUSE_MS = 300
FRAME_MS = 1000 // FPS
LOOP_MS = (IN_FRAMES + OUT_FRAMES) * FRAME_MS + HOLD_MS + PAUSE_MS
GIF_DPI = 90
_EASE = (0.23, 1.0, 0.32, 1.0)  # cubic-bezier(0.23, 1, 0.32, 1)


def ease_out(p: float) -> float:
    """Strong ease-out: cubic-bezier(0.23, 1, 0.32, 1) evaluated at progress p in [0, 1]."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    x1, y1, x2, y2 = _EASE

    def bez(t, a, b):
        return 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t**2 * b + t**3

    lo, hi = 0.0, 1.0
    for _ in range(40):  # solve x(t) = p
        mid = (lo + hi) / 2
        if bez(mid, x1, x2) < p:
            lo = mid
        else:
            hi = mid
    return bez((lo + hi) / 2, y1, y2)


def _frame(fig, bbox):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=GIF_DPI, bbox_inches=bbox)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_gif(fig, path: Path) -> None:
    from matplotlib.transforms import Bbox, TransformedBbox
    from PIL import Image

    # Crop every frame to the finished chart's tight box, same as the PNG (stable frame size).
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    clips, texts = [], []
    for ax in fig.axes:
        # A live clip box in axes coordinates; widening it reveals the data left to right.
        box = Bbox([[0, 0], [0, 1]])
        clip = TransformedBbox(box, ax.transAxes)
        for art in [*ax.lines, *ax.collections, *ax.patches, *ax.images]:
            art.set_clip_box(clip)
            art.set_clip_on(True)
        clips.append(box)
        texts += [(t, t.get_alpha() if t.get_alpha() is not None else 1.0) for t in ax.texts]

    def at(progress: float):
        for box in clips:
            box.x1 = max(progress, 1e-6)
        fade = min(max((progress - 0.8) / 0.2, 0.0), 1.0)  # labels fade in over the last 20%
        for t, alpha in texts:
            t.set_alpha(alpha * fade)
            if getattr(t, "arrow_patch", None) is not None:
                t.arrow_patch.set_alpha(alpha * fade)
            if t.get_bbox_patch() is not None:
                t.get_bbox_patch().set_alpha(fade)
        return _frame(fig, bbox)

    frames = [at(ease_out(i / IN_FRAMES)) for i in range(1, IN_FRAMES + 1)]
    durations = [FRAME_MS] * IN_FRAMES
    frames.append(at(1.0))
    durations.append(HOLD_MS)
    frames += [at(1 - ease_out(i / OUT_FRAMES)) for i in range(1, OUT_FRAMES + 1)]
    durations += [FRAME_MS] * OUT_FRAMES
    frames.append(at(0.0))
    durations.append(PAUSE_MS)
    palette = frames[IN_FRAMES].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)


def save(fig, out_dir: Path, name: str, animate: bool = True) -> Path:
    """Save a PNG, plus an animated GIF version for the README unless animate=False."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    if animate:
        _save_gif(fig, out_dir / f"{name}.gif")
    plt.close(fig)
    return path
