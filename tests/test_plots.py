import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rv import plots


def test_ease_out_endpoints_and_monotonic():
    p = np.linspace(0, 1, 101)
    v = [plots.ease_out(x) for x in p]
    assert v[0] == 0 and abs(v[-1] - 1) < 1e-9
    assert all(b >= a for a, b in zip(v, v[1:], strict=False))
    assert plots.ease_out(0.2) > 0.5  # strong ease-out: most of the motion happens early


def test_save_writes_png_and_looping_boomerang_gif(tmp_path):
    plots.style()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [1, 3, 2])
    ax.annotate("label", (2, 2))
    png = plots.save(fig, tmp_path, "demo")
    gif = tmp_path / "demo.gif"
    assert png.exists() and gif.exists()
    with Image.open(gif) as im:
        assert im.info.get("loop") == 0
        n = im.n_frames
        durations = []
        for i in range(n):
            im.seek(i)
            durations.append(im.info["duration"])
    # Identical consecutive frames (the tail of the ease-out) are merged, durations summed.
    assert 10 < n <= plots.IN_FRAMES + 1 + plots.OUT_FRAMES + 1
    assert max(durations) >= plots.HOLD_MS
    assert abs(sum(durations) - plots.LOOP_MS) <= 10 * n


def test_save_can_skip_animation(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plots.save(fig, tmp_path, "still", animate=False)
    assert not (tmp_path / "still.gif").exists()
