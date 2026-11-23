"""Build site/index.html from data/prices.parquet."""

from pathlib import Path

from rv.data import INDEX, SYMBOLS, VIX, load_table
from rv.site import build_html


def main() -> None:
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(load_table([INDEX] + SYMBOLS + [VIX])), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
