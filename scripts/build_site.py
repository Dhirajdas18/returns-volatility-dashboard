"""Build site/index.html from data/prices.parquet."""

from pathlib import Path

from rv.data import INDEX, SYMBOLS, load
from rv.site import build_html


def main() -> None:
    prices = load("adj_close", [INDEX] + SYMBOLS)
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(prices), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
