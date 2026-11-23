"""Build site/index.html from data/prices.parquet."""

from pathlib import Path

import pandas as pd

from rv.data import DATA_PATH, INDEX, SYMBOLS
from rv.site import build_html


def main() -> None:
    table = pd.read_parquet(DATA_PATH)
    keep = [INDEX] + SYMBOLS
    table = table.loc[:, table.columns.get_level_values("symbol").isin(keep)]
    table = table.reindex(columns=keep, level="symbol")
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(table), encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
