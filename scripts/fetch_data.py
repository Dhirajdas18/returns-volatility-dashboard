"""Build data/prices.parquet from Yahoo Finance (no setup needed)."""

import sys
import time
from pathlib import Path

from rv.data import (
    DATA_PATH,
    INDEX,
    SYMBOLS,
    fetch_yfinance,
    flag_adjusted_drops,
    missing_symbols,
)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    df = fetch_yfinance()
    for _ in range(3):  # Yahoo occasionally drops a ticker; retry before giving up
        if not missing_symbols(df):
            break
        time.sleep(5)
        df = fetch_yfinance()
    if missing := missing_symbols(df):
        sys.exit(f"download failed for {missing}; not writing an incomplete table")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    span = f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    print(f"Yahoo Finance: {len(df)} rows, {span} -> {out}")
    for sym, date in flag_adjusted_drops(df["adj_close"][SYMBOLS + [INDEX]]):
        print(f"  check {sym} {date:%Y-%m-%d}: adjusted price fell >20% in a day")


if __name__ == "__main__":
    main()
