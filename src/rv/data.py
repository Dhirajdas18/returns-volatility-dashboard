"""Daily OHLC + adjusted closes from Yahoo Finance, stored as one parquet table.

Columns are a (field, symbol) MultiIndex. Fields: open, high, low, close, adj_close.
India VIX only has a meaningful close.
"""

from pathlib import Path

import pandas as pd

SYMBOLS = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ITC"]
INDEX = "NIFTY"
VIX = "INDIAVIX"
START = "2013-01-01"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "prices.parquet"
FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close"}


def fetch_yfinance(start: str = START, end: str | None = None) -> pd.DataFrame:
    """Download every symbol. Yahoo's open/high/low/close are split-adjusted, not dividend-adjusted;
    that is fine for intraday ranges (ratios within one day). Returns use adj_close."""
    import yfinance as yf

    tickers = {f"{s}.NS": s for s in SYMBOLS} | {"^NSEI": INDEX, "^INDIAVIX": VIX}
    # threads=False: parallel downloads can lock yfinance's sqlite cache and drop a ticker
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=False, progress=False,
                      threads=False)
    raw = raw[list(FIELDS)].rename(columns=FIELDS, level=0).rename(columns=tickers, level=1)
    raw.columns.names = ["field", "symbol"]
    raw.index = pd.DatetimeIndex(raw.index.tz_localize(None), name="date")
    return raw.sort_index(axis=1)


def missing_symbols(table: pd.DataFrame, expected: list[str] | None = None) -> list[str]:
    """Symbols with no adjusted close at all: an all-empty column or one Yahoo left out.

    `expected` defaults to every symbol we download (the stocks, NIFTY and India VIX).
    """
    adj = table["adj_close"] if "adj_close" in table.columns.get_level_values(0) else pd.DataFrame()
    want = expected if expected is not None else SYMBOLS + [INDEX, VIX]
    return [s for s in want if s not in adj.columns or adj[s].isna().all()]


def load_table(symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """The whole (field, symbol) table, limited to `symbols` (default: the stocks + NIFTY)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    keep = symbols or [INDEX] + SYMBOLS
    table = pd.read_parquet(path)
    table = table.loc[:, table.columns.get_level_values("symbol").isin(keep)]
    return table.reindex(columns=keep, level="symbol").sort_index()


def load(field: str, symbols: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """One field as a wide date x symbol frame.

    Keeps only rows where every requested symbol (all, if None) has data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    table = pd.read_parquet(path)
    if field not in table.columns.get_level_values("field"):
        raise KeyError(f"field not in price table: {field}")
    df = table[field]
    if symbols is not None:
        missing = [s for s in symbols if s not in df.columns]
        if missing:
            raise KeyError(f"symbols not in price table: {missing}")
        df = df[symbols]
    df.columns.name = None
    return df.sort_index().dropna(how="any")


def flag_adjusted_drops(
    adj: pd.DataFrame, threshold: float = 0.2
) -> list[tuple[str, pd.Timestamp]]:
    """(symbol, date) pairs where an adjusted price falls more than `threshold` in a day.

    Large caps rarely do that for real; each hit is worth checking for an unadjusted split or bonus.
    """
    moves = adj.pct_change().stack()
    hits = moves[moves < -threshold]
    return [(sym, date) for date, sym in hits.index]
