import pandas as pd
import pytest

from rv.data import flag_adjusted_drops, load


def _table(tmp_path):
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-03"], name="date")
    cols = pd.MultiIndex.from_tuples(
        [("adj_close", "NIFTY"), ("adj_close", "ITC"), ("high", "NIFTY"), ("close", "INDIAVIX")],
        names=["field", "symbol"],
    )
    df = pd.DataFrame(
        [[2.0, 20.0, 2.1, 15.0], [1.0, 10.0, 1.1, 14.0], [3.0, None, 3.1, 16.0]],
        index=idx, columns=cols,
    )
    path = tmp_path / "prices.parquet"
    df.to_parquet(path)
    return path


def test_load_missing_file_points_to_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load("adj_close", path=tmp_path / "nope.parquet")


def test_load_field_returns_wide_sorted_frame(tmp_path):
    out = load("adj_close", path=_table(tmp_path))
    assert list(out.columns) == ["NIFTY", "ITC"]
    assert out.index.is_monotonic_increasing
    assert out["NIFTY"].tolist() == [1.0, 2.0]  # 2020-01-03 has no ITC price


def test_load_symbol_subset_drops_rows_missing_only_those(tmp_path):
    out = load("adj_close", ["NIFTY"], path=_table(tmp_path))
    assert len(out) == 3
    both = load("adj_close", ["NIFTY", "ITC"], path=_table(tmp_path))
    assert len(both) == 2


def test_load_unknown_field_or_symbol(tmp_path):
    path = _table(tmp_path)
    with pytest.raises(KeyError):
        load("volume", path=path)
    with pytest.raises(KeyError):
        load("adj_close", ["TCS"], path=path)


def test_flag_adjusted_drops_catches_missed_bonus():
    adj = pd.DataFrame({"ITC": [300.0, 301.0, 201.0], "TCS": [10.0, 10.1, 10.2]},
                       index=pd.date_range("2016-06-29", periods=3, freq="D"))
    assert flag_adjusted_drops(adj) == [("ITC", pd.Timestamp("2016-07-01"))]


def test_missing_symbols_lists_symbols_with_no_prices():
    from rv.data import missing_symbols

    cols = pd.MultiIndex.from_tuples([("adj_close", "ITC"), ("adj_close", "TCS")],
                                     names=["field", "symbol"])
    df = pd.DataFrame([[1.0, None], [2.0, None]], columns=cols)
    assert missing_symbols(df, expected=["ITC", "TCS"]) == ["TCS"]


def test_missing_symbols_also_catches_columns_yahoo_left_out():
    from rv.data import missing_symbols

    cols = pd.MultiIndex.from_tuples([("adj_close", "ITC")], names=["field", "symbol"])
    df = pd.DataFrame([[1.0], [2.0]], columns=cols)
    assert missing_symbols(df, expected=["ITC", "TCS"]) == ["TCS"]
