"""SEC insider data set parsing (D-026) on a zip in the SEC's layout."""

from datetime import date

import pandas as pd
import pytest

from quantlab.data import sec_insider
from insider_fixtures import write_quarter_zip


def test_quarter_list_stops_at_the_last_ended_quarter():
    assert sec_insider.quarter_list("2025q3", date(2026, 4, 1)) == ["2025q3", "2025q4", "2026q1"]
    assert sec_insider.quarter_list("2025q4", date(2026, 3, 31)) == ["2025q4"]


def test_user_agent_needs_an_email():
    assert sec_insider.user_agent("a@b.co").endswith("a@b.co")
    with pytest.raises(ValueError):
        sec_insider.user_agent("nobody")


def test_parse_quarter_keeps_original_form4_open_market_purchases(tmp_path):
    p = tmp_path / "2012q1_form345.zip"
    write_quarter_zip(p, [
        {"acc": "1", "date": "2012-02-03", "cik": "000123", "symbol": "brk.b",
         "owners": [("9", "Director"), ("8", "Officer,TenPercentOwner")],
         "trans": [("P", "A", 100, 10.5), ("P", "A", 50, 11), ("S", "D", 10, 12), ("A", "A", 999, 0)]},
        {"acc": "2", "date": "2012-02-06", "doc": "4/A", "cik": "123", "symbol": "BRK.B",
         "owners": [("9", "Director")], "trans": [("P", "A", 100, 10)]},
        {"acc": "3", "date": "2012-02-07", "cik": "123", "symbol": "BRK.B",
         "owners": [("7", "TenPercentOwner")], "trans": [("S", "D", 100, 10)]},
    ])
    rows, fil, counts = sec_insider.parse_quarter(p)
    assert counts["form4"] == 2 and counts["purchase_filings"] == 1 and counts["sale_filings"] == 2
    sales = rows[rows["side"] == "sell"].set_index(["accession", "owner"])
    assert sales.loc[("1", "9"), "value"] == pytest.approx(120) and sales.loc[("3", "7"), "value"] == pytest.approx(1000)
    pur = rows[rows["side"] == "buy"]
    assert list(pur["owner"]) == ["8", "9"] or list(pur["owner"]) == ["9", "8"]
    row = pur.set_index("owner")
    assert row.loc["9", "value"] == pytest.approx(100 * 10.5 + 50 * 11)
    assert row.loc["9", "director"] and not row.loc["9", "officer"] and row.loc["8", "officer"]
    assert (pur["known"] == pd.Timestamp("2012-02-03")).all() and (pur["symbol"] == "BRK-B").all()
    assert (pur["issuer_cik"] == "123").all()
    assert set(fil["symbol"]) == {"BRK-B"} and int(fil["n"].sum()) == 2


def test_ticker_base_strips_only_the_delisting_suffix():
    assert sec_insider.ticker_base("AAL-199702") == "AAL"
    assert sec_insider.ticker_base("BRK-B") == "BRK-B"
