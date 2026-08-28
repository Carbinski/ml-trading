"""Fetch helpers: GitHub-only, no Yahoo. Network is mocked."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from ml_trading import fetch
from ml_trading.constants import JACKOVICH_DATA_DIR, JACKOVICH_REF


def test_fetch_module_does_not_touch_yahoo():
    source = Path(fetch.__file__).read_text(encoding="utf-8")
    assert "import yfinance" not in source
    assert "from yfinance" not in source
    assert "https://query1.finance.yahoo.com" not in source
    assert "https://finance.yahoo.com" not in source
    assert "https://api.tiingo.com" not in source
    # GitHub only.
    assert "raw.githubusercontent.com" in source
    assert "github.com" in source


def test_raw_csv_url_encodes_dollar_tickers():
    url = fetch.raw_csv_url("$DJI")
    assert JACKOVICH_REF in url
    assert "raw.githubusercontent.com" in url
    assert "yahoo" not in url.lower()
    assert "%24DJI.csv" in url or "$DJI.csv" in url


def test_fetch_symbols_writes_cache(tmp_path, monkeypatch):
    csv_bytes = (
        b"Date,Open,High,Low,Close,Volume,Adj Close\n"
        b"2010-06-10,1,1,1,1,1,1\n"
    )

    def fake_request(url, timeout=60):
        assert "yahoo" not in url.lower()
        assert "FAKE1.csv" in url
        return csv_bytes

    monkeypatch.setattr(fetch, "_request", fake_request)
    written = fetch.fetch_symbols(["FAKE1"], tmp_path)
    assert written[0].name == "FAKE1.csv"
    text = written[0].read_text()
    assert text.startswith("Date,Open,High,Low,Close,Volume,Adj Close")


def test_fetch_zipball_extracts_data_dir_only(tmp_path, monkeypatch):
    buf = io.BytesIO()
    root = f"machine-learning-for-trading-{JACKOVICH_REF}"
    data = f"{root}/{JACKOVICH_DATA_DIR}"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{data}/SPY.csv", "Date,Open,High,Low,Close,Volume,Adj Close\n")
        zf.writestr(f"{data}/ML4T-000.csv", "synthetic\n")
        zf.writestr(f"{data}/Lists/index.txt", "SPY\n")
        zf.writestr(f"{root}/README.md", "not data\n")
        zf.writestr(f"{root}/ml4t/data/SPY.csv", "duplicate tree, skip\n")

    blob = buf.getvalue()
    monkeypatch.setattr(fetch, "_request", lambda url, timeout=60: blob)

    n = fetch.fetch_zipball(tmp_path, skip_synthetic=True)
    assert n == 2
    assert (tmp_path / "SPY.csv").is_file()
    assert (tmp_path / "Lists" / "index.txt").is_file()
    assert not (tmp_path / "ML4T-000.csv").exists()
    assert not (tmp_path / "README.md").exists()


def test_list_remote_csv_names(monkeypatch):
    tree = {
        "tree": [
            {
                "path": f"{JACKOVICH_DATA_DIR}/SPY.csv",
                "type": "blob",
            },
            {
                "path": f"{JACKOVICH_DATA_DIR}/FAKE2.csv",
                "type": "blob",
            },
            {
                "path": f"{JACKOVICH_DATA_DIR}/Lists/sp5002012.txt",
                "type": "blob",
            },
            {
                "path": "ml4t/data/SPY.csv",
                "type": "blob",
            },
        ]
    }
    monkeypatch.setattr(fetch, "_request", lambda url, timeout=60: json.dumps(tree).encode())
    names = fetch.list_remote_csv_names()
    assert names == ["FAKE2", "SPY"]
