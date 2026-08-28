"""Cache the Jackovich ML4T homework CSVs from GitHub. Not a live vendor.

This module talks only to GitHub (API + raw/zipball). Do not add yfinance,
Yahoo Finance, query1.finance.yahoo.com, Tiingo, or any brokerage fetch here.
Carson pulls Yahoo/yfinance himself for personal research; see docs/data.md.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from ml_trading.constants import (
    JACKOVICH_DATA_DIR,
    JACKOVICH_OWNER,
    JACKOVICH_REF,
    JACKOVICH_REPO,
    STARTER_SYMBOLS,
)

# GitHub rejects requests without a User-Agent.
_USER_AGENT = "ml-trading-learning-repo (Jackovich CSV cache only)"
_TREE_URL = (
    f"https://api.github.com/repos/{JACKOVICH_OWNER}/{JACKOVICH_REPO}"
    f"/git/trees/{JACKOVICH_REF}?recursive=1"
)
_ZIPBALL_URL = (
    f"https://github.com/{JACKOVICH_OWNER}/{JACKOVICH_REPO}/archive/{JACKOVICH_REF}.zip"
)
_RAW_BASE = (
    f"https://raw.githubusercontent.com/{JACKOVICH_OWNER}/{JACKOVICH_REPO}/{JACKOVICH_REF}/"
)


class FetchError(RuntimeError):
    """Remote homework data could not be listed or downloaded."""


def default_data_dir() -> Path:
    return Path("data/ml4t")


def _request(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise FetchError(f"Failed to download {url}: {exc}") from exc


def list_remote_csv_names(data_dir: str = JACKOVICH_DATA_DIR) -> list[str]:
    """CSV filenames under the pinned Jackovich data folder (not ml4t/data dup)."""
    payload = json.loads(_request(_TREE_URL).decode("utf-8"))
    prefix = data_dir.rstrip("/") + "/"
    names: list[str] = []
    for entry in payload.get("tree", []):
        path = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix) :]
        if "/" in rel:
            continue  # skip Lists/ here; zip --all copies those separately
        if rel.endswith(".csv"):
            names.append(rel[:-4])  # symbol, may include $DJI
    names.sort()
    if not names:
        raise FetchError(f"No CSVs found under {data_dir} at ref {JACKOVICH_REF}")
    return names


def raw_csv_url(symbol: str, data_dir: str = JACKOVICH_DATA_DIR) -> str:
    rel = f"{data_dir}/{symbol}.csv"
    return _RAW_BASE + urllib.parse.quote(rel)


def fetch_symbols(
    symbols: list[str],
    dest: str | Path,
    *,
    force: bool = False,
    delay_s: float = 0.0,
) -> list[Path]:
    """Download selected `{SYMBOL}.csv` files from GitHub raw URLs."""
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, symbol in enumerate(symbols):
        out = dest_dir / f"{symbol}.csv"
        if out.is_file() and not force:
            written.append(out)
            continue
        url = raw_csv_url(symbol)
        out.write_bytes(_request(url))
        written.append(out)
        if delay_s and i + 1 < len(symbols):
            time.sleep(delay_s)
    return written


def fetch_zipball(dest: str | Path, *, skip_synthetic: bool = False) -> int:
    """Download the pinned repo zip and copy the homework data folder locally."""
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    blob = _request(_ZIPBALL_URL, timeout=180)
    copied = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            rel = _zip_member_under_data_dir(info.filename)
            if rel is None or info.is_dir():
                continue
            if skip_synthetic and _is_synthetic(rel):
                continue
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info.filename))
            copied += 1
    if copied == 0:
        raise FetchError("Zipball downloaded but no data files were extracted.")
    return copied


def _zip_member_under_data_dir(filename: str) -> str | None:
    """Return path relative to the homework data dir, or None to skip."""
    marker = f"/{JACKOVICH_DATA_DIR}/"
    idx = filename.find(marker)
    if idx < 0:
        return None
    rel = filename[idx + len(marker) :]
    if not rel:
        return None
    return rel


def _is_synthetic(rel_path: str) -> bool:
    name = Path(rel_path).name.upper()
    return name.startswith("ML4T-") or name.startswith("SINE_")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml_trading.fetch",
        description=(
            "Download stale ML4T homework CSVs from the pinned JeffreyJackovich "
            "GitHub commit into a local cache. Does not contact Yahoo Finance."
        )
    )
    parser.add_argument(
        "--dest",
        default=str(default_data_dir()),
        help="Local cache directory (default: data/ml4t)",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols to pull via raw GitHub URLs, e.g. SPY,FAKE1,FAKE2",
    )
    parser.add_argument(
        "--starter",
        action="store_true",
        help=f"Fetch the starter set: {', '.join(STARTER_SYMBOLS)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the full homework data folder (~170MB unzipped) via zipball",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="With --all, skip ML4T-* and SINE_* course-generated files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the local CSV already exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print remote CSV symbols and exit (GitHub git trees API)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dest = Path(args.dest)

    if args.list:
        names = list_remote_csv_names()
        sys.stdout.write("\n".join(names) + "\n")
        sys.stdout.write(f"# {len(names)} symbols at {JACKOVICH_REF}\n")
        return 0

    if args.all:
        n = fetch_zipball(dest, skip_synthetic=args.skip_synthetic)
        print(f"Extracted {n} files into {dest}")
        return 0

    symbols: list[str] = []
    if args.symbols.strip():
        symbols.extend(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.starter or not symbols:
        # Default action: small starter set, not the 170MB dump.
        if not symbols:
            symbols = list(STARTER_SYMBOLS)
            print(
                f"No --symbols/--all given; fetching starter set into {dest}: "
                + ", ".join(symbols)
            )
        elif args.starter:
            for sym in STARTER_SYMBOLS:
                if sym not in symbols:
                    symbols.append(sym)

    written = fetch_symbols(symbols, dest, force=args.force)
    print(f"Cached {len(written)} files in {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
