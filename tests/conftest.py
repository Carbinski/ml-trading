from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "ml4t"
DIRTY_CSV = ROOT / "tests" / "fixtures" / "dirty_ohlcv.csv"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture
def dirty_csv() -> Path:
    return DIRTY_CSV
