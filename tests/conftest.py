from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def raw_sample_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "sample_data.csv")
