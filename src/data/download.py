"""Download and assemble the UCI Heart Disease dataset.

We deliberately combine all four original data collection sites (Cleveland,
Hungary, Switzerland, VA Long Beach) rather than using only the commonly
distributed 303-row Cleveland subset. The combined dataset has realistic
missingness that the rest of the pipeline must handle, which is closer to a
production data-quality scenario than the frequently-used clean subset.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease (UCI ML Repository)
"""

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DATA_PATH

logger = logging.getLogger(__name__)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va_long_beach": "processed.va.data",
}


def _parse_site_file(raw_bytes: bytes, source: str) -> pd.DataFrame:
    text = raw_bytes.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), header=None, names=COLUMN_NAMES, na_values="?")
    df["source"] = source
    return df


def download_raw_dataset(zip_url: str = UCI_ZIP_URL, timeout: int = 30) -> pd.DataFrame:
    """Fetch the official UCI zip archive and combine the 4 processed site files."""
    logger.info("Downloading UCI Heart Disease archive from %s", zip_url)
    response = requests.get(zip_url, timeout=timeout)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    frames = []
    for source, filename in SITE_FILES.items():
        raw_bytes = archive.read(filename)
        frames.append(_parse_site_file(raw_bytes, source))
        logger.info("Parsed %s: %d rows", filename, len(frames[-1]))

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows, %d columns", *combined.shape)
    return combined


def save_raw_dataset(df: pd.DataFrame, output_path: Path = RAW_DATA_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved raw dataset to %s", output_path)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = download_raw_dataset()
    save_raw_dataset(df)


if __name__ == "__main__":
    main()
