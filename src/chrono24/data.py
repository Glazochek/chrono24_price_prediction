from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "philmorekoung11/luxury-watch-listings"
CSV_NAME = "Watches.csv"


def _find_local_csv() -> Path | None:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for hit in kaggle_input.rglob(CSV_NAME):
            return hit

    candidates = [
        Path.cwd() / "data" / "raw" / CSV_NAME,
        Path(__file__).resolve().parents[2] / "data" / "raw" / CSV_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def load_watches() -> pd.DataFrame:
    csv = _find_local_csv()
    if csv is None:
        import kagglehub
        from dotenv import load_dotenv

        load_dotenv()
        download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csv = next(download_dir.rglob(CSV_NAME))
    return pd.read_csv(csv)
