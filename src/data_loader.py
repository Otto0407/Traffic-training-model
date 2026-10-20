"""Load the CityFlow CSV into a DataFrame."""
from pathlib import Path

import pandas as pd

from config import TIMESTAMP


def load_data(path: Path) -> pd.DataFrame:
    """Read the dataset and parse timestamps. Unparseable timestamps become
    NaT; invalid rows are handled by cleaning.clean_data, not here.

    If the CSV is missing but a .zip with the same name is next to it (as
    downloaded from Kaggle), the CSV is read straight from the zip.
    """
    path = Path(path)
    if not path.exists() and path.with_suffix(".zip").exists():
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place "
            "smart_city_traffic_mobility.csv (or the .zip) in the data/ folder."
        )
    df = pd.read_csv(path)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    return df.sort_values(TIMESTAMP).reset_index(drop=True)
