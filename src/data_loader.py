"""Load the CityFlow CSV into a DataFrame."""
from pathlib import Path

import pandas as pd

from config import TARGET, TIMESTAMP, CLASS_ORDER


def load_data(path: Path) -> pd.DataFrame:
    """Read the dataset, parse timestamps and check the target labels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from Kaggle and place "
            "smart_city_traffic_mobility.csv in the data/ folder."
        )
    df = pd.read_csv(path, parse_dates=[TIMESTAMP])

    unknown = set(df[TARGET].unique()) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unexpected {TARGET} values: {sorted(unknown)}")

    return df.sort_values(TIMESTAMP).reset_index(drop=True)
