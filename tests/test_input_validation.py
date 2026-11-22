"""Check prediction-time validation on clean and deliberately messy input.

Run: python tests/test_input_validation.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import torch

from cleaning import clean_data
from config import CLASS_ORDER, DEFAULT_DATA_PATH
from data_loader import load_data
from features import select_features
from input_validation import PEAK_PERIOD_BY_HOUR, WEEKDAY_RUSH_HOURS, known_categories, validate_input
from model import CongestionMLP
from predict import predict_table, pseudonymise, read_table
from preprocessing import build_preprocessor

data, _ = clean_data(load_data(DEFAULT_DATA_PATH))
sample = data.sample(3000, random_state=0).sort_index()
preprocessor = build_preprocessor().fit(select_features(sample)[0])
categories = known_categories(preprocessor)
model = CongestionMLP(preprocessor.transform(select_features(sample)[0]).shape[1], [32], len(CLASS_ORDER), 0.0)
model.eval()
device = torch.device("cpu")


def as_text(df: pd.DataFrame) -> pd.DataFrame:
    """What read_table produces: every cell as a string."""
    return df.astype(str).replace({"NaT": "", "nan": ""})


# 1. The calendar rules used to rebuild missing fields match the training data.
assert (data["peak_period"] == data["hour"].map(PEAK_PERIOD_BY_HOUR)).all()
rush = (data["hour"].isin(WEEKDAY_RUSH_HOURS) & (data["is_weekend"] == 0)).astype(int)
assert (data["rush_hour"] == rush).all()
assert (data["is_weekend"] == (data["day_of_week"] >= 5).astype(int)).all()

# 2. Clean input passes through unchanged: same model input as the training path.
clean_rows = sample.head(500)
features, quality, report = validate_input(as_text(clean_rows), categories)
expected = preprocessor.transform(select_features(clean_rows)[0])
assert np.allclose(preprocessor.transform(features).astype(float), expected.astype(float))
assert (quality["data_quality"] == "ok").all(), quality["data_quality"].value_counts()
print("clean input:", report)

# 3. Messy, real-world style input.
messy = as_text(sample.head(12)).reset_index(drop=True)
messy = messy.rename(columns={"weather_condition": "Weather", "city_zone": "Zone ", "rainfall_mm": "Rainfall",
                              "temperature": "temp", "is_holiday": "Is Holiday"})
messy = messy.drop(columns=["parking_occupancy"])
messy["Plate Number"] = [f"ABC-{i:04d}" for i in range(12)]
messy["driver_name"] = "Jane Doe"
messy["Phone"] = "0912-345-678"
messy.loc[0, "Rainfall"] = "3.5 mm"
messy.loc[1, "Rainfall"] = "1,200"            # out of range -> missing
messy.loc[2, "temp"] = "N/A"
messy.loc[3, "lanes"] = "two"                 # not a number -> missing
messy.loc[4, "Is Holiday"] = "yes"
messy.loc[5, "Is Holiday"] = "No"
messy.loc[6, "Weather"] = " heavy_rain "      # spelling variant -> Heavy Rain
messy.loc[7, "Weather"] = "Sandstorm"         # unknown category -> missing
messy.loc[8, "hour"] = ""                     # rebuilt from the timestamp
messy.loc[9, ["hour", "timestamp"]] = ""      # no way to know the hour
messy.loc[10, "day_of_week"] = "9"            # impossible -> rebuilt from timestamp
messy.loc[11, "Zone "] = "downtown core"      # case variant

features, quality, report = validate_input(messy, categories)
print("messy input:", report)
assert features.loc[0, "rainfall_mm"] == 3.5
assert np.isnan(features.loc[1, "rainfall_mm"]) and np.isnan(features.loc[2, "temperature"])
assert np.isnan(features.loc[3, "lanes"])
assert features.loc[4, "is_holiday"] == 1 and features.loc[5, "is_holiday"] == 0
assert features.loc[6, "weather_condition"] == "Heavy Rain"
assert pd.isna(features.loc[7, "weather_condition"])
assert features.loc[8, "hour"] == pd.Timestamp(sample.iloc[8]["timestamp"]).hour
assert features.loc[10, "day_of_week"] == pd.Timestamp(sample.iloc[10]["timestamp"]).dayofweek
assert features.loc[11, "city_zone"] == "Downtown Core"
assert quality.loc[9, "data_quality"] == "insufficient_data"
assert report["columns_missing"] == ["parking_occupancy"]
assert report["columns_dropped_personal_data"] == 5  # plate, driver, phone, latitude, longitude
assert report["unknown_category_values"] == 1 and report["out_of_range_values"] == 2

# 4. End to end through a semicolon-separated Windows-1252 file: predictions for
#    every row except the one without an hour, and no personal data in the output.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "messy.csv"
    messy.assign(driver_name="José Müller").to_csv(path, sep=";", index=False, encoding="cp1252")
    result, report = predict_table(read_table(path), model, preprocessor, device)
    assert report["rows_predicted"] == 11
    assert result.loc[9, "predicted_congestion_level"] is pd.NA
    assert result["predicted_congestion_level"].dropna().isin(CLASS_ORDER).all()
    out_text = result.to_csv(index=False)
    for secret in ("ABC-0000", "José", "Müller", "0912-345-678"):
        assert secret not in out_text, secret
    assert set(result.columns) == {"n_features_imputed", "data_quality", "predicted_congestion_level", "confidence"}

# 5. Pseudonymised IDs are stable for the same key and unreadable without it.
ids = pd.Series(["REC_1", "REC_2"])
assert (pseudonymise(ids, "k1") == pseudonymise(ids, "k1")).all()
assert (pseudonymise(ids, "k1") != pseudonymise(ids, "k2")).all()
assert not pseudonymise(ids, "k1").str.contains("REC").any()

print("All input validation checks passed.")
