"""Check that cleaning leaves the real data untouched and fixes a dirty sample.

Run: python tests/test_cleaning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np, pandas as pd
from data_loader import load_data
from cleaning import clean_data
from preprocessing import build_preprocessor
from features import select_features
from config import DEFAULT_DATA_PATH
raw = load_data(DEFAULT_DATA_PATH)
clean, rep = clean_data(raw)
print("real data:", rep)
assert len(clean) == len(raw)
assert (clean.dtypes == raw.dtypes).all(), "dtypes changed on clean data"
# dirty sample
d = raw.head(300).copy()
d = pd.concat([d, d.iloc[:5]])                      # 5 exact duplicates
d.loc[d.index[10], "congestion_level"] = "Unknown"   # bad label
d.loc[d.index[11], "congestion_level"] = " Low "     # whitespace label -> fixed
d.loc[d.index[12], "timestamp"] = pd.NaT             # missing time
d.loc[d.index[13], "temperature"] = -100000          # impossible value
d.loc[d.index[14], "is_weekend"] = 7                 # bad flag
d.loc[d.index[15], "weather_condition"] = np.nan     # missing category
d.loc[d.index[16], "rainfall_mm"] = np.nan           # missing number
c, r = clean_data(d)
print("dirty sample:", r)
assert r["duplicate_rows_dropped"] == 5 and r["invalid_label_rows_dropped"] == 1
assert r["missing_timestamp_rows_dropped"] == 1 and r["out_of_range_values_set_missing"] == 1
assert r["invalid_binary_values_set_missing"] == 1 and r["rows_out"] == 298
X, _ = select_features(c)
out = build_preprocessor().fit_transform(X)
assert not np.isnan(out.astype(float)).any(), "NaN left after imputation"
print("preprocessed shape", out.shape, "- no NaN")
print("All cleaning checks passed.")
