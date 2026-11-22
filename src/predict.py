"""Predict congestion levels for new, real-world rows with a trained model.

Usage:
  python src/predict.py --input new_rows.csv [--output predictions.csv]
                        [--id-column record_id] [--hash-ids]
                        [--model-dir outputs] [--report report.json]

The input may have renamed columns, numbers written as text, yes/no flags,
spelling variants, missing columns and unrelated (including personal) columns;
see input_validation.py. Rows without an hour (or a timestamp to derive it
from) are not predicted.

Privacy: the output holds only the prediction, its confidence and a data
quality flag, in the input's row order. Input columns are not copied over.
With --id-column, that column is included so rows can be matched up; add
--hash-ids to replace it with a keyed hash (HMAC-SHA256). The key is read from
the PREDICT_ID_SALT environment variable and is never written anywhere, so
the original IDs cannot be recovered from the output without it.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import CLASS_ORDER, DEFAULT_OUTPUT_DIR
from input_validation import known_categories, normalise_column_name, validate_input
from model import CongestionMLP

SALT_ENV = "PREDICT_ID_SALT"


def load_trained(model_dir: Path, device: torch.device) -> tuple:
    # weights_only=True refuses to unpickle arbitrary objects from the checkpoint.
    checkpoint = torch.load(model_dir / "model.pt", map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = CongestionMLP(checkpoint["input_dim"], cfg["hidden_sizes"], len(CLASS_ORDER), cfg["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    # joblib files are pickles: only load a preprocessor you trained yourself.
    return model, joblib.load(model_dir / "preprocessor.joblib")


def read_table(path: Path) -> pd.DataFrame:
    """Read every cell as text (parsing happens in validation), trying common
    encodings and detecting the separator (comma, semicolon, tab)."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8, Windows-1252 or Latin-1.")


@torch.no_grad()
def predict_proba(model, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    probs = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probs) if probs else np.empty((0, len(CLASS_ORDER)))


def pseudonymise(ids: pd.Series, key: str) -> pd.Series:
    return ids.map(lambda v: hmac.new(key.encode(), str(v).encode(), hashlib.sha256).hexdigest()[:16])


def predict_table(raw: pd.DataFrame, model, preprocessor, device) -> tuple:
    """Return (result, report). result has one row per input row."""
    features, quality, report = validate_input(raw, known_categories(preprocessor))
    result = quality.copy()
    result["predicted_congestion_level"] = pd.Series(pd.NA, index=raw.index, dtype="object")
    result["confidence"] = np.nan

    usable = result["data_quality"] != "insufficient_data"
    if usable.any():
        X = preprocessor.transform(features[usable]).astype(np.float32)
        probs = predict_proba(model, X, device)
        result.loc[usable, "predicted_congestion_level"] = [CLASS_ORDER[i] for i in probs.argmax(axis=1)]
        result.loc[usable, "confidence"] = probs.max(axis=1).astype(float).round(4)
    report["rows_predicted"] = int(usable.sum())
    return result, report


def main():
    parser = argparse.ArgumentParser(description="Predict congestion levels for new rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-column", help="Input column to carry over so rows can be matched up.")
    parser.add_argument("--hash-ids", action="store_true", help=f"Replace the ID with a keyed hash (key in ${SALT_ENV}).")
    parser.add_argument("--report", type=Path, help="Also save the validation report (counts only) as JSON.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessor = load_trained(args.model_dir, device)
    raw = read_table(args.input)
    result, report = predict_table(raw, model, preprocessor, device)

    columns = ["predicted_congestion_level", "confidence", "data_quality", "n_features_imputed"]
    if args.id_column:
        id_col = normalise_column_name(args.id_column)
        names = {normalise_column_name(c): c for c in raw.columns}
        if id_col not in names:
            raise SystemExit(f"--id-column '{args.id_column}' is not in the input.")
        ids = raw[names[id_col]]
        if args.hash_ids:
            key = os.environ.get(SALT_ENV)
            if not key:
                raise SystemExit(f"--hash-ids needs a secret key in the {SALT_ENV} environment variable.")
            ids = pseudonymise(ids, key)
        result.insert(0, "row_id", ids)
        columns = ["row_id"] + columns
    result[columns].to_csv(args.output, index=False)

    print("Input validation (counts only):")
    for key, value in report.items():
        print(f"  {key.replace('_', ' ')}: {value}")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(result)} rows ({report['rows_predicted']} predicted) to {args.output}")


if __name__ == "__main__":
    main()
