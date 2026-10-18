"""Predict congestion levels for new rows with a trained model.

Usage: python src/predict.py --input new_rows.csv [--model-dir outputs] [--output predictions.csv]
The input CSV needs the feature columns listed in config.py; other columns are ignored.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import CLASS_ORDER, DEFAULT_OUTPUT_DIR
from evaluate import predict
from features import feature_columns
from model import CongestionMLP
from preprocessing import decode_labels


def load_trained(model_dir: Path, device: torch.device) -> tuple:
    checkpoint = torch.load(model_dir / "model.pt", map_location=device)
    cfg = checkpoint["config"]
    model = CongestionMLP(checkpoint["input_dim"], cfg["hidden_sizes"], len(CLASS_ORDER), cfg["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, joblib.load(model_dir / "preprocessor.joblib")


def main():
    parser = argparse.ArgumentParser(description="Predict congestion levels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessor = load_trained(args.model_dir, device)

    df = pd.read_csv(args.input)
    X = preprocessor.transform(df[feature_columns()]).astype(np.float32)
    df["predicted_congestion_level"] = decode_labels(predict(model, X, device))
    df.to_csv(args.output, index=False)
    print(df["predicted_congestion_level"].value_counts().to_string())
    print(f"Wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
