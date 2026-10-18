"""End-to-end pipeline: load -> select features -> time split -> preprocess
-> train MLP -> compare with the majority baseline on the test set.

Usage: python src/main.py [--data PATH] [--output-dir DIR] [--epochs N] ...
"""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch

from baseline import majority_baseline
from config import CLASS_ORDER, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TIMESTAMP, TrainConfig
from data_loader import load_data
from dataset import make_loader
from evaluate import compute_metrics, predict
from features import select_features
from model import CongestionMLP
from plots import plot_confusion_matrix, plot_history
from preprocessing import build_preprocessor, encode_labels
from split import time_split
from train import class_weights, train_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train the congestion level MLP.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--no-class-weights", action="store_true", help="Use unweighted CrossEntropyLoss.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


def summarise(name: str, m: dict) -> str:
    return (
        f"{name:<10} acc {m['accuracy']:.4f} | macro P {m['precision_macro']:.4f} "
        f"R {m['recall_macro']:.4f} F1 {m['f1_macro']:.4f} | weighted F1 {m['f1_weighted']:.4f}"
    )


def main():
    args = parse_args()
    cfg = TrainConfig(
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        class_weighted_loss=not args.no_class_weights,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load and split chronologically
    df = load_data(args.data)
    train_df, val_df, test_df = time_split(df, cfg.train_frac, cfg.val_frac)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"{name:<5} {len(part):>7} rows  {part[TIMESTAMP].min()} -> {part[TIMESTAMP].max()}")

    # 2. Feature selection and preprocessing (fitted on train only)
    X_train, y_train = select_features(train_df)
    X_val, y_val = select_features(val_df)
    X_test, y_test = select_features(test_df)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val = preprocessor.transform(X_val).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)
    y_train, y_val, y_test = encode_labels(y_train), encode_labels(y_val), encode_labels(y_test)
    print(f"Input dimension after encoding: {X_train.shape[1]}")

    # 3. Train
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(X_val, y_val, cfg.batch_size, shuffle=False)
    model = CongestionMLP(X_train.shape[1], cfg.hidden_sizes, len(CLASS_ORDER), cfg.dropout)
    weights = class_weights(y_train, len(CLASS_ORDER)) if cfg.class_weighted_loss else None
    model, history = train_model(model, train_loader, val_loader, cfg, device, weights)

    # 4. Evaluate on the held-out test set
    mlp_metrics = compute_metrics(y_test, predict(model, X_test, device))
    base_metrics = compute_metrics(y_test, majority_baseline(y_train, len(y_test)))

    print("\n=== Test set ===")
    print(summarise("baseline", base_metrics))
    print(summarise("MLP", mlp_metrics))
    print("\nMLP per-class report:\n" + mlp_metrics["report"])

    # 5. Save artefacts
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": X_train.shape[1], "config": asdict(cfg)},
        out / "model.pt",
    )
    joblib.dump(preprocessor, out / "preprocessor.joblib")
    with open(out / "metrics.json", "w") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "best_epoch": history["best_epoch"],
                "mlp": {k: v for k, v in mlp_metrics.items() if k != "report"},
                "majority_baseline": {k: v for k, v in base_metrics.items() if k != "report"},
            },
            f,
            indent=2,
        )
    plot_history(history, out / "training_curves.png")
    plot_confusion_matrix(mlp_metrics["confusion_matrix"], out / "confusion_matrix.png")
    print(f"Saved model, preprocessor, metrics and plots to {out}")


if __name__ == "__main__":
    main()
