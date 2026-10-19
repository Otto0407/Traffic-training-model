"""End-to-end pipeline: load -> select features -> time split -> preprocess
-> train MLP -> compare with the majority baseline on the test set.

Usage: python src/main.py [--data PATH] [--output-dir DIR] [--epochs N] ...
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import torch

from baseline import majority_baseline
from config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TIMESTAMP, TrainConfig
from data_loader import load_data
from evaluate import compute_metrics, predict
from pipeline import fit_model, prepare_data, set_seed
from plots import plot_confusion_matrix, plot_history


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
    parser.add_argument("--class-weights", action="store_true", help="Weight CrossEntropyLoss by inverse class frequency.")
    parser.add_argument("--balanced-sampling", action="store_true", help="Oversample minority classes.")
    parser.add_argument("--input-noise", type=float, default=defaults.input_noise, help="Gaussian noise std (augmentation).")
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
        class_weighted_loss=args.class_weights,
        balanced_sampling=args.balanced_sampling,
        input_noise=args.input_noise,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1-2. Load, split chronologically, preprocess (fitted on train only)
    df = load_data(args.data)
    data = prepare_data(df, cfg)
    for name, X in (("train", data.X_train), ("val", data.X_val), ("test", data.X_test)):
        print(f"{name:<5} {len(X):>7} rows")
    print(f"Test period: {data.test_df[TIMESTAMP].min()} -> {data.test_df[TIMESTAMP].max()}")
    print(f"Input dimension after encoding: {data.X_train.shape[1]}")

    # 3. Train
    model, history = fit_model(data, cfg, device)

    # 4. Evaluate on the held-out test set
    mlp_metrics = compute_metrics(data.y_test, predict(model, data.X_test, device))
    base_metrics = compute_metrics(data.y_test, majority_baseline(data.y_train, len(data.y_test)))

    print("\n=== Test set ===")
    print(summarise("baseline", base_metrics))
    print(summarise("MLP", mlp_metrics))
    print("\nMLP per-class report:\n" + mlp_metrics["report"])

    # 5. Save artefacts
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": data.X_train.shape[1], "config": asdict(cfg)},
        out / "model.pt",
    )
    joblib.dump(data.preprocessor, out / "preprocessor.joblib")
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
