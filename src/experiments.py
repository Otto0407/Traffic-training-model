"""Compare modelling choices: feature set, split, imbalance handling,
augmentation and regularisation. Every MLP variant is trained with several
seeds and the mean ± std of the test metrics is reported.

Usage: python src/experiments.py [--seeds 42 43 44] [--only NAME ...]
Writes outputs/experiments/{results.csv, summary.md, macro_f1.png}.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baseline import lookup_baseline, majority_baseline
from config import CLASS_ORDER, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TrainConfig
from cleaning import clean_data
from data_loader import load_data
from evaluate import compute_metrics, predict
from features import LEAK_FREE, VEHICLE_RATIOS, FeatureSet, add_vehicle_ratios, all_columns_feature_set
from md_table import to_markdown
from pipeline import fit_model, prepare_data
from split import time_split
from style import INK_SECONDARY, SERIES, apply_style

TIME_FEATURES = ("hour", "rush_hour", "peak_period")

# (name, description, feature set key, split, TrainConfig overrides)
EXPERIMENTS = [
    ("all_columns", "Proposal plan: all 44 columns incl. leaking ones", "all", "time", {}),
    ("random_split", "Final features, random 60/20/20 split", "leak_free", "random", {}),
    ("final", "Final model: leak-free features, time split, unweighted loss", "leak_free", "time", {}),
    ("class_weights", "CrossEntropyLoss weighted by inverse class frequency", "leak_free", "time",
     {"class_weighted_loss": True}),
    (
        "oversampling",
        "Balanced mini-batches (random oversampling of minority classes)",
        "leak_free",
        "time",
        {"balanced_sampling": True},
    ),
    ("input_noise", "Gaussian noise (std 0.1) on numerical inputs", "leak_free", "time", {"input_noise": 0.1}),
    ("no_dropout", "Dropout 0 (early stopping still on)", "leak_free", "time", {"dropout": 0.0}),
    (
        "no_regularisation",
        "Dropout 0 and no early stopping (40 epochs, last weights)",
        "leak_free",
        "time",
        {"dropout": 0.0, "max_epochs": 40, "patience": 10**6, "restore_best_weights": False},
    ),
    ("no_time_features", "Final features minus hour / rush_hour / peak_period", "no_time", "time", {}),
    ("with_intersection_id", "Final features plus intersection_id (172 inputs)", "with_intersection", "time", {}),
    ("vehicle_ratios", "Final features plus 4 vehicle-mix ratios", "vehicle_ratios", "time", {}),
]


def feature_sets(df: pd.DataFrame) -> dict:
    return {
        "all": all_columns_feature_set(df),
        "leak_free": LEAK_FREE,
        "no_time": LEAK_FREE.without(*TIME_FEATURES),
        "with_intersection": FeatureSet(LEAK_FREE.categorical + ("intersection_id",), LEAK_FREE.numerical,
                                        LEAK_FREE.binary),
        "vehicle_ratios": FeatureSet(LEAK_FREE.categorical, LEAK_FREE.numerical + VEHICLE_RATIOS, LEAK_FREE.binary),
    }


def metric_row(name: str, seed, y_true, y_pred, extra: dict = None) -> dict:
    m = compute_metrics(y_true, y_pred)
    row = {
        "experiment": name,
        "seed": seed,
        "accuracy": m["accuracy"],
        "macro_f1": m["f1_macro"],
        "weighted_f1": m["f1_weighted"],
        "macro_precision": m["precision_macro"],
        "macro_recall": m["recall_macro"],
    }
    # Per-class F1 from the confusion matrix (avoids parsing the text report).
    cm = np.asarray(m["confusion_matrix"], dtype=float)
    for i, cls in enumerate(CLASS_ORDER):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        row[f"f1_{cls}"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    row.update(extra or {})
    return row


def run_baselines(df: pd.DataFrame, cfg: TrainConfig) -> list:
    train_df, _, test_df = time_split(df, cfg.train_frac, cfg.val_frac)
    data = prepare_data(df, cfg)
    rows = [metric_row("baseline_majority", "-", data.y_test, majority_baseline(data.y_train, len(data.y_test)))]
    for name, keys in (
        ("baseline_hour_lookup", ["hour"]),
        ("baseline_hour_weekend_zone_lookup", ["hour", "is_weekend", "city_zone"]),
    ):
        rows.append(metric_row(name, "-", data.y_test, lookup_baseline(train_df, test_df, keys)))
    return rows


def run_mlp(df, name, fs_key, split, overrides, seed, sets, device) -> dict:
    cfg = replace(TrainConfig(), seed=seed, **overrides)
    if fs_key == "vehicle_ratios":
        df = add_vehicle_ratios(df)
    data = prepare_data(df, cfg, sets[fs_key], split)
    model, history = fit_model(data, cfg, device, verbose=False)
    best = history["best_epoch"] - 1
    return metric_row(
        name,
        seed,
        data.y_test,
        predict(model, data.X_test, device),
        {
            "input_dim": data.X_train.shape[1],
            "epochs_run": len(history["train_loss"]),
            "best_epoch": history["best_epoch"],
            "train_loss_at_best": history["train_loss"][best],
            "val_loss_at_best": history["val_loss"][best],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        },
    )


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"] + [
        f"f1_{c}" for c in CLASS_ORDER
    ]
    order = list(dict.fromkeys(results["experiment"]))
    agg = results.groupby("experiment", sort=False)[metrics].agg(["mean", "std"]).reindex(order)
    out = pd.DataFrame(index=agg.index)
    for m in metrics:
        mean, std = agg[(m, "mean")], agg[(m, "std")].fillna(0)
        out[m] = [f"{a:.3f} ± {s:.3f}" if s > 0 else f"{a:.3f}" for a, s in zip(mean, std)]
    if "best_epoch" in results:
        ep = results.groupby("experiment", sort=False)["best_epoch"].mean().reindex(order)
        out["best_epoch"] = [f"{v:.0f}" if pd.notna(v) else "-" for v in ep]
    return out


def plot_macro_f1(results: pd.DataFrame, path: Path) -> None:
    apply_style()
    order = list(dict.fromkeys(results["experiment"]))
    g = results.groupby("experiment", sort=False)["macro_f1"]
    mean, std = g.mean().reindex(order), g.std().fillna(0).reindex(order)
    is_final = [n == "final" for n in order]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    colors = [SERIES[1] if f else SERIES[0] for f in is_final]
    ax.barh(y, mean, xerr=std, height=0.6, color=colors, error_kw={"ecolor": INK_SECONDARY, "lw": 1})
    for yi, m in zip(y, mean):
        ax.text(m + 0.015, yi, f"{m:.3f}", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Test macro F1 (mean ± std over seeds)")
    ax.set_title("Macro F1 by experiment (final model highlighted)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--only", nargs="+", help="Run only these experiment names.")
    parser.add_argument("--max-epochs", type=int, help="Cap epochs for every run (for quick checks).")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _ = clean_data(load_data(args.data))
    sets = feature_sets(df)

    rows = run_baselines(df, TrainConfig())
    for name, desc, fs_key, split, overrides in EXPERIMENTS:
        if args.only and name not in args.only:
            continue
        if args.max_epochs:
            overrides = {**overrides, "max_epochs": min(args.max_epochs, overrides.get("max_epochs", 10**6))}
        for seed in args.seeds:
            row = run_mlp(df, name, fs_key, split, overrides, seed, sets, device)
            rows.append(row)
            print(
                f"{name:<18} seed {seed}: macro F1 {row['macro_f1']:.4f} acc {row['accuracy']:.4f} "
                f"(best epoch {row['best_epoch']}/{row['epochs_run']})",
                flush=True,
            )

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    summary = summarise(results)
    descriptions = {"baseline_majority": "Always predict the most common class",
                    "baseline_hour_lookup": "Most common label for each hour of day",
                    "baseline_hour_weekend_zone_lookup": "Most common label per hour × weekend × zone"}
    descriptions.update({n: d for n, d, *_ in EXPERIMENTS})
    summary.insert(0, "description", [descriptions[n] for n in summary.index])
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(to_markdown(summary))
    print("\n" + summary.to_string())
    plot_macro_f1(results, out / "macro_f1.png")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
