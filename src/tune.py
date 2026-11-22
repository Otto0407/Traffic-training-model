"""Hyperparameter search for the MLP, judged on validation macro F1 only.

One parameter is changed at a time from the current defaults, each setting is
trained with several seeds, and the mean ± std of validation macro F1 is
reported, so the effect of each parameter is visible on its own. Values that
beat the default by more than the seed-to-seed spread are then combined into
a "tuned" configuration. Only at that point is the test set used, once, to
score the default and the tuned configuration.

Usage: python src/tune.py [--seeds 42 43 44]
Writes outputs/tuning/{results.csv, summary.md, tuning.png, final.md}.
"""
import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cleaning import clean_data
from config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TrainConfig
from data_loader import load_data
from evaluate import compute_metrics, predict
from md_table import to_markdown
from pipeline import fit_model, prepare_data
from style import INK_SECONDARY, SERIES, apply_style

# (parameter, [(label, TrainConfig overrides), ...]). The default value of each
# parameter is not listed: it is the shared "default" run.
SEARCH = [
    ("hidden_sizes", [
        ("64-32", {"hidden_sizes": [64, 32]}),
        ("128-64", {"hidden_sizes": [128, 64]}),
        ("512-256-128", {"hidden_sizes": [512, 256, 128]}),
        ("256-128-64-32", {"hidden_sizes": [256, 128, 64, 32]}),
    ]),
    ("learning_rate", [
        ("1e-4", {"learning_rate": 1e-4}),
        ("3e-4", {"learning_rate": 3e-4}),
        ("3e-3", {"learning_rate": 3e-3}),
    ]),
    ("dropout", [
        ("0.1", {"dropout": 0.1}),
        ("0.5", {"dropout": 0.5}),
    ]),
    ("batch_size", [
        ("256", {"batch_size": 256}),
        ("1024", {"batch_size": 1024}),
    ]),
    ("weight_decay", [
        ("AdamW 1e-4", {"optimizer": "adamw", "weight_decay": 1e-4}),
        ("AdamW 1e-3", {"optimizer": "adamw", "weight_decay": 1e-3}),
        ("AdamW 1e-2", {"optimizer": "adamw", "weight_decay": 1e-2}),
    ]),
]
DEFAULT_LABELS = {
    "hidden_sizes": "256-128-64",
    "learning_rate": "1e-3",
    "dropout": "0.3",
    "batch_size": "512",
    "weight_decay": "Adam, none",
}


def train_once(df, overrides: dict, seed: int, device, score_test: bool = False) -> dict:
    cfg = replace(TrainConfig(), seed=seed, **overrides)
    data = prepare_data(df, cfg)
    start = time.time()
    model, history = fit_model(data, cfg, device, verbose=False)
    row = {
        "seed": seed,
        "val_macro_f1": compute_metrics(data.y_val, predict(model, data.X_val, device))["f1_macro"],
        "best_epoch": history["best_epoch"],
        "epochs_run": len(history["train_loss"]),
        "seconds": round(time.time() - start, 1),
    }
    if score_test:
        test = compute_metrics(data.y_test, predict(model, data.X_test, device))
        row.update({"test_macro_f1": test["f1_macro"], "test_accuracy": test["accuracy"], "test_qwk": test["qwk"]})
    return row


def run_setting(df, parameter, label, overrides, seeds, device, rows):
    for seed in seeds:
        row = {"parameter": parameter, "value": label, **train_once(df, overrides, seed, device)}
        rows.append(row)
        print(f"{parameter:<14} {label:<14} seed {seed}: val macro F1 {row['val_macro_f1']:.4f} "
              f"(best epoch {row['best_epoch']}/{row['epochs_run']}, {row['seconds']}s)", flush=True)


def plot_tuning(summary: pd.DataFrame, default_mean: float, default_std: float, path: Path) -> None:
    apply_style()
    params = [p for p, _ in SEARCH]
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.6), sharey=True)
    for ax, param in zip(axes, params):
        part = summary[summary.parameter == param]
        labels = list(part["value"])
        x = np.arange(len(labels))
        colors = [SERIES[1] if v == DEFAULT_LABELS[param] else SERIES[0] for v in labels]
        ax.axhspan(default_mean - default_std, default_mean + default_std, color=SERIES[1], alpha=0.12, lw=0)
        ax.errorbar(x, part["mean"], yerr=part["std"], fmt="none", ecolor=INK_SECONDARY, lw=1, capsize=3)
        ax.scatter(x, part["mean"], c=colors, s=50, zorder=3)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(param, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Validation macro F1 (mean ± std)")
    fig.suptitle("One parameter changed at a time (orange = current default, band = default ± std)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _ = clean_data(load_data(args.data))

    # 1. One-at-a-time sweep on validation
    rows = []
    run_setting(df, "default", "default", {}, args.seeds, device, rows)
    for param, options in SEARCH:
        for label, overrides in options:
            run_setting(df, param, label, overrides, args.seeds, device, rows)
    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)

    default = results[results.parameter == "default"]["val_macro_f1"]
    default_mean, default_std = default.mean(), default.std()
    summary_rows = []
    for param, options in SEARCH:
        summary_rows.append({"parameter": param, "value": DEFAULT_LABELS[param], "mean": default_mean,
                             "std": default_std, "best_epoch": results[results.parameter == "default"]["best_epoch"].mean(),
                             "seconds": results[results.parameter == "default"]["seconds"].mean()})
        for label, _ in options:
            part = results[(results.parameter == param) & (results.value == label)]
            summary_rows.append({"parameter": param, "value": label, "mean": part["val_macro_f1"].mean(),
                                 "std": part["val_macro_f1"].std(), "best_epoch": part["best_epoch"].mean(),
                                 "seconds": part["seconds"].mean()})
    summary = pd.DataFrame(summary_rows)
    summary["diff_vs_default"] = summary["mean"] - default_mean
    plot_tuning(summary, default_mean, default_std, out / "tuning.png")

    # 2. Combine values that beat the default by more than the seed spread
    threshold = max(default_std, 1e-4)
    chosen, chosen_labels = {}, {}
    for param, options in SEARCH:
        part = summary[(summary.parameter == param) & (summary.value != DEFAULT_LABELS[param])]
        best = part.loc[part["mean"].idxmax()]
        if best["diff_vs_default"] > threshold:
            chosen.update(dict(options)[best["value"]])
            chosen_labels[param] = best["value"]
    shown = summary.assign(
        mean=lambda t: t["mean"].round(4), std=lambda t: t["std"].round(4),
        diff_vs_default=lambda t: t["diff_vs_default"].round(4), best_epoch=lambda t: t["best_epoch"].round(1),
        seconds=lambda t: t["seconds"].round(0),
    )
    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"Validation macro F1, mean ± std over seeds {args.seeds}. Default: {default_mean:.4f} ± {default_std:.4f}.\n")
        f.write(f"A value counts as better only if it beats the default by more than {threshold:.4f}.\n\n")
        f.write(to_markdown(shown, index=False))
    print("\n" + shown.to_string(index=False))
    print(f"\nChosen changes: {chosen_labels or 'none (no value beat the default by more than the seed spread)'}")

    # 3. Score default and tuned configuration on the test set, once
    final_rows = []
    configs = {"default": {}}
    if chosen:
        configs["tuned"] = chosen
    for name, overrides in configs.items():
        runs = pd.DataFrame([train_once(df, overrides, s, device, score_test=True) for s in args.seeds])
        final_rows.append({"configuration": name, "changes": str(chosen_labels) if name == "tuned" else "-",
                           **{c: f"{runs[c].mean():.4f} ± {runs[c].std():.4f}"
                              for c in ("val_macro_f1", "test_macro_f1", "test_accuracy", "test_qwk")}})
    final = pd.DataFrame(final_rows)
    with open(out / "final.md", "w", encoding="utf-8") as f:
        f.write("Default vs tuned configuration (mean ± std over seeds). The tuned configuration was chosen on\n")
        f.write("validation macro F1 only; the test columns were computed after that choice.\n\n")
        f.write(to_markdown(final, index=False))
        f.write(f"\nTuned settings: {asdict(replace(TrainConfig(), **chosen))}\n")
    print("\n" + final.to_string(index=False))
    print(f"\nSaved tuning results to {out}")


if __name__ == "__main__":
    main()
