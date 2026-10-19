"""Where does the trained model fail? Breaks test-set errors down by hour,
weekday/weekend, zone and conditions, and compares with the lookup baseline.

Run main.py first (this loads outputs/model.pt).
Usage: python src/error_analysis.py [--model-dir outputs]
Writes outputs/errors/{error_summary.md, error_by_hour.png}.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baseline import lookup_baseline
from config import CLASS_ORDER, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TARGET, TrainConfig
from data_loader import load_data
from evaluate import predict
from features import select_features
from md_table import to_markdown
from predict import load_trained
from preprocessing import decode_labels
from split import time_split
from style import SERIES, apply_style


def error_rate_by(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys)
    return pd.DataFrame({"rows": g.size(), "errors": g["wrong"].sum(), "error_rate": g["wrong"].mean().round(3)})


def plot_error_by_hour(df: pd.DataFrame, path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for color, (label, flag) in zip(SERIES, (("Weekday", 0), ("Weekend", 1))):
        rate = df[df.is_weekend == flag].groupby("hour")["wrong"].mean().reindex(range(24), fill_value=0)
        ax.plot(rate.index, rate.values, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Test error rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(bottom=0)
    ax.set_title("Where the MLP makes mistakes (test set)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "errors")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessor = load_trained(args.model_dir, device)
    cfg = TrainConfig()
    train_df, _, test_df = time_split(load_data(args.data), cfg.train_frac, cfg.val_frac)

    X_test, _ = select_features(test_df)
    pred = predict(model, preprocessor.transform(X_test).astype(np.float32), device)
    lookup = lookup_baseline(train_df, test_df, ["hour", "is_weekend", "city_zone"])

    df = test_df.copy()
    df["predicted"] = decode_labels(pred)
    df["lookup"] = decode_labels(lookup)
    df["wrong"] = df["predicted"] != df[TARGET]
    df["lookup_wrong"] = df["lookup"] != df[TARGET]
    df["error_type"] = df[TARGET] + " -> " + df["predicted"]
    errors = df[df.wrong]
    weekday_evening = df[(df.hour.isin([18, 19])) & (df.is_weekend == 0)]

    def conditions(frame):
        return frame.assign(
            bad_weather=frame.weather_condition != "Clear",
            any_incident=(frame.accident_reported == 1) | (frame.public_event == 1),
        )

    sections = [
        ("Overall", pd.DataFrame({"MLP": [df.wrong.mean()], "lookup hour x weekend x zone": [df.lookup_wrong.mean()]},
                                 index=["test error rate"]).round(4)),
        ("Most common error types", errors["error_type"].value_counts().head(8).to_frame("errors")),
        ("Error rate by hour and weekend flag", error_rate_by(df, ["is_weekend", "hour"])
            .query("errors > 0")),
        ("Errors by hour group", errors.assign(
            period=pd.cut(errors.hour, [-1, 6, 9, 16, 19, 23], labels=["0-6", "7-9", "10-16", "17-19", "20-23"]))
            .groupby("period", observed=False).size().to_frame("errors")),
        ("Weekday 18-19h: error rate by zone", error_rate_by(weekday_evening, "city_zone")),
        ("Weekday 18-19h: error rate by conditions", error_rate_by(conditions(weekday_evening),
                                                                  ["bad_weather", "any_incident"])),
        ("Weekday 18-19h: what the MLP predicts vs truth", pd.crosstab(weekday_evening[TARGET],
                                                                       weekday_evening["predicted"])
            .reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)),
        ("MLP vs lookup on the same rows", pd.crosstab(df.wrong.map({True: "MLP wrong", False: "MLP right"}),
                                                       df.lookup_wrong.map({True: "lookup wrong",
                                                                            False: "lookup right"}))),
    ]
    with open(out / "error_summary.md", "w", encoding="utf-8") as f:
        for title, table in sections:
            f.write(f"## {title}\n\n{to_markdown(table)}\n")
    print((out / "error_summary.md").read_text(encoding="utf-8"))
    plot_error_by_hour(df, out / "error_by_hour.png")
    print(f"Saved error analysis to {out}")


if __name__ == "__main__":
    main()
