"""Exploratory data analysis: data quality, class balance, how the label
depends on time, and which columns leak the label.

Usage: python src/explore.py [--data PATH]
Writes outputs/data/{data_summary.md, class_distribution.png,
label_by_hour.png, single_feature.png}.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASS_ORDER, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TARGET, TIMESTAMP
from data_loader import load_data
from leakage_check import single_feature_scores
from md_table import to_markdown
from style import INK_SECONDARY, LEVEL_COLORS, SERIES, SURFACE, apply_style


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    per_int = df.groupby("intersection_id").size()
    return pd.DataFrame(
        {
            "value": [
                len(df),
                df.shape[1],
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                df["intersection_id"].nunique(),
                df["city_zone"].nunique(),
                f"{df[TIMESTAMP].min()} to {df[TIMESTAMP].max()}",
                f"{per_int.min()}-{per_int.max()}",
            ]
        },
        index=[
            "rows",
            "columns",
            "missing values",
            "duplicate rows",
            "intersections",
            "zones",
            "time range (hourly)",
            "rows per intersection",
        ],
    )


def share_table(df: pd.DataFrame, by) -> pd.DataFrame:
    return (pd.crosstab(df[by] if isinstance(by, str) else [df[b] for b in by], df[TARGET], normalize="index")
            .reindex(columns=CLASS_ORDER, fill_value=0).round(2))


def plot_class_distribution(df: pd.DataFrame, path: Path) -> None:
    share = df[TARGET].value_counts(normalize=True).reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bars = ax.bar(CLASS_ORDER, share, color=[LEVEL_COLORS[c] for c in CLASS_ORDER], width=0.6)
    for bar, v in zip(bars, share):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.1%}", ha="center", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Share of rows")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Class distribution of congestion_level")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_label_by_hour(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (label, part) in zip(axes, (("Weekdays", df[df.is_weekend == 0]), ("Weekends", df[df.is_weekend == 1]))):
        share = share_table(part, "hour")
        bottom = np.zeros(len(share))
        for cls in CLASS_ORDER:
            # White edges give the 2px gap between stacked segments.
            ax.bar(share.index, share[cls], bottom=bottom, color=LEVEL_COLORS[cls], width=0.85,
                   edgecolor=SURFACE, linewidth=1, label=cls)
            bottom += share[cls].to_numpy()
        ax.set_title(f"Label share by hour: {label}")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(visible=False)
    axes[0].set_ylabel("Share of rows")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_single_feature(scores: pd.DataFrame, majority: float, path: Path) -> None:
    scores = scores.iloc[::-1]
    kept = scores["status"] == "kept"
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(scores) + 1.2))
    y = np.arange(len(scores))
    ax.barh(y, scores["single_feature_accuracy"], height=0.7,
            color=np.where(kept, SERIES[0], SERIES[1]))
    ax.axvline(majority, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.text(majority, len(scores) - 0.3, " majority-class rate", fontsize=8, color=INK_SECONDARY, va="bottom")
    ax.set_yticks(y, scores["column"], fontsize=8)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("Accuracy of a depth-4 tree using this column alone (3-fold CV)")
    ax.set_title("How much each column alone reveals about the label")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])],
              labels=["kept", "dropped"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "data")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = load_data(args.data)
    weekly = df.assign(week=df[TIMESTAMP].dt.to_period("W").astype(str))
    evening = df[df.hour.isin([18, 19])]
    weekday_evening = evening[evening.is_weekend == 0]
    scores = single_feature_scores(df, sample=40000)
    majority = df[TARGET].value_counts(normalize=True).max()

    sections = [
        ("Data quality", quality_summary(df)),
        ("Class balance", df[TARGET].value_counts().reindex(CLASS_ORDER).to_frame("rows")
            .assign(share=lambda t: (t.rows / t.rows.sum()).round(3))),
        ("Label share by week (stability over time)", share_table(weekly, "week")),
        ("Label share by hour, weekdays", share_table(df[df.is_weekend == 0], "hour")),
        ("Label share by hour, weekends", share_table(df[df.is_weekend == 1], "hour")),
        ("18-19h: by weekend flag", share_table(evening, "is_weekend")),
        ("18-19h weekdays: by zone", share_table(weekday_evening, "city_zone")),
        ("18-19h weekdays: by weather", share_table(weekday_evening, "weather_condition")),
        ("18-19h weekdays: by accident", share_table(weekday_evening, "accident_reported")),
        ("18-19h weekdays: by public event", share_table(weekday_evening, "public_event")),
        ("Rare-event frequency", df[["accident_reported", "public_event", "construction_activity", "is_holiday",
                                    "emergency_vehicle_detected"]].mean().round(4).to_frame("share of rows")),
        ("Single-feature predictiveness", scores.set_index("column").round(3)),
    ]
    with open(out / "data_summary.md", "w", encoding="utf-8") as f:
        for title, table in sections:
            f.write(f"## {title}\n\n{to_markdown(table)}\n")
    print((out / "data_summary.md").read_text(encoding="utf-8"))

    plot_class_distribution(df, out / "class_distribution.png")
    plot_label_by_hour(df, out / "label_by_hour.png")
    plot_single_feature(scores, majority, out / "single_feature.png")
    print(f"Saved data summary and figures to {out}")


if __name__ == "__main__":
    main()
