"""Which input features does the MLP actually need? (drop-column importance)

Stage 1: retrain without each feature, and without each feature group, and
         compare validation macro F1 with the full model.
Stage 2: backward elimination. Starting from the features whose removal did
         not hurt, remove them one at a time (least useful first) and keep a
         removal only if validation macro F1 stays within the seed spread of
         the full model. This handles redundant features (e.g. hour,
         peak_period and rush_hour carry overlapping information), which a
         one-at-a-time test cannot.
Stage 3: score the full and the reduced feature set on the test set, once.

All choices use validation macro F1 only. Every setting uses several seeds.

Usage: python src/feature_ablation.py [--seeds 42 43 44]
Writes outputs/feature_ablation/{results.csv, single.md, elimination.md,
final.md, drop_importance.png}.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cleaning import clean_data
from config import DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR, TrainConfig
from data_loader import load_data
from evaluate import compute_metrics, predict
from features import LEAK_FREE
from md_table import to_markdown
from pipeline import fit_model, prepare_data
from style import INK_SECONDARY, SERIES, apply_style

GROUPS = {
    "time": ["hour", "day_of_week", "is_weekend", "is_holiday", "rush_hour", "peak_period"],
    "location_road": ["city_zone", "road_type", "lanes", "speed_limit", "traffic_camera_count",
                      "nearby_school", "nearby_hospital", "nearby_market"],
    "weather": ["weather_condition", "temperature", "rainfall_mm", "visibility_km", "road_condition"],
    "events": ["construction_activity", "public_event", "accident_reported", "emergency_vehicle_detected"],
    "sensor_parking": ["iot_sensor_health", "parking_occupancy"],
}
MIN_THRESHOLD = 0.0005


def evaluate_set(df, removed: tuple, seeds, device, score_test: bool = False) -> dict:
    features = LEAK_FREE.without(*removed)
    runs = []
    for seed in seeds:
        cfg = replace(TrainConfig(), seed=seed)
        data = prepare_data(df, cfg, features)
        model, _ = fit_model(data, cfg, device, verbose=False)
        run = {"val_macro_f1": compute_metrics(data.y_val, predict(model, data.X_val, device))["f1_macro"]}
        if score_test:
            test = compute_metrics(data.y_test, predict(model, data.X_test, device))
            run.update({"test_macro_f1": test["f1_macro"], "test_accuracy": test["accuracy"], "test_qwk": test["qwk"]})
        runs.append(run)
    runs = pd.DataFrame(runs)
    out = {"n_features": len(features.columns), "input_dim": data.X_train.shape[1]}
    for col in runs:
        out[f"{col}_mean"], out[f"{col}_std"] = runs[col].mean(), runs[col].std()
    return out


def plot_importance(single: pd.DataFrame, threshold: float, path: Path) -> None:
    apply_style()
    single = single.sort_values("diff")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(single) + 1.4))
    y = np.arange(len(single))
    colors = [SERIES[1] if d < -threshold else SERIES[0] for d in single["diff"]]
    ax.barh(y, single["diff"], xerr=single["std"], color=colors, height=0.65,
            error_kw={"ecolor": INK_SECONDARY, "lw": 1})
    ax.axvspan(-threshold, threshold, color=INK_SECONDARY, alpha=0.12, lw=0)
    ax.axvline(0, color=INK_SECONDARY, lw=1)
    ax.set_yticks(y, single["removed"], fontsize=8)
    ax.set_xscale("symlog", linthresh=0.002)
    ax.set_xlabel("Change in validation macro F1 when removed (symlog scale; band = seed spread)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES[1]), plt.Rectangle((0, 0), 1, 1, color=SERIES[0])],
              labels=["hurts when removed (needed)", "no clear effect"], loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "feature_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, _ = clean_data(load_data(args.data))
    rows = []

    def record(stage, name, removed, res):
        rows.append({"stage": stage, "removed": name, "removed_features": ",".join(removed), **res})
        pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        print(f"[{stage}] {name:<34} dim {res['input_dim']:>3}  val macro F1 "
              f"{res['val_macro_f1_mean']:.4f} ± {res['val_macro_f1_std']:.4f}", flush=True)

    # Stage 1: one feature / one group at a time
    full = evaluate_set(df, (), args.seeds, device)
    record("full", "(none)", (), full)
    base, threshold = full["val_macro_f1_mean"], max(full["val_macro_f1_std"], MIN_THRESHOLD)
    single = []
    for feature in LEAK_FREE.columns:
        res = evaluate_set(df, (feature,), args.seeds, device)
        record("single", feature, (feature,), res)
        single.append({"removed": feature, "diff": res["val_macro_f1_mean"] - base, "std": res["val_macro_f1_std"],
                       "input_dim": res["input_dim"]})
    for group, members in GROUPS.items():
        res = evaluate_set(df, tuple(members), args.seeds, device)
        record("group", f"group: {group}", tuple(members), res)
        single.append({"removed": f"group: {group}", "diff": res["val_macro_f1_mean"] - base,
                       "std": res["val_macro_f1_std"], "input_dim": res["input_dim"]})
    single = pd.DataFrame(single)
    plot_importance(single, threshold, out / "drop_importance.png")
    with open(out / "single.md", "w", encoding="utf-8") as f:
        f.write(f"Full model: validation macro F1 {base:.4f} ± {full['val_macro_f1_std']:.4f} "
                f"({full['input_dim']} inputs). Removals that lower it by more than {threshold:.4f} hurt.\n\n")
        f.write(to_markdown(single.sort_values("diff").round(4), index=False))

    # Stage 2: backward elimination over features that did not hurt on their own
    candidates = single[~single["removed"].str.startswith("group:") & (single["diff"] >= -threshold)]
    candidates = candidates.sort_values("diff", ascending=False)["removed"].tolist()
    removed, steps = [], []
    for feature in candidates:
        trial = tuple(removed + [feature])
        res = evaluate_set(df, trial, args.seeds, device)
        keep_removal = res["val_macro_f1_mean"] >= base - threshold
        record("elimination", f"+ drop {feature}" + ("" if keep_removal else " (reverted)"), trial, res)
        steps.append({"try_removing": feature, "val_macro_f1": round(res["val_macro_f1_mean"], 4),
                      "input_dim": res["input_dim"], "kept_removed": keep_removal})
        if keep_removal:
            removed.append(feature)
    with open(out / "elimination.md", "w", encoding="utf-8") as f:
        f.write(f"Removal kept if validation macro F1 >= {base - threshold:.4f} (full model minus seed spread).\n\n")
        f.write(to_markdown(pd.DataFrame(steps), index=False))
        f.write(f"\nRemoved ({len(removed)}): {', '.join(removed) or 'none'}\n")
        f.write(f"Kept ({len(LEAK_FREE.columns) - len(removed)}): "
                f"{', '.join(c for c in LEAK_FREE.columns if c not in removed)}\n")

    # Stage 3: test set, once, for the full and the reduced feature set
    final = []
    for name, drop in (("full", ()), ("reduced", tuple(removed))):
        res = evaluate_set(df, drop, args.seeds, device, score_test=True)
        final.append({"feature_set": name, "n_features": res["n_features"], "input_dim": res["input_dim"],
                      **{k: f"{res[k + '_mean']:.4f} ± {res[k + '_std']:.4f}"
                         for k in ("val_macro_f1", "test_macro_f1", "test_accuracy", "test_qwk")}})
    final = pd.DataFrame(final)
    with open(out / "final.md", "w", encoding="utf-8") as f:
        f.write("Full vs reduced feature set (mean ± std over seeds). The reduced set was chosen on validation\n")
        f.write("macro F1 only; the test columns were computed after that choice.\n\n")
        f.write(to_markdown(final, index=False))
        f.write(f"\nRemoved features: {', '.join(removed) or 'none'}\n")
    print("\n" + final.to_string(index=False))
    print(f"\nRemoved features: {', '.join(removed) or 'none'}")
    print(f"Saved feature ablation results to {out}")


if __name__ == "__main__":
    main()
