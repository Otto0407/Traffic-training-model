"""Measure how much each column alone reveals about the label.

Fits a shallow decision tree on one column at a time and reports its
cross-validated accuracy. Columns far above the majority-class rate (~0.50)
carry strong label information; this is the evidence used to decide which
columns to drop in config.DROPPED_FEATURES.

Usage: python src/leakage_check.py [--data PATH] [--sample N]
"""
import argparse

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from config import DEFAULT_DATA_PATH, DROPPED_FEATURES, TARGET, TIMESTAMP
from data_loader import load_data


def single_feature_scores(df: pd.DataFrame, sample: int, seed: int = 0) -> pd.DataFrame:
    subset = df.sample(min(sample, len(df)), random_state=seed)
    dropped = {c: reason for reason, cols in DROPPED_FEATURES.items() for c in cols}
    rows = []
    for col in df.columns:
        if col in (TARGET, TIMESTAMP, "record_id"):
            continue
        X = subset[[col]]
        if not pd.api.types.is_numeric_dtype(X[col]):
            X = pd.get_dummies(X)
        acc = cross_val_score(DecisionTreeClassifier(max_depth=4), X, subset[TARGET], cv=3).mean()
        rows.append({"column": col, "single_feature_accuracy": acc, "status": dropped.get(col, "kept")})
    return pd.DataFrame(rows).sort_values("single_feature_accuracy", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--sample", type=int, default=40000)
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Majority-class rate: {df[TARGET].value_counts(normalize=True).max():.3f}\n")
    print(single_feature_scores(df, args.sample).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
