"""Compare the MLP with baselines and standard tabular models.

Selection uses validation macro F1 only; the test set is scored once per
model, with 95% confidence intervals from a bootstrap over test hours.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from baseline import lookup_baseline, majority_baseline
from bootstrap import bootstrap_ci
from config import TIMESTAMP
from evaluate import compute_metrics

LOOKUP_KEYS = ["hour", "is_weekend", "city_zone"]


def _macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2, 3], zero_division=0)


def _accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3], weights="quadratic")


def baseline_predictions(data) -> dict:
    """{name: (val_pred, test_pred)} for the non-learned baselines."""
    return {
        "Majority class": (
            majority_baseline(data.y_train, len(data.y_val)),
            majority_baseline(data.y_train, len(data.y_test)),
        ),
        "Lookup: hour x weekend x zone": (
            lookup_baseline(data.train_df, data.val_df, LOOKUP_KEYS),
            lookup_baseline(data.train_df, data.test_df, LOOKUP_KEYS),
        ),
    }


def compare(data, predictions: dict, reference: str = "MLP", n_boot: int = 1000) -> tuple:
    """predictions: {name: (val_pred, test_pred)}.

    Returns (table, details): the table has validation macro F1 and test
    metrics with 95% CIs; details holds full test metrics per model and the
    paired bootstrap difference in macro F1 between `reference` and each model.
    """
    groups = data.test_df[TIMESTAMP].to_numpy()
    rows, details = [], {}
    for name, (val_pred, test_pred) in predictions.items():
        test = compute_metrics(data.y_test, test_pred)
        row = {"model": name, "val_macro_f1": round(_macro_f1(data.y_val, val_pred), 4)}
        for label, fn in (("macro_f1", _macro_f1), ("accuracy", _accuracy), ("qwk", _qwk)):
            point, low, high = bootstrap_ci(data.y_test, test_pred, groups, fn, n_boot=n_boot)
            row[f"test_{label}"] = f"{point:.3f} [{low:.3f}, {high:.3f}]"
        row["test_mean_level_error"] = round(test["mean_level_error"], 4)
        row["test_errors_2+_levels"] = test["errors_two_or_more_levels"]
        rows.append(row)
        details[name] = {k: v for k, v in test.items() if k != "report"}

    # Paired comparison: resample the same test hours for both models.
    ref_pred = predictions[reference][1]
    paired = {}
    for name, (_, test_pred) in predictions.items():
        if name == reference:
            continue
        both = np.column_stack([ref_pred, test_pred])
        diff, low, high = bootstrap_ci(
            data.y_test, both, groups,
            lambda y, p: _macro_f1(y, p[:, 0]) - _macro_f1(y, p[:, 1]), n_boot=n_boot,
        )
        paired[name] = {"difference": diff, "ci_low": low, "ci_high": high,
                        "significant": bool(low > 0 or high < 0)}
    details["paired_macro_f1_difference_vs_" + reference] = paired

    table = pd.DataFrame(rows).set_index("model").sort_values("val_macro_f1", ascending=False)
    return table, details
