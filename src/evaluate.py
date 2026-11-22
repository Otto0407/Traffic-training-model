"""Prediction and test-set metrics."""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import CLASS_ORDER


@torch.no_grad()
def predict(model, X: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, macro- and weighted-averaged precision, recall and F1, plus
    ordinal metrics.

    Macro F1 is the primary metric: it weights the four classes equally, so a
    model cannot score well by only predicting the common classes. The labels
    are ordered (Low < Moderate < High < Severe), which F1 ignores, so the
    quadratic weighted kappa (QWK) and the mean distance in levels between
    prediction and truth are reported too: predicting Severe for a Low hour
    costs more than predicting Moderate.
    """
    labels = list(range(len(CLASS_ORDER)))
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        metrics.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
    level_error = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    metrics["mean_level_error"] = float(level_error.mean())
    metrics["errors_two_or_more_levels"] = int((level_error >= 2).sum())
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    metrics["report"] = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASS_ORDER, zero_division=0, digits=4
    )
    return metrics
