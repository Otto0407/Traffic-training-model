"""Training curves and confusion matrix figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CLASS_ORDER


def plot_history(history: dict, path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((ax_loss, "loss", "Loss"), (ax_acc, "acc", "Accuracy")):
        ax.plot(epochs, history[f"train_{key}"], label="train")
        ax.plot(epochs, history[f"val_{key}"], label="validation")
        ax.axvline(history["best_epoch"], color="grey", linestyle="--", linewidth=1, label="best epoch")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list, path: Path, title: str = "Test confusion matrix") -> None:
    cm = np.asarray(cm)
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if row_share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{row_share[i, j]:.0%}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Share of actual class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
