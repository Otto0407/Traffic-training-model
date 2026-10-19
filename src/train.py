"""Mini-batch training loop with per-epoch validation and early stopping."""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised so they average to 1."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.as_tensor(weights, dtype=torch.float32)


def _run_epoch(model, loader, criterion, device, optimizer=None, noise_std=0.0, num_numeric=0) -> tuple:
    """One pass over loader; trains when an optimizer is given. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if training and noise_std > 0 and num_numeric > 0:
                # Augmentation: jitter the standardised numerical features only;
                # one-hot and binary columns stay exact.
                xb = xb.clone()
                xb[:, :num_numeric] += noise_std * torch.randn_like(xb[:, :num_numeric])
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            seen += len(yb)
    return total_loss / seen, correct / seen


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg,
    device: torch.device,
    loss_weights: torch.Tensor = None,
    num_numeric: int = 0,
    verbose: bool = True,
) -> tuple:
    """Train until validation loss stops improving for cfg.patience epochs.

    Returns the model with its best weights restored, plus the per-epoch history.
    """
    model.to(device)
    weight = loss_weights.to(device) if loss_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer, cfg.input_noise, num_numeric
        )
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        if verbose:
            print(
                f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                f"| val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"Early stopping: no val loss improvement for {cfg.patience} epochs.")
                break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    if verbose:
        print(f"Restored best weights from epoch {best_epoch} (val loss {best_loss:.4f}).")
    return model, history
