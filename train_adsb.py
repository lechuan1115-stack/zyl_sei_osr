#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Complete training pipeline for the ADS-B signal classification task."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from Confusion_matrix import plot_confusion_matrix
from mydata_read import ADSBSignalDataset, create_datasets
from mymodel1 import create as create_model
from pytorchtools import EarlyStopping
from utils import AverageMeter

# ---------------------------------------------------------------------------
# Argument parsing and reproducibility helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the ADS-B MATLAB v7.3 (.mat) file containing the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_outputs"),
        help="Directory where models, logs and figures will be stored.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay used by the AdamW optimiser.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="Number of epochs with no validation improvement before early stopping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes used by the dataloaders.",
    )
    parser.add_argument(
        "--feature-key",
        type=str,
        default=None,
        help="Optional dataset key containing the feature tensor.  Leave unset to auto-detect.",
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default=None,
        help="Optional dataset key containing the labels.  Leave unset to auto-detect.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting and weight initialisation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataloader creation
# ---------------------------------------------------------------------------


def create_dataloaders(
    mat_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    feature_key: str | None,
    label_key: str | None,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_split, val_split, test_split = create_datasets(
        mat_path,
        feature_key=feature_key,
        label_key=label_key,
        random_state=seed,
    )

    train_dataset = ADSBSignalDataset(train_split.data, train_split.labels)
    val_dataset = ADSBSignalDataset(val_split.data, val_split.labels)
    test_dataset = ADSBSignalDataset(test_split.data, test_split.labels)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_acc = preds.eq(targets).float().mean().item()
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(batch_acc, inputs.size(0))

    return loss_meter.avg, acc_meter.avg


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    collect_predictions: bool = False,
) -> Tuple[float, float, np.ndarray | None, np.ndarray | None]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            batch_acc = preds.eq(targets).float().mean().item()
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(batch_acc, inputs.size(0))

            if collect_predictions:
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

    if collect_predictions:
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
    else:
        y_true = y_pred = None

    return loss_meter.avg, acc_meter.avg, y_true, y_pred


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def plot_training_curves(history: Dict[str, list], output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, history["train_loss"], label="Train", marker="o")
    ax[0].plot(epochs, history["val_loss"], label="Validation", marker="o")
    ax[0].set_title("Loss over epochs")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Cross-entropy loss")
    ax[0].grid(True, linestyle="--", alpha=0.5)
    ax[0].legend()

    ax[1].plot(epochs, history["train_acc"], label="Train", marker="o")
    ax[1].plot(epochs, history["val_acc"], label="Validation", marker="o")
    ax[1].set_title("Accuracy over epochs")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True, linestyle="--", alpha=0.5)
    ax[1].legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "training_curves.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_key=args.feature_key,
        label_key=args.label_key,
        seed=args.seed,
    )

    num_classes = len(np.unique(train_loader.dataset.labels.cpu().numpy()))
    model = create_model("CNN_Transformer", num_classes=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, verbose=True
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=str(args.output_dir / "best_model.pt"),
    )

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Restore the best checkpoint (lowest validation loss).
    best_model_path = args.output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    plot_training_curves(history, args.output_dir)

    # Evaluate on the held-out test set.
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, collect_predictions=True
    )
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # Persist metrics and detailed reports for later inspection.
    metrics = {
        "train_history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    with open(args.output_dir / "classification_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    class_names = [f"Class {idx}" for idx in sorted(np.unique(y_true))]
    fig = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        normalize=True,
        title="Normalised confusion matrix",
        save_path=args.output_dir / "confusion_matrix.png",
    )
    plt.close(fig)

    # Also store the raw confusion matrix for completeness.
    fig = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        normalize=False,
        title="Confusion matrix (counts)",
        save_path=args.output_dir / "confusion_matrix_counts.png",
    )
    plt.close(fig)

    print("Training artefacts written to", args.output_dir.resolve())


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
