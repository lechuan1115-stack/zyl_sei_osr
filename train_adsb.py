#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Complete training pipeline for the ADS-B signal classification task."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from Confusion_matrix import plot_confusion_matrix
from mydata_read import ADSBSignalDataset, create_datasets
from mymodel1 import create as create_model
from pytorchtools import EarlyStopping
from utils import AverageMeter

from open_set import OpenSetSplit, determine_open_set_split, filter_by_classes
from open_set_pipeline import run_open_set_pipeline

# ---------------------------------------------------------------------------
# Configuration and reproducibility helpers
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Container holding the full training configuration.

    All hyper-parameters can be tweaked by modifying the dataclass fields below.
    This avoids the need for command-line arguments and makes experiments fully
    reproducible from within the script itself.
    """

    train_data: Path
    test_data: Path
    output_dir: Path = Path("training_outputs")
    epochs: int = 80
    batch_size: int = 128
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 5e-5
    warmup_epochs: int = 5
    patience: int = 20
    min_epochs: int = 20
    early_stop_delta: float = 1e-4
    num_workers: int = 0
    val_ratio: float = 0.1
    feature_key: str | None = None
    label_key: str | None = None
    test_feature_key: str | None = None
    test_label_key: str | None = None
    seed: int = 42
    known_class_ratio: float = 0.7
    openmax_tail_size: int = 20
    openmax_alpha: int = 3
    gan_latent_dim: int = 128
    gan_hidden_dim: int = 1024
    gan_epochs: int = 80
    gan_batch_size: int = 128
    tsne_samples: int = 2000


DEFAULT_CONFIG = TrainingConfig(
    train_data=Path(r"E:\数据集\ADS-B_Train_10X360-2_5-10-15-20dB.mat"),
    test_data=Path(r"E:\数据集\ADS-B_Test_10X360-2_5-10-15-20dB.mat"),
)


@dataclass
class PreparedData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_known_loader: DataLoader
    test_unknown_loader: DataLoader
    train_dataset: ADSBSignalDataset
    val_dataset: ADSBSignalDataset
    open_split: OpenSetSplit
    unknown_pool_data: np.ndarray
    unknown_pool_labels: np.ndarray
    known_train_data: np.ndarray
    known_train_labels: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataloader creation
# ---------------------------------------------------------------------------


def prepare_dataloaders(config: TrainingConfig) -> PreparedData:
    train_split, val_split, test_split = create_datasets(
        config.train_data,
        test_mat_path=config.test_data,
        val_ratio=config.val_ratio,
        feature_key=config.feature_key,
        label_key=config.label_key,
        test_feature_key=config.test_feature_key,
        test_label_key=config.test_label_key,
        random_state=config.seed,
    )

    open_split = determine_open_set_split(train_split.labels, config.known_class_ratio)

    class_to_index = {cls: idx for idx, cls in enumerate(open_split.known_classes)}

    def remap(labels: np.ndarray) -> np.ndarray:
        return np.vectorize(class_to_index.__getitem__, otypes=[np.int64])(labels)

    known_train_data, known_train_labels_raw = filter_by_classes(
        train_split.data, train_split.labels, open_split.known_classes
    )
    known_val_data, known_val_labels_raw = filter_by_classes(
        val_split.data, val_split.labels, open_split.known_classes
    )
    known_test_data, known_test_labels_raw = filter_by_classes(
        test_split.data, test_split.labels, open_split.known_classes
    )

    known_train_labels = remap(known_train_labels_raw)
    known_val_labels = remap(known_val_labels_raw)
    known_test_labels = remap(known_test_labels_raw)

    unknown_train_data, unknown_train_labels = filter_by_classes(
        train_split.data, train_split.labels, open_split.unknown_classes
    )
    unknown_val_data, unknown_val_labels = filter_by_classes(
        val_split.data, val_split.labels, open_split.unknown_classes
    )
    unknown_test_data, unknown_test_labels = filter_by_classes(
        test_split.data, test_split.labels, open_split.unknown_classes
    )

    if unknown_test_data.size == 0:
        raise RuntimeError(
            "No samples found for the unknown classes. Adjust `known_class_ratio` to keep at least one unknown class."
        )

    train_dataset = ADSBSignalDataset(known_train_data, known_train_labels)
    val_dataset = ADSBSignalDataset(known_val_data, known_val_labels)
    known_test_dataset = ADSBSignalDataset(known_test_data, known_test_labels)
    unknown_test_dataset = ADSBSignalDataset(unknown_test_data, unknown_test_labels)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    known_test_loader = DataLoader(known_test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    unknown_test_loader = DataLoader(unknown_test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    unknown_sources = [
        arr
        for arr in [unknown_train_data, unknown_val_data, unknown_test_data]
        if arr.size > 0
    ]
    unknown_label_sources = [
        arr
        for arr in [unknown_train_labels, unknown_val_labels, unknown_test_labels]
        if arr.size > 0
    ]
    unknown_pool_data = (
        np.concatenate(unknown_sources, axis=0)
        if unknown_sources
        else np.empty((0,) + train_split.data.shape[1:], dtype=train_split.data.dtype)
    )
    unknown_pool_labels = (
        np.concatenate(unknown_label_sources, axis=0)
        if unknown_label_sources
        else np.empty((0,), dtype=train_split.labels.dtype)
    )

    return PreparedData(
        train_loader=train_loader,
        val_loader=val_loader,
        test_known_loader=known_test_loader,
        test_unknown_loader=unknown_test_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        open_split=open_split,
        unknown_pool_data=unknown_pool_data,
        unknown_pool_labels=unknown_pool_labels,
        known_train_data=known_train_data,
        known_train_labels=known_train_labels,
    )


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


def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, config: TrainingConfig) -> float:
    if epoch <= config.warmup_epochs:
        lr = config.lr * epoch / max(1, config.warmup_epochs)
    else:
        progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
        lr = config.min_lr + 0.5 * (config.lr - config.min_lr) * (1 + math.cos(math.pi * progress))

    lr = max(lr, config.min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


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


def main(config: TrainingConfig = DEFAULT_CONFIG) -> None:
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_dataloaders(config)

    train_loader = prepared.train_loader
    val_loader = prepared.val_loader
    known_test_loader = prepared.test_known_loader
    unknown_test_loader = prepared.test_unknown_loader

    num_classes = len(prepared.open_split.known_classes)
    model = create_model("CNN_Transformer", num_classes=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    early_stopping = EarlyStopping(
        patience=config.patience,
        verbose=True,
        path=str(config.output_dir / "best_model.pt"),
        delta=config.early_stop_delta,
    )

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, config.epochs + 1):
        current_lr = adjust_learning_rate(optimizer, epoch, config)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop and epoch < config.min_epochs:
            early_stopping.early_stop = False
            early_stopping.counter = 0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
        )

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Restore the best checkpoint (lowest validation loss).
    best_model_path = config.output_dir / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    plot_training_curves(history, config.output_dir)

    # Evaluate on the held-out test set.
    known_loss, known_acc, y_true_known, y_pred_known = evaluate(
        model, known_test_loader, criterion, device, collect_predictions=True
    )
    print(f"Known-class test loss: {known_loss:.4f} | Known-class accuracy: {known_acc:.4f}")

    # Persist metrics and detailed reports for later inspection.
    metrics = {
        "train_history": history,
        "known_test_loss": known_loss,
        "known_test_accuracy": known_acc,
        "open_set_split": prepared.open_split.as_dict(),
    }
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    class_mapping = prepared.open_split.known_classes
    target_names = [f"Class {cls}" for cls in class_mapping]
    report = classification_report(
        y_true_known,
        y_pred_known,
        labels=list(range(len(class_mapping))),
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    with open(config.output_dir / "classification_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    class_names = target_names
    fig = plot_confusion_matrix(
        y_true_known,
        y_pred_known,
        class_names=class_names,
        normalize=True,
        title="Normalised confusion matrix (known classes)",
        save_path=config.output_dir / "confusion_matrix.png",
    )
    plt.close(fig)

    # Also store the raw confusion matrix for completeness.
    fig = plot_confusion_matrix(
        y_true_known,
        y_pred_known,
        class_names=class_names,
        normalize=False,
        title="Confusion matrix (counts, known classes)",
        save_path=config.output_dir / "confusion_matrix_counts.png",
    )
    plt.close(fig)

    # ----------------------- Open-set evaluation -------------------------
    run_open_set_pipeline(
        model,
        train_loader,
        known_test_loader,
        unknown_test_loader,
        unknown_pool=prepared.unknown_pool_data,
        num_classes=num_classes,
        open_split=prepared.open_split,
        config=config,
        device=device,
        output_dir=config.output_dir,
    )

    print(
        "Open-set evaluation complete. Metrics saved to",
        config.output_dir / "open_set_metrics.json",
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
