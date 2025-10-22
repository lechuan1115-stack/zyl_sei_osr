#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Complete training pipeline for the ADS-B signal classification task."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
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
        "--train-data",
        type=Path,
        required=True,
        help=(
            "Path to the MATLAB v7.3 (.mat) file containing the training "
            "samples (will be split into train/validation)."
        ),
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="Path to the MATLAB v7.3 (.mat) file containing the held-out test set.",
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
        "--lr-factor",
        type=float,
        default=0.5,
        help=(
            "Multiplicative factor applied to the learning rate when the validation "
            "loss stagnates."
        ),
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=4,
        help="Number of epochs with no improvement before the learning rate is reduced.",
    )
    parser.add_argument(
        "--lr-threshold",
        type=float,
        default=1e-4,
        help="Minimum relative improvement in validation loss to avoid LR reduction.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Lower bound applied to the adaptive learning rate scheduler.",
    )
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
        "--early-stop-delta",
        type=float,
        default=1e-4,
        help="Minimum absolute validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes used by the dataloaders.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the training set used for validation (0 < ratio < 1).",
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
        "--test-feature-key",
        type=str,
        default=None,
        help="Optional dataset key containing the test feature tensor.",
    )
    parser.add_argument(
        "--test-label-key",
        type=str,
        default=None,
        help="Optional dataset key containing the test labels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting and weight initialisation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help=(
            "Print intermediate training statistics every N batches. "
            "Set to 0 to disable in-epoch logging."
        ),
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
    train_mat_path: Path,
    *,
    test_mat_path: Path,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    feature_key: str | None,
    label_key: str | None,
    test_feature_key: str | None,
    test_label_key: str | None,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_split, val_split, test_split = create_datasets(
        train_mat_path,
        test_mat_path=test_mat_path,
        val_ratio=val_ratio,
        feature_key=feature_key,
        label_key=label_key,
        test_feature_key=test_feature_key,
        test_label_key=test_label_key,
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
# Logging helpers
# ---------------------------------------------------------------------------


def _summarise_split(name: str, dataset: ADSBSignalDataset) -> Dict[str, object]:
    """Log and return a concise summary for a dataset split."""

    labels = dataset.labels.cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {int(label): int(count) for label, count in zip(unique, counts)}

    summary = {
        "name": name,
        "num_samples": int(len(dataset)),
        "class_distribution": distribution,
    }

    print(
        f"{name}: {summary['num_samples']:6d} samples | "
        f"classes: {sorted(distribution.keys())}"
    )
    print(f"        Class distribution: {distribution}")
    return summary


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    log_interval: int,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
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

        if log_interval > 0 and batch_idx % log_interval == 0:
            print(
                f"  [Epoch {epoch:03d}/{total_epochs:03d} | Batch {batch_idx:04d}/"
                f"{len(dataloader):04d}] "
                f"loss={loss_meter.avg:.4f} acc={acc_meter.avg * 100:.2f}%"
            )

    return loss_meter.avg, acc_meter.avg


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    collect_details: bool = False,
) -> Tuple[float, float, Dict[str, np.ndarray] | None]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs, features = model(inputs)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            batch_acc = preds.eq(targets).float().mean().item()
            loss_meter.update(loss.item(), inputs.size(0))
            acc_meter.update(batch_acc, inputs.size(0))

            if collect_details:
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_features.append(features.cpu().numpy())
                all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())

    details = None
    if collect_details:
        details = {
            "y_true": np.concatenate(all_targets),
            "y_pred": np.concatenate(all_preds),
            "features": np.concatenate(all_features),
            "probabilities": np.concatenate(all_probs),
        }

    return loss_meter.avg, acc_meter.avg, details


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


def plot_tsne_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    max_points: int = 5000,
) -> None:
    """Project the high-dimensional features to 2-D with t-SNE."""

    if features.shape[0] < 2:
        return

    rng = np.random.default_rng(seed)
    if features.shape[0] > max_points:
        indices = rng.choice(features.shape[0], size=max_points, replace=False)
        features = features[indices]
        labels = labels[indices]

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    max_valid = max(1, features_std.shape[0] - 1)
    perplexity = min(30, max_valid)
    perplexity = max(perplexity, min(5, max_valid))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    embeddings = tsne.fit_transform(features_std)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.get_cmap("tab10", int(labels.max() + 1))
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap=cmap,
        s=10,
        alpha=0.8,
    )
    legend_handles = scatter.legend_elements()[0]
    class_names = [f"Class {idx}" for idx in np.unique(labels)]
    ax.legend(legend_handles, class_names, title="Classes", loc="best", fontsize="small")
    ax.set_title("t-SNE projection of test embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne_embeddings.png", dpi=300)
    plt.close(fig)


def plot_distance_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    *,
    seed: int,
    num_pairs: int = 20000,
) -> None:
    """Visualise intra/inter class distance distributions."""

    if features.shape[0] < 2:
        return

    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, features.shape[0], size=num_pairs)
    idx2 = rng.integers(0, features.shape[0], size=num_pairs)

    same_class = labels[idx1] == labels[idx2]
    valid = idx1 != idx2
    same_class &= valid
    different_class = (~same_class) & valid

    diffs = features[idx1] - features[idx2]
    dists = np.linalg.norm(diffs, axis=1)

    intra = dists[same_class]
    inter = dists[different_class]

    if intra.size == 0 or inter.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(intra, bins=50, alpha=0.6, label="Intra-class", color="#1f77b4", density=True)
    ax.hist(inter, bins=50, alpha=0.6, label="Inter-class", color="#ff7f0e", density=True)
    ax.set_title("Feature distance distributions")
    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distance_distribution.png", dpi=300)
    plt.close(fig)


def plot_per_class_accuracy(report: Dict[str, dict], output_dir: Path) -> None:
    """Render a bar chart for per-class recall (accuracy)."""

    class_ids = sorted(k for k in report.keys() if k.isdigit())
    if not class_ids:
        return

    recalls = [report[k]["recall"] for k in class_ids]
    labels = [f"Class {int(k)}" for k in class_ids]

    x = np.arange(len(labels))
    colors = plt.cm.tab10.colors[: len(labels)]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, recalls, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Per-class accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.bar_label(bars, labels=[f"{recall*100:.1f}%" for recall in recalls], padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_accuracy.png", dpi=300)
    plt.close(fig)


def plot_confidence_histogram(
    probabilities: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot a histogram of prediction confidences."""

    confidences = probabilities.max(axis=1)
    correct_mask = y_pred == y_true

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        confidences[correct_mask],
        bins=30,
        alpha=0.7,
        label="Correct",
        color="#2ca02c",
        range=(0, 1),
    )
    ax.hist(
        confidences[~correct_mask],
        bins=30,
        alpha=0.7,
        label="Incorrect",
        color="#d62728",
        range=(0, 1),
    )
    ax.set_title("Prediction confidence distribution")
    ax.set_xlabel("Maximum softmax confidence")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_histogram.png", dpi=300)
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
        args.train_data,
        test_mat_path=args.test_data,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_key=args.feature_key,
        label_key=args.label_key,
        test_feature_key=args.test_feature_key,
        test_label_key=args.test_label_key,
        seed=args.seed,
    )

    split_summaries = [
        _summarise_split("Train", train_loader.dataset),
        _summarise_split("Valid", val_loader.dataset),
        _summarise_split("Test", test_loader.dataset),
    ]

    num_classes = len(np.unique(train_loader.dataset.labels.cpu().numpy()))
    model = create_model("CNN_Transformer", num_classes=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.lr_threshold,
        threshold_mode="rel",
        min_lr=args.min_lr,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        path=str(args.output_dir / "best_model.pt"),
        delta=args.early_stop_delta,
    )

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_snapshot = {
        "epoch": 0,
        "val_loss": float("inf"),
        "val_acc": 0.0,
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )
        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_snapshot["val_loss"]:
            best_snapshot.update({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr={lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% | "
            f"best_val_acc={best_snapshot['val_acc'] * 100:.2f}% (epoch {best_snapshot['epoch']:03d}) | "
            f"elapsed={elapsed:.1f}s"
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
    test_loss, test_acc, test_details = evaluate(
        model, test_loader, criterion, device, collect_details=True
    )
    assert test_details is not None
    y_true = test_details["y_true"]
    y_pred = test_details["y_pred"]
    features = test_details["features"]
    probabilities = test_details["probabilities"]
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # Persist metrics and detailed reports for later inspection.
    metrics = {
        "train_history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "dataset_summaries": split_summaries,
        "best_epoch": best_snapshot,
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

    plot_per_class_accuracy(report, args.output_dir)

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

    plot_tsne_embeddings(features, y_true, args.output_dir, seed=args.seed)
    plot_distance_distributions(features, y_true, args.output_dir, seed=args.seed)
    plot_confidence_histogram(probabilities, y_pred, y_true, args.output_dir)

    print("Training artefacts written to", args.output_dir.resolve())


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
