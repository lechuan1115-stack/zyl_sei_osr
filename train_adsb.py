#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Complete training pipeline for the ADS-B signal classification task."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
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
from mymodel1 import CNN_Transformer
from pytorchtools import EarlyStopping
from torch.nn.utils import clip_grad_norm_
from utils import AverageMeter

# ---------------------------------------------------------------------------
# Configuration and reproducibility helpers
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Runtime configuration loaded before kicking off training.

    The defaults are ready for the ADS-B dataset described in the project
    README.  Adjust the paths or any hyper-parameters below and simply run the
    script without additional command-line arguments.
    """

    train_data: Path = Path(r"E:\数据集\ADS-B_Train_10X360-2_5-10-15-20dB.mat")
    test_data: Path = Path(r"E:\数据集\ADS-B_Test_10X360-2_5-10-15-20dB.mat")
    output_dir: Path = Path("training_outputs")
    epochs: int = 180
    batch_size: int = 128
    lr: float = 8e-4
    min_lr: float = 5e-5
    weight_decay: float = 5e-5
    warmup_epochs: int = 12
    patience: int = 60
    min_epochs: int = 70
    early_stop_delta: float = 5e-5
    max_grad_norm: float = 2.0
    num_workers: int = 0
    val_ratio: float = 0.1
    feature_key: str | None = None
    label_key: str | None = None
    test_feature_key: str | None = None
    test_label_key: str | None = None
    seed: int = 42
    log_interval: int = 50
    label_smoothing: float = 0.03

    def __post_init__(self) -> None:
        # Allow users to provide string paths in custom configurations.
        self.train_data = Path(self.train_data)
        self.test_data = Path(self.test_data)
        self.output_dir = Path(self.output_dir)

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if self.min_epochs < 1:
            raise ValueError("min_epochs must be at least 1")
        if self.epochs < self.min_epochs:
            raise ValueError("epochs must be >= min_epochs")
        if self.warmup_epochs >= self.epochs:
            self.warmup_epochs = max(0, self.epochs - 1)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1).")


CONFIG_PATH = Path("training_config.json")


def load_config() -> TrainingConfig:
    """Load configuration from ``training_config.json`` if present.

    Users who prefer editing JSON can copy the generated template, otherwise the
    defaults defined in :class:`TrainingConfig` are used.
    """

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
            user_config = json.load(fp)

        valid_keys = TrainingConfig.__dataclass_fields__.keys()
        filtered_config = {k: v for k, v in user_config.items() if k in valid_keys}
        if len(filtered_config) != len(user_config):
            unknown = sorted(set(user_config) - set(filtered_config))
            if unknown:
                print(
                    "Ignoring unsupported configuration keys:",
                    ", ".join(unknown),
                )
        return TrainingConfig(**filtered_config)

    # When no user configuration exists we still provide a template so that the
    # required keys are obvious for the next run.
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w", encoding="utf-8") as fp:
            json.dump(TrainingConfig().__dict__, fp, indent=2, default=str)
        print(
            "Generated training_config.json with default values. "
            "Update the dataset paths before the next run if needed."
        )

    return TrainingConfig()


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
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, list[float]]]:
    train_split, val_split, test_split, normalisation = create_datasets(
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
    return train_loader, val_loader, test_loader, normalisation


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


def compute_epoch_lr(
    *,
    base_lr: float,
    min_lr: float,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    """Cosine decay with linear warm-up."""

    epoch_index = epoch - 1
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        scale = (epoch_index + 1) / warmup_epochs
        return max(min_lr, base_lr * scale)

    if total_epochs <= warmup_epochs:
        return max(min_lr, base_lr)

    progress = (epoch_index - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine = 0.5 * (1 + np.cos(np.pi * np.clip(progress, 0.0, 1.0)))
    lr = min_lr + (base_lr - min_lr) * cosine
    return max(min_lr, lr)


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
    max_grad_norm: float | None,
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
        if max_grad_norm is not None and max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), max_grad_norm)
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

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

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

    ax[2].plot(epochs, history["lr"], color="#9467bd", marker="o")
    ax[2].set_title("Learning rate schedule")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Learning rate")
    ax[2].grid(True, linestyle="--", alpha=0.5)

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
    config = load_config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    missing_paths = [
        (name, path)
        for name, path in (("train_data", config.train_data), ("test_data", config.test_data))
        if not path.exists()
    ]
    if missing_paths:
        formatted = ", ".join(f"{name}='{path}'" for name, path in missing_paths)
        raise SystemExit(
            f"Could not locate the following paths: {formatted}. "
            "Edit training_config.json to point to the correct files."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, normalisation = create_dataloaders(
        config.train_data,
        test_mat_path=config.test_data,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        feature_key=config.feature_key,
        label_key=config.label_key,
        test_feature_key=config.test_feature_key,
        test_label_key=config.test_label_key,
        seed=config.seed,
    )

    print(
        "Normalisation (per-channel mean/std):",
        {k: [round(vv, 6) for vv in values] for k, values in normalisation.items()},
    )

    split_summaries = [
        _summarise_split("Train", train_loader.dataset),
        _summarise_split("Valid", val_loader.dataset),
        _summarise_split("Test", test_loader.dataset),
    ]

    num_classes = len(np.unique(train_loader.dataset.labels.cpu().numpy()))
    model = CNN_Transformer(num_cls=num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
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

    best_snapshot = {
        "epoch": 0,
        "val_loss": float("inf"),
        "val_acc": 0.0,
    }

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        current_lr = compute_epoch_lr(
            base_lr=config.lr,
            min_lr=config.min_lr,
            epoch=epoch,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=config.epochs,
            log_interval=config.log_interval,
            max_grad_norm=config.max_grad_norm,
        )
        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, device
        )
        early_stopping(val_loss, model)
        if epoch <= config.min_epochs:
            early_stopping.early_stop = False
            early_stopping.counter = 0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_loss < best_snapshot["val_loss"]:
            best_snapshot.update({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% | "
            f"best_val_acc={best_snapshot['val_acc'] * 100:.2f}% (epoch {best_snapshot['epoch']:03d}) | "
            f"elapsed={elapsed:.1f}s"
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
        "normalisation": normalisation,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.__dict__.items()
        },
    }
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    with open(config.output_dir / "classification_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    plot_per_class_accuracy(report, config.output_dir)

    class_names = [f"Class {idx}" for idx in sorted(np.unique(y_true))]
    fig = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        normalize=True,
        title="Normalised confusion matrix",
        save_path=config.output_dir / "confusion_matrix.png",
    )
    plt.close(fig)

    # Also store the raw confusion matrix for completeness.
    fig = plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        normalize=False,
        title="Confusion matrix (counts)",
        save_path=config.output_dir / "confusion_matrix_counts.png",
    )
    plt.close(fig)

    plot_tsne_embeddings(features, y_true, config.output_dir, seed=config.seed)
    plot_distance_distributions(features, y_true, config.output_dir, seed=config.seed)
    plot_confidence_histogram(probabilities, y_pred, y_true, config.output_dir)

    print("Training artefacts written to", config.output_dir.resolve())


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
