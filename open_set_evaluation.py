#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Open-set evaluation utilities for the ADS-B classifier.

This module leaves the closed-set training script untouched while providing a
stand-alone workflow that can be executed after the classifier has been trained
on the first seven classes only.  During open-set evaluation the trained model
is exposed to the full ten-class dataset: classes 0--6 are treated as known
while classes 7--9 serve as unknown examples.  The script benchmarks a suite of
open-set detectors including OpenMax, MSP, entropy, energy, logit-margin,
Mahalanobis, Gaussian NLL, prototype distances, KL-to-uniform, and inverse
feature-norm baselines while producing quantitative metrics together with
visual diagnostics saved under ``open_set_outputs/``.

The entry point is :func:`run_open_set_evaluation`, which can be executed
directly via ``python open_set_evaluation.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from mydata_read import ADSBSignalDataset, create_datasets
from mymodel1 import create as create_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class OpenSetConfig:
    """Configuration container for the open-set evaluation pipeline."""

    train_data: Path = Path(r"E:\数据集\ADS-B_Train_10X360-2_5-10-15-20dB.mat")
    test_data: Path = Path(r"E:\数据集\ADS-B_Test_10X360-2_5-10-15-20dB.mat")
    checkpoint_path: Path = Path("training_outputs/best_model.pt")
    output_dir: Path = Path("open_set_outputs")
    batch_size: int = 256
    known_classes: Sequence[int] = tuple(range(7))
    tail_size: int = 25
    alpha: int = 5
    seed: int = 42
    val_ratio: float = 0.1
    feature_key: str | None = None
    label_key: str | None = None
    test_feature_key: str | None = None
    test_label_key: str | None = None
    temperature_lr: float = 5e-3
    temperature_steps: int = 500


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _build_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> DataLoader:
    dataset = ADSBSignalDataset(data, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs, features = model(inputs)
            logits_list.append(outputs.cpu().numpy())
            features_list.append(features.cpu().numpy())
            labels_list.append(targets.cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return logits, features, labels


def _calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    *,
    lr: float,
    steps: int,
) -> float:
    """Estimate a temperature scaling factor using known-class validation logits."""

    if logits.size == 0:
        return 1.0

    logits_tensor = torch.from_numpy(logits).to(device)
    labels_tensor = torch.from_numpy(labels.astype(np.int64, copy=False)).to(device)

    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=1e3).item()

    return float(temperature)


# ---------------------------------------------------------------------------
# OpenMax implementation
# ---------------------------------------------------------------------------


class WeibullCalibrator:
    """Fits Weibull models on activation distances for OpenMax."""

    def __init__(self, tail_size: int, alpha: int) -> None:
        self.tail_size = tail_size
        self.alpha = alpha
        self.class_means: Dict[int, np.ndarray] = {}
        self.shapes: Dict[int, float] = {}
        self.scales: Dict[int, float] = {}
        self.tail_means: Dict[int, float] = {}
        self.tail_stds: Dict[int, float] = {}

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        known_classes: Iterable[int],
    ) -> None:
        for cls in known_classes:
            cls_features = features[labels == cls]
            if cls_features.size == 0:
                raise RuntimeError(f"No features available to fit Weibull for class {cls}.")

            mean_vector = cls_features.mean(axis=0)
            distances = np.linalg.norm(cls_features - mean_vector, axis=1)
            tail = np.sort(distances)[-min(self.tail_size, len(distances)) :]
            tail = tail[tail > 0]
            if tail.size < 2:
                # Degenerate case: fall back to a small scale to avoid division by zero.
                shape, scale = 1.0, max(float(np.max(distances)), 1e-6)
            else:
                shape, scale = _fit_weibull_tail(tail)

            tail_mean = float(tail.mean()) if tail.size else float(distances.mean())
            tail_std = float(tail.std(ddof=0)) if tail.size > 1 else float(
                distances.std(ddof=0) if distances.size else 1.0
            )

            self.class_means[cls] = mean_vector
            self.shapes[cls] = shape
            self.scales[cls] = scale
            self.tail_means[cls] = tail_mean
            self.tail_stds[cls] = max(tail_std, 1e-6)

    def recalibrate(
        self,
        logits: np.ndarray,
        feature: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        ranked = np.argsort(logits)[::-1]
        alpha = min(self.alpha, len(ranked))
        recalibrated = logits.copy()
        unknown_activation = 0.0

        for rank, cls in enumerate(ranked[:alpha], start=1):
            if cls not in self.class_means:
                continue
            mean = self.class_means[cls]
            shape = self.shapes[cls]
            scale = self.scales[cls]
            distance = float(np.linalg.norm(feature - mean))
            survival = math.exp(-((distance / (scale + 1e-12)) ** shape))
            weight = (alpha - rank + 1) / alpha

            tail_mean = self.tail_means.get(cls, 0.0)
            tail_std = self.tail_stds.get(cls, 1.0)
            z_score = (distance - tail_mean) / tail_std
            z_score -= 0.5  # encourage smaller weights near the tail boundary
            logistic_weight = 1.0 / (1.0 + math.exp(-z_score))

            cdf = 1.0 - survival
            blended = 0.5 * cdf + 0.5 * logistic_weight
            omega = max(0.0, min(0.95, weight * blended))
            adjusted = logits[cls] * (1 - omega)
            unknown_activation += logits[cls] - adjusted
            recalibrated[cls] = adjusted

        unknown_activation /= (alpha + 1e-6)

        augmented = np.concatenate([recalibrated, np.array([unknown_activation], dtype=np.float32)])
        max_logit = float(np.max(augmented))
        exp_scores = np.exp(augmented - max_logit)
        probabilities = exp_scores / exp_scores.sum()
        unknown_probability = float(probabilities[-1])
        return probabilities[:-1], unknown_probability


def _fit_weibull_tail(tail: np.ndarray) -> Tuple[float, float]:
    """Estimate Weibull shape/scale parameters via Newton iterations."""

    tail = tail.astype(np.float64)
    logs = np.log(tail)
    k = 1.0
    for _ in range(100):
        x_k = tail ** k
        sum_x_k = x_k.sum()
        sum_x_k_log = (x_k * logs).sum()
        sum_x_k_log2 = (x_k * logs * logs).sum()
        log_mean = logs.mean()

        f = (1.0 / k) + log_mean - (sum_x_k_log / sum_x_k)
        if abs(f) < 1e-6:
            break
        derivative = (
            -1.0 / (k * k)
            - (sum_x_k_log2 * sum_x_k - sum_x_k_log * sum_x_k_log) / (sum_x_k * sum_x_k)
        )
        if derivative == 0:
            break
        k_new = k - f / derivative
        if k_new <= 0 or not np.isfinite(k_new):
            break
        k = k_new

    scale = (tail ** k).mean() ** (1.0 / k)
    return float(k), float(scale)


# ---------------------------------------------------------------------------
# Detector scores
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    max_logits = logits.max(axis=1, keepdims=True)
    return max_logits + np.log(np.exp(logits - max_logits).sum(axis=1, keepdims=True))


def _kl_to_uniform(logits: np.ndarray) -> np.ndarray:
    probs = _softmax(logits)
    num_classes = probs.shape[1]
    uniform = 1.0 / num_classes
    kl = np.sum(probs * (np.log(probs + 1e-12) - math.log(uniform)), axis=1)
    return 1.0 / (kl + 1e-6)


def _feature_norm(features: np.ndarray) -> np.ndarray:
    return np.linalg.norm(features, axis=1)


def _compute_mahalanobis_stats(
    features: np.ndarray,
    labels: np.ndarray,
    known_classes: Sequence[int],
    epsilon: float = 1e-6,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, float]:
    means: Dict[int, np.ndarray] = {}
    centered: List[np.ndarray] = []

    for cls in known_classes:
        cls_features = features[labels == cls]
        if cls_features.size == 0:
            raise RuntimeError(f"No features available for class {cls}.")
        mean = cls_features.mean(axis=0)
        means[cls] = mean
        centered.append(cls_features - mean)

    stacked = np.vstack(centered)
    covariance = np.cov(stacked, rowvar=False, bias=True)
    covariance += epsilon * np.eye(covariance.shape[0], dtype=covariance.dtype)
    precision = np.linalg.inv(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        # Fall back to an explicit determinant to avoid invalid logarithms.
        logdet = float(np.log(np.abs(np.linalg.det(covariance)) + 1e-12))
    return means, precision, float(logdet)


def _mahalanobis_distance(
    feature: np.ndarray,
    means: Dict[int, np.ndarray],
    precision: np.ndarray,
) -> float:
    distances = []
    for mean in means.values():
        diff = feature - mean
        distances.append(float(diff @ precision @ diff))
    return min(distances)


def _gaussian_negative_log_likelihood(
    feature: np.ndarray,
    means: Dict[int, np.ndarray],
    precision: np.ndarray,
    log_det_cov: float,
) -> float:
    values = []
    for mean in means.values():
        diff = feature - mean
        mahal = float(diff @ precision @ diff)
        values.append(0.5 * (mahal + log_det_cov))
    return min(values)


def _cosine_distance(feature: np.ndarray, means: Dict[int, np.ndarray]) -> float:
    feature_norm = np.linalg.norm(feature)
    if feature_norm < 1e-12:
        return 1.0
    scores = []
    feature_unit = feature / feature_norm
    for mean in means.values():
        mean_norm = np.linalg.norm(mean)
        if mean_norm < 1e-12:
            continue
        cosine = float(np.dot(feature_unit, mean / mean_norm))
        scores.append(1.0 - cosine)
    return min(scores) if scores else 1.0


def _euclidean_distance(feature: np.ndarray, means: Dict[int, np.ndarray]) -> float:
    distances = [float(np.linalg.norm(feature - mean)) for mean in means.values()]
    return min(distances)


# ---------------------------------------------------------------------------
# Metrics and visualisations
# ---------------------------------------------------------------------------


def _compute_oscr(
    known_detection: np.ndarray,
    known_correct: np.ndarray,
    unknown_detection: np.ndarray,
) -> float:
    scores = np.concatenate([known_detection, unknown_detection])
    labels = np.concatenate([known_correct.astype(int), np.zeros_like(unknown_detection, dtype=int)])
    order = np.argsort(scores)[::-1]

    tot_known = len(known_detection)
    tot_unknown = len(unknown_detection)
    if tot_unknown == 0 or tot_known == 0:
        return float("nan")

    tp = 0
    fp = 0
    ccr = [0.0]
    fpr = [0.0]

    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        ccr.append(tp / tot_known)
        fpr.append(fp / tot_unknown)

    return float(np.trapz(ccr, fpr))


def _compute_metrics(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_logits: np.ndarray,
    known_labels: np.ndarray,
) -> Dict[str, float]:
    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    y_scores = np.concatenate([known_scores, unknown_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall, precision)

    mask = tpr >= 0.95
    fpr95 = float(np.min(fpr[mask])) if np.any(mask) else 1.0

    softmax = _softmax(known_logits)
    known_pred = softmax.argmax(axis=1)
    known_correct = (known_pred == known_labels).astype(int)

    known_detection = -known_scores
    unknown_detection = -unknown_scores
    oscr = _compute_oscr(known_detection, known_correct, unknown_detection)

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr95": fpr95,
        "oscr": oscr,
    }


def _plot_roc_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Open-set ROC curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.close()


def _plot_score_histograms(
    scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    cols = 2
    rows = math.ceil(len(scores) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, (known, unknown)) in zip(axes, scores.items()):
        ax.hist(known, bins=40, alpha=0.7, label="Known", color="#1f77b4")
        ax.hist(unknown, bins=40, alpha=0.7, label="Unknown", color="#ff7f0e")
        ax.set_title(name)
        ax.set_xlabel("Unknownness score")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[len(scores) :]:
        ax.axis("off")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "score_histograms.png", dpi=300)
    plt.close(fig)


def _plot_metric_bars(metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    names = list(metrics.keys())
    auroc_values = [metrics[name]["auroc"] for name in names]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, auroc_values, color="#4c72b0")
    plt.ylabel("AUROC")
    plt.ylim(0, 1)
    plt.title("Open-set detector comparison")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, auroc_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "auroc_comparison.png", dpi=300)
    plt.close()


def _plot_tsne(
    known_features: np.ndarray,
    known_labels: np.ndarray,
    unknown_features: np.ndarray,
    output_dir: Path,
    seed: int,
) -> None:
    max_samples = 2000
    rng = np.random.default_rng(seed)

    def _sample(features: np.ndarray, labels: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray | None]:
        if features.shape[0] > max_samples:
            idx = rng.choice(features.shape[0], size=max_samples, replace=False)
            features = features[idx]
            if labels is not None:
                labels = labels[idx]
        return features, labels

    known_features, known_labels = _sample(known_features, known_labels)
    unknown_features, _ = _sample(unknown_features, None)

    combined = np.vstack([known_features, unknown_features])
    tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=seed)
    embedding = tsne.fit_transform(combined)

    known_points = embedding[: known_features.shape[0]]
    unknown_points = embedding[known_features.shape[0] :]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        known_points[:, 0],
        known_points[:, 1],
        c=known_labels,
        cmap="tab10",
        s=15,
        alpha=0.7,
        label="Known classes",
    )
    plt.scatter(
        unknown_points[:, 0],
        unknown_points[:, 1],
        c="#ff7f0e",
        s=15,
        alpha=0.7,
        label="Unknown classes",
    )
    handles_known, _ = scatter.legend_elements()
    known_labels_unique = np.unique(known_labels)
    legend1 = plt.legend(
        handles_known,
        [f"Class {cls}" for cls in known_labels_unique],
        title="Known classes",
        loc="upper right",
    )
    plt.gca().add_artist(legend1)
    plt.legend(
        handles=[Line2D([], [], marker="o", color="#ff7f0e", linestyle="", label="Unknown")],
        loc="lower left",
    )
    plt.title("t-SNE of test embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "tsne_embeddings.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------


def run_open_set_evaluation(config: OpenSetConfig = OpenSetConfig()) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trained checkpoint not found at {config.checkpoint_path}. Run train_adsb.py first."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets and split into known/unknown partitions
    # ------------------------------------------------------------------
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

    def _filter_known(split_data: np.ndarray, split_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(split_labels, config.known_classes)
        return split_data[mask], split_labels[mask]

    known_train_data, known_train_labels = _filter_known(train_split.data, train_split.labels)
    known_val_data, known_val_labels = _filter_known(val_split.data, val_split.labels)
    known_test_data, known_test_labels = _filter_known(test_split.data, test_split.labels)

    test_mask_known = np.isin(test_split.labels, config.known_classes)
    unknown_test_data = test_split.data[~test_mask_known]
    unknown_test_labels = test_split.labels[~test_mask_known]

    if known_train_data.size == 0 or known_test_data.size == 0:
        raise RuntimeError(
            "No samples from the configured known classes were found. "
            "Check that the training pipeline used the same label indices."
        )
    if unknown_test_data.size == 0:
        raise RuntimeError(
            "No unknown-class samples detected in the test dataset. "
            "Ensure classes beyond the known set remain present for open-set evaluation."
        )

    unknown_class_ids = sorted(set(int(cls) for cls in np.unique(unknown_test_labels)))

    print(
        f"Known train/val/test samples: {known_train_data.shape[0]} / {known_val_data.shape[0]} / {known_test_data.shape[0]} | "
        f"Unknown test samples: {unknown_test_data.shape[0]} | "
        f"Unknown class ids: {unknown_class_ids}"
    )

    num_known_classes = len(config.known_classes)

    # ------------------------------------------------------------------
    # Load the trained classifier
    # ------------------------------------------------------------------
    model = create_model("CNN_Transformer", num_classes=num_known_classes).to(device)
    state_dict = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    train_loader = _build_dataloader(known_train_data, known_train_labels, config.batch_size)
    val_loader = (
        _build_dataloader(known_val_data, known_val_labels, config.batch_size)
        if known_val_data.size
        else None
    )
    known_test_loader = _build_dataloader(known_test_data, known_test_labels, config.batch_size)
    unknown_test_loader = _build_dataloader(unknown_test_data, unknown_test_labels, config.batch_size)

    train_logits, train_features, train_labels_known = _extract_embeddings(model, train_loader, device)
    if val_loader is not None:
        val_logits, val_features, val_labels_known = _extract_embeddings(model, val_loader, device)
    else:
        val_logits = np.empty((0, num_known_classes), dtype=np.float32)
        val_features = np.empty((0, train_features.shape[1]), dtype=np.float32)
        val_labels_known = np.empty((0,), dtype=np.int64)
    known_logits, known_features, known_labels = _extract_embeddings(model, known_test_loader, device)
    unknown_logits, unknown_features, _ = _extract_embeddings(model, unknown_test_loader, device)

    # ------------------------------------------------------------------
    # Feature normalisation & temperature scaling
    # ------------------------------------------------------------------
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1e-6

    def _standardise(features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return features
        return (features - feature_mean) / feature_std

    train_features = _standardise(train_features)
    val_features = _standardise(val_features)
    known_features = _standardise(known_features)
    unknown_features = _standardise(unknown_features)

    temperature = _calibrate_temperature(
        val_logits,
        val_labels_known,
        device,
        lr=config.temperature_lr,
        steps=config.temperature_steps,
    ) if val_logits.size else 1.0

    temperature = float(np.clip(temperature, 1e-3, 1e3))
    print(f"Calibrated temperature: {temperature:.3f}")

    def _apply_temperature(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return logits
        return logits / temperature

    train_logits = _apply_temperature(train_logits)
    val_logits = _apply_temperature(val_logits)
    known_logits = _apply_temperature(known_logits)
    unknown_logits = _apply_temperature(unknown_logits)

    # ------------------------------------------------------------------
    # Prepare detectors
    # ------------------------------------------------------------------
    weibull = WeibullCalibrator(tail_size=config.tail_size, alpha=config.alpha)
    weibull.fit(train_features, train_labels_known, config.known_classes)

    class_means, precision, log_det_cov = _compute_mahalanobis_stats(
        train_features, train_labels_known, config.known_classes
    )

    known_softmax = _softmax(known_logits)
    unknown_softmax = _softmax(unknown_logits)
    known_feature_norms = _feature_norm(known_features)
    unknown_feature_norms = _feature_norm(unknown_features)

    # ------------------------------------------------------------------
    # Compute detector scores
    # ------------------------------------------------------------------
    roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    hist_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    def _compute_unknown_scores(
        known_scores: np.ndarray,
        unknown_scores: np.ndarray,
        name: str,
    ) -> None:
        roc_fpr, roc_tpr, _ = roc_curve(
            np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)]),
            np.concatenate([known_scores, unknown_scores]),
        )
        roc_curves[name] = (roc_fpr, roc_tpr)
        hist_scores[name] = (known_scores, unknown_scores)
        metrics[name] = _compute_metrics(known_scores, unknown_scores, known_logits, known_labels)

    # MSP
    msp_known = 1.0 - known_softmax.max(axis=1)
    msp_unknown = 1.0 - unknown_softmax.max(axis=1)
    _compute_unknown_scores(msp_known, msp_unknown, "MSP")

    # Entropy
    eps = 1e-12
    entropy_norm = max(math.log(num_known_classes), 1e-6)
    entropy_known = -np.sum(known_softmax * np.log(known_softmax + eps), axis=1) / entropy_norm
    entropy_unknown = -np.sum(unknown_softmax * np.log(unknown_softmax + eps), axis=1) / entropy_norm
    _compute_unknown_scores(entropy_known, entropy_unknown, "Entropy")

    # Energy (scaled by calibrated temperature)
    energy_known = -temperature * _logsumexp(known_logits).ravel()
    energy_unknown = -temperature * _logsumexp(unknown_logits).ravel()
    _compute_unknown_scores(energy_known, energy_unknown, "Energy")

    # Logit margin
    def _margin_scores(logits: np.ndarray) -> np.ndarray:
        if logits.shape[1] < 2:
            return np.ones(logits.shape[0], dtype=np.float32)
        sorted_logits = np.sort(logits, axis=1)
        top1 = sorted_logits[:, -1]
        top2 = sorted_logits[:, -2]
        margin = top1 - top2
        return 1.0 / (margin + 1e-6)

    margin_known = _margin_scores(known_logits)
    margin_unknown = _margin_scores(unknown_logits)
    _compute_unknown_scores(margin_known, margin_unknown, "Logit margin")

    # Mahalanobis
    mahalanobis_known = np.array(
        [_mahalanobis_distance(feat, class_means, precision) for feat in known_features],
        dtype=np.float32,
    )
    mahalanobis_unknown = np.array(
        [_mahalanobis_distance(feat, class_means, precision) for feat in unknown_features],
        dtype=np.float32,
    )
    _compute_unknown_scores(mahalanobis_known, mahalanobis_unknown, "Mahalanobis")

    # Gaussian NLL
    gaussian_known = np.array(
        [
            _gaussian_negative_log_likelihood(feat, class_means, precision, log_det_cov)
            for feat in known_features
        ],
        dtype=np.float32,
    )
    gaussian_unknown = np.array(
        [
            _gaussian_negative_log_likelihood(feat, class_means, precision, log_det_cov)
            for feat in unknown_features
        ],
        dtype=np.float32,
    )
    _compute_unknown_scores(gaussian_known, gaussian_unknown, "Gaussian NLL")

    # Euclidean distance to class prototypes
    euclidean_known = np.array(
        [_euclidean_distance(feat, class_means) for feat in known_features], dtype=np.float32
    )
    euclidean_unknown = np.array(
        [_euclidean_distance(feat, class_means) for feat in unknown_features], dtype=np.float32
    )
    _compute_unknown_scores(euclidean_known, euclidean_unknown, "Euclidean distance")

    # Cosine distance to class prototypes
    cosine_known = np.array(
        [_cosine_distance(feat, class_means) for feat in known_features], dtype=np.float32
    )
    cosine_unknown = np.array(
        [_cosine_distance(feat, class_means) for feat in unknown_features], dtype=np.float32
    )
    _compute_unknown_scores(cosine_known, cosine_unknown, "Cosine distance")

    # KL divergence to uniform distribution (lower divergence -> more unknown)
    kl_known = _kl_to_uniform(known_logits)
    kl_unknown = _kl_to_uniform(unknown_logits)
    _compute_unknown_scores(kl_known, kl_unknown, "KL to uniform")

    # Feature norm (smaller norms often correlate with unknowns)
    norm_known = 1.0 / (known_feature_norms + 1e-6)
    norm_unknown = 1.0 / (unknown_feature_norms + 1e-6)
    _compute_unknown_scores(norm_known, norm_unknown, "Inverse feature norm")

    # OpenMax
    openmax_known = []
    openmax_unknown = []
    for logit, feat in zip(known_logits, known_features):
        _, unknown_prob = weibull.recalibrate(logit, feat)
        openmax_known.append(unknown_prob)
    for logit, feat in zip(unknown_logits, unknown_features):
        _, unknown_prob = weibull.recalibrate(logit, feat)
        openmax_unknown.append(unknown_prob)
    openmax_known = np.array(openmax_known, dtype=np.float32)
    openmax_unknown = np.array(openmax_unknown, dtype=np.float32)
    _compute_unknown_scores(openmax_known, openmax_unknown, "OpenMax")

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    results_path = config.output_dir / "open_set_metrics.json"
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Saved metrics to {results_path}")

    summary_path = config.output_dir / "open_set_summary.json"
    summary_payload = {
        "temperature": temperature,
        "known_classes": list(map(int, config.known_classes)),
        "unknown_class_ids": unknown_class_ids,
        "counts": {
            "train_known": int(known_train_data.shape[0]),
            "val_known": int(known_val_data.shape[0]),
            "test_known": int(known_test_data.shape[0]),
            "test_unknown": int(unknown_test_data.shape[0]),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)
    print(f"Saved evaluation summary to {summary_path}")

    _plot_roc_curves(roc_curves, config.output_dir)
    _plot_score_histograms(hist_scores, config.output_dir)
    _plot_metric_bars(metrics, config.output_dir)
    _plot_tsne(known_features, known_labels, unknown_features, config.output_dir, config.seed)

    # Save raw scores for external analysis.
    scores_path = config.output_dir / "detection_scores.npz"
    score_arrays: Dict[str, np.ndarray] = {}
    for name, (known_scores, unknown_scores) in hist_scores.items():
        score_arrays[f"{name}_known"] = known_scores
        score_arrays[f"{name}_unknown"] = unknown_scores
    np.savez(scores_path, **score_arrays)
    print(f"Saved detection scores to {scores_path}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    run_open_set_evaluation()

