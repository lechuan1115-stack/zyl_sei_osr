#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Open-set evaluation utilities for the ADS-B classifier.

This module keeps the closed-set training script untouched while providing a
stand-alone workflow that can be executed after the classifier has been trained
on all 10 classes.  During open-set evaluation the model is only expected to
recognise the first seven classes, whereas the remaining three classes act as
unknown examples.  The script benchmarks a collection of open-set detectors
including OpenMax and produces quantitative metrics together with visual
diagnostics saved under ``open_set_outputs/``.

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
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from mydata_read import ADSBSignalDataset, load_adsb_dataset
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
    total_classes: int = 10
    tail_size: int = 25
    alpha: int = 5
    seed: int = 42


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

            self.class_means[cls] = mean_vector
            self.shapes[cls] = shape
            self.scales[cls] = scale

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
            omega = survival * weight
            adjusted = logits[cls] * (1 - omega)
            unknown_activation += logits[cls] - adjusted
            recalibrated[cls] = adjusted

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


def _compute_mahalanobis_stats(
    features: np.ndarray,
    labels: np.ndarray,
    known_classes: Sequence[int],
    epsilon: float = 1e-6,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
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
    return means, precision


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
    train_data, train_labels = load_adsb_dataset(config.train_data)
    test_data, test_labels = load_adsb_dataset(config.test_data)

    known_mask_train = np.isin(train_labels, config.known_classes)
    known_train_data = train_data[known_mask_train]
    known_train_labels = train_labels[known_mask_train]

    known_mask_test = np.isin(test_labels, config.known_classes)
    known_test_data = test_data[known_mask_test]
    known_test_labels = test_labels[known_mask_test]

    unknown_mask_test = ~known_mask_test
    unknown_test_data = test_data[unknown_mask_test]
    unknown_test_labels = test_labels[unknown_mask_test]

    print(
        f"Known training samples: {known_train_data.shape[0]} | "
        f"Known test samples: {known_test_data.shape[0]} | "
        f"Unknown test samples: {unknown_test_data.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Load the trained classifier
    # ------------------------------------------------------------------
    model = create_model("CNN_Transformer", num_classes=config.total_classes).to(device)
    state_dict = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    train_loader = _build_dataloader(known_train_data, known_train_labels, config.batch_size)
    known_test_loader = _build_dataloader(known_test_data, known_test_labels, config.batch_size)
    unknown_test_loader = _build_dataloader(unknown_test_data, unknown_test_labels, config.batch_size)

    train_logits, train_features, train_labels_known = _extract_embeddings(model, train_loader, device)
    known_logits, known_features, known_labels = _extract_embeddings(model, known_test_loader, device)
    unknown_logits, unknown_features, _ = _extract_embeddings(model, unknown_test_loader, device)

    # ------------------------------------------------------------------
    # Prepare detectors
    # ------------------------------------------------------------------
    weibull = WeibullCalibrator(tail_size=config.tail_size, alpha=config.alpha)
    weibull.fit(train_features, train_labels_known, config.known_classes)

    class_means, precision = _compute_mahalanobis_stats(
        train_features, train_labels_known, config.known_classes
    )

    known_softmax = _softmax(known_logits)
    unknown_softmax = _softmax(unknown_logits)

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
    entropy_known = -np.sum(known_softmax * np.log(known_softmax + eps), axis=1) / math.log(
        config.total_classes
    )
    entropy_unknown = -np.sum(unknown_softmax * np.log(unknown_softmax + eps), axis=1) / math.log(
        config.total_classes
    )
    _compute_unknown_scores(entropy_known, entropy_unknown, "Entropy")

    # Energy
    energy_known = -_logsumexp(known_logits).ravel()
    energy_unknown = -_logsumexp(unknown_logits).ravel()
    _compute_unknown_scores(energy_known, energy_unknown, "Energy")

    # Logit margin
    def _margin_scores(logits: np.ndarray) -> np.ndarray:
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

