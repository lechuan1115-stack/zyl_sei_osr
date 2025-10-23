#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Open-set recognition utilities for the ADS-B experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from scipy.stats import weibull_min
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

__all__ = [
    "OpenSetSplit",
    "determine_open_set_split",
    "filter_by_classes",
    "OpenMaxCalibrator",
    "extract_features",
    "compute_open_set_metrics",
    "evaluate_open_set_methods",
]


@dataclass(frozen=True)
class OpenSetSplit:
    """Holds the partition of known and unknown class identifiers."""

    known_classes: List[int]
    unknown_classes: List[int]

    def as_dict(self) -> Dict[str, List[int]]:
        return {"known": self.known_classes, "unknown": self.unknown_classes}


def determine_open_set_split(labels: np.ndarray, known_ratio: float = 0.7) -> OpenSetSplit:
    """Split the label space into known and unknown subsets.

    Parameters
    ----------
    labels:
        Array of integer encoded labels present in the dataset.
    known_ratio:
        Fraction of unique classes that will be treated as known during
        training.  The remainder is reserved as ``unknown`` classes.
    """

    if labels.ndim != 1:
        raise ValueError("`labels` must be a one-dimensional array of ints.")

    unique = sorted(np.unique(labels))
    if not unique:
        raise ValueError("No labels supplied to determine the open-set split.")

    known_count = max(1, int(math.ceil(len(unique) * known_ratio)))
    known_count = min(known_count, len(unique) - 1) if len(unique) > 1 else len(unique)
    if known_count <= 0:
        known_count = max(1, len(unique) - 1)

    known_classes = unique[:known_count]
    unknown_classes = [cls for cls in unique if cls not in known_classes]

    if not unknown_classes:
        # Fallback to leave the largest label as unknown when the ratio assigns all.
        unknown_classes = [known_classes.pop()]

    return OpenSetSplit(known_classes=known_classes, unknown_classes=unknown_classes)


def filter_by_classes(
    data: np.ndarray,
    labels: np.ndarray,
    classes: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a subset of ``data`` containing only the requested ``classes``."""

    mask = np.isin(labels, list(classes))
    return data[mask], labels[mask]


class OpenMaxCalibrator:
    """Implementation of the OpenMax recalibration procedure.

    The calibrator fits a Weibull distribution to the tail of the distances
    between class-specific mean activation vectors (MAVs) and the training
    features.  During inference the logits are adjusted with the estimated tail
    probabilities, producing an additional ``unknown`` score.
    """

    def __init__(self, tail_size: int = 20, alpha: int = 3) -> None:
        self.tail_size = tail_size
        self.alpha = alpha
        self.mavs: Dict[int, np.ndarray] = {}
        self.weibull_models: Dict[int, Tuple[float, float, float]] = {}

    def fit(self, features: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != logits.shape[0] or logits.shape[0] != labels.shape[0]:
            raise ValueError("Features, logits and labels must have matching lengths.")

        self.mavs.clear()
        self.weibull_models.clear()

        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_features = features[cls_mask]
            if cls_features.size == 0:
                continue
            mav = cls_features.mean(axis=0)
            self.mavs[int(cls)] = mav

            distances = np.linalg.norm(cls_features - mav, axis=1)
            distances = np.sort(distances)[::-1]
            tail = distances[: self.tail_size]
            if len(tail) < 2:
                # Avoid fitting degenerate Weibull parameters.
                tail = np.pad(tail, (0, max(0, 2 - len(tail))), constant_values=tail[-1] if len(tail) else 1.0)

            shape, loc, scale = weibull_min.fit(tail, floc=0)
            self.weibull_models[int(cls)] = (shape, loc, scale)

    def recalibrate(
        self, logits: np.ndarray, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.mavs:
            raise RuntimeError("OpenMaxCalibrator must be fitted before calling `recalibrate`.")

        adjusted_logits = logits.copy()
        unknown_scores = np.zeros(logits.shape[0], dtype=np.float64)

        for idx in range(logits.shape[0]):
            sample_logits = logits[idx]
            sample_features = features[idx]
            ranked_classes = np.argsort(sample_logits)[::-1]
            alpha_weights = self._alpha_weights(len(ranked_classes))

            for rank, cls in enumerate(ranked_classes[: self.alpha]):
                if cls not in self.mavs:
                    continue
                mav = self.mavs[cls]
                shape, loc, scale = self.weibull_models[cls]
                distance = np.linalg.norm(sample_features - mav)
                w_score = weibull_min.cdf(distance, shape, loc=loc, scale=scale)
                # Redistribute mass from known logits to the unknown bucket.
                weight = alpha_weights[rank] * w_score
                adjusted_logits[idx, cls] = sample_logits[cls] * (1 - weight)
                unknown_scores[idx] += sample_logits[cls] * weight

        return adjusted_logits, unknown_scores

    def _alpha_weights(self, num_classes: int) -> np.ndarray:
        alpha_range = min(self.alpha, num_classes)
        weights = np.ones(alpha_range, dtype=np.float64)
        weights = weights / np.sum(weights)
        return weights


def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return logits, features and labels for ``dataloader``."""

    model.eval()
    all_logits: List[np.ndarray] = []
    all_features: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, features = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_features, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def _energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return temperature * np.log(np.exp(logits / temperature).sum(axis=1))


def compute_open_set_metrics(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_predictions: np.ndarray,
    known_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute standard open-set metrics given detection scores."""

    y_true = np.concatenate([np.zeros_like(known_scores), np.ones_like(unknown_scores)])
    y_scores = np.concatenate([known_scores, unknown_scores])

    metrics: Dict[str, float] = {}
    metrics["auroc"] = roc_auc_score(y_true, y_scores)
    metrics["aupr_in"] = average_precision_score(1 - y_true, -y_scores)
    metrics["aupr_out"] = average_precision_score(y_true, y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    metrics["pr_auc"] = auc(recall, precision)

    metrics["closed_accuracy"] = accuracy_score(known_labels, known_predictions)
    metrics["closed_f1_macro"] = f1_score(known_labels, known_predictions, average="macro")

    return metrics


def _oscr(known_logits: np.ndarray, unknown_logits: np.ndarray, known_labels: np.ndarray) -> float:
    """Compute the Open Set Classification Rate (OSCR)."""

    known_scores = known_logits.max(axis=1)
    known_correct = (known_logits.argmax(axis=1) == known_labels).astype(np.float32)
    unknown_scores = unknown_logits.max(axis=1)

    scores = np.concatenate([known_scores, unknown_scores])
    correctness = np.concatenate([known_correct, np.zeros_like(unknown_scores)])
    labels = np.concatenate([np.ones_like(known_scores), np.zeros_like(unknown_scores)])

    order = np.argsort(scores)[::-1]
    correct = correctness[order]
    known_mask = labels[order]

    cum_correct = np.cumsum(correct)
    cum_known = np.cumsum(known_mask)
    cum_unknown = np.cumsum(1 - known_mask)

    tpr = np.divide(cum_correct, np.maximum(cum_known, 1), out=np.zeros_like(cum_correct), where=cum_known > 0)
    fpr = np.divide(cum_unknown, np.arange(1, len(order) + 1), out=np.zeros_like(cum_unknown), where=np.arange(1, len(order) + 1) > 0)

    return auc(fpr, tpr)


def evaluate_open_set_methods(
    known_logits: np.ndarray,
    known_features: np.ndarray,
    known_labels: np.ndarray,
    unknown_logits: np.ndarray,
    unknown_features: np.ndarray,
    calibrator: OpenMaxCalibrator,
    class_means: np.ndarray,
    precision: np.ndarray,
    log_det_covariance: float,
) -> Dict[str, Dict[str, float]]:
    """Run multiple open-set scoring functions and return their metrics."""

    results: Dict[str, Dict[str, float]] = {}

    softmax_probs = _softmax(known_logits)
    unknown_softmax_probs = _softmax(unknown_logits)
    known_scores = 1.0 - softmax_probs.max(axis=1)
    unknown_scores = 1.0 - unknown_softmax_probs.max(axis=1)
    results["maximum_softmax_probability"] = compute_open_set_metrics(
        known_scores,
        unknown_scores,
        known_logits.argmax(axis=1),
        known_labels,
    )
    results["maximum_softmax_probability"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    entropy_known = _entropy(softmax_probs)
    entropy_unknown = _entropy(unknown_softmax_probs)
    results["entropy"] = compute_open_set_metrics(
        entropy_known,
        entropy_unknown,
        known_logits.argmax(axis=1),
        known_labels,
    )
    results["entropy"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    energy_known = _energy(known_logits)
    energy_unknown = _energy(unknown_logits)
    results["energy"] = compute_open_set_metrics(
        energy_known,
        energy_unknown,
        known_logits.argmax(axis=1),
        known_labels,
    )
    results["energy"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # Mahalanobis distance based detector.
    diff_known = known_features - class_means[known_logits.argmax(axis=1)]
    diff_unknown = unknown_features - class_means.mean(axis=0)
    maha_known = np.einsum("bi,ij,bj->b", diff_known, precision, diff_known)
    maha_unknown = np.einsum("bi,ij,bj->b", diff_unknown, precision, diff_unknown)
    results["mahalanobis"] = compute_open_set_metrics(
        maha_known,
        maha_unknown,
        known_logits.argmax(axis=1),
        known_labels,
    )
    results["mahalanobis"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # Feature magnitude thresholding.
    feature_norm_known = np.linalg.norm(known_features, axis=1)
    feature_norm_unknown = np.linalg.norm(unknown_features, axis=1)
    results["feature_magnitude"] = compute_open_set_metrics(
        feature_norm_known,
        feature_norm_unknown,
        known_logits.argmax(axis=1),
        known_labels,
    )
    results["feature_magnitude"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # Logit margin (difference between top-1 and top-2 logits).
    def _margin_scores(logit_array: np.ndarray) -> np.ndarray:
        if logit_array.shape[1] < 2:
            return np.zeros(logit_array.shape[0], dtype=np.float64)
        top2 = np.partition(logit_array, -2, axis=1)[:, -2:]
        top1 = top2.max(axis=1)
        top2_second = top2.min(axis=1)
        margin = top1 - top2_second
        return 1.0 / (margin + 1e-6)

    margin_known = _margin_scores(known_logits)
    margin_unknown = _margin_scores(unknown_logits)
    results["logit_margin"] = compute_open_set_metrics(
        margin_known,
        margin_unknown,
        known_logits.argmax(axis=1),
        known_labels,
    )
    results["logit_margin"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # Cosine similarity between features and predicted class mean.
    pred_known = known_logits.argmax(axis=1)
    pred_unknown = unknown_logits.argmax(axis=1)
    known_means = class_means[pred_known]
    unknown_means = class_means[pred_unknown]

    def _cosine_distance(features: np.ndarray, means: np.ndarray) -> np.ndarray:
        numerator = np.einsum("bi,bi->b", features, means)
        feat_norm = np.linalg.norm(features, axis=1)
        mean_norm = np.linalg.norm(means, axis=1)
        cosine = numerator / (feat_norm * mean_norm + 1e-6)
        return 1.0 - np.clip(cosine, -1.0, 1.0)

    cosine_known = _cosine_distance(known_features, known_means)
    cosine_unknown = _cosine_distance(unknown_features, unknown_means)
    results["cosine_distance"] = compute_open_set_metrics(
        cosine_known,
        cosine_unknown,
        pred_known,
        known_labels,
    )
    results["cosine_distance"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # Euclidean distance to predicted class mean.
    euclid_known = np.linalg.norm(known_features - known_means, axis=1)
    euclid_unknown = np.linalg.norm(unknown_features - unknown_means, axis=1)
    results["euclidean_distance"] = compute_open_set_metrics(
        euclid_known,
        euclid_unknown,
        pred_known,
        known_labels,
    )
    results["euclidean_distance"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # Gaussian log-likelihood under the shared covariance model.
    feature_dim = known_features.shape[1]
    gaussian_known = 0.5 * (maha_known + log_det_covariance + feature_dim * math.log(2 * math.pi))
    gaussian_unknown = 0.5 * (maha_unknown + log_det_covariance + feature_dim * math.log(2 * math.pi))
    results["gaussian_likelihood"] = compute_open_set_metrics(
        gaussian_known,
        gaussian_unknown,
        pred_known,
        known_labels,
    )
    results["gaussian_likelihood"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # KL divergence to a uniform categorical distribution.
    num_classes = known_logits.shape[1]
    uniform_log = -math.log(num_classes)
    kl_known = np.sum(
        softmax_probs * (np.log(softmax_probs + 1e-12) - uniform_log),
        axis=1,
    )
    kl_unknown = np.sum(
        unknown_softmax_probs * (np.log(unknown_softmax_probs + 1e-12) - uniform_log),
        axis=1,
    )
    inv_kl_known = 1.0 / (kl_known + 1e-6)
    inv_kl_unknown = 1.0 / (kl_unknown + 1e-6)
    results["kl_uniform"] = compute_open_set_metrics(
        inv_kl_known,
        inv_kl_unknown,
        pred_known,
        known_labels,
    )
    results["kl_uniform"]["oscr"] = _oscr(known_logits, unknown_logits, known_labels)

    # OpenMax recalibration.
    recal_known_logits, known_unknown_scores = calibrator.recalibrate(known_logits, known_features)
    recal_unknown_logits, unknown_unknown_scores = calibrator.recalibrate(
        unknown_logits,
        unknown_features,
    )
    results["openmax"] = compute_open_set_metrics(
        known_unknown_scores,
        unknown_unknown_scores,
        recal_known_logits.argmax(axis=1),
        known_labels,
    )
    results["openmax"]["oscr"] = _oscr(recal_known_logits, recal_unknown_logits, known_labels)

    return results
