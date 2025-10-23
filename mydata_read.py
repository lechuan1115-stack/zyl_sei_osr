#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Data loading utilities for the ADS-B modulation recognition project.

The previous version of this module contained numerous legacy helpers that
were specific to old experiments.  The implementation below focuses on the
10-class ADS-B dataset stored in MATLAB v7.3 (HDF5) ``.mat`` files.  Each
sample is an I/Q complex time-series with the layout ``(4800, 2)``.  The
helpers centralise the logic required to read the dataset, normalise the
shape to PyTorch's channel-first convention and split the data into
train/validation/test partitions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

__all__ = [
    "ADSBSignalDataset",
    "DatasetSplit",
    "create_datasets",
    "load_adsb_dataset",
]


@dataclass(frozen=True)
class DatasetSplit:
    """Container returning NumPy arrays for a single dataset split."""

    data: np.ndarray
    labels: np.ndarray
    metadata: dict | None = None


class ADSBSignalDataset(Dataset):
    """PyTorch ``Dataset`` wrapping ADS-B I/Q samples.

    Parameters
    ----------
    data:
        NumPy array with shape ``(N, 2, 4800)`` containing the I and Q channels
        in the second dimension.
    labels:
        Integer encoded labels with shape ``(N,)``.  The helper function
        :func:`load_adsb_dataset` already converts MATLAB's one-based labels to
        zero-based indices.
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        if data.ndim != 3 or data.shape[1] != 2:
            raise ValueError(
                "`data` must have shape (num_samples, 2, signal_length). "
                f"Got {data.shape}."
            )
        if labels.ndim != 1:
            raise ValueError("`labels` must be a one-dimensional array.")

        self.data = torch.from_numpy(data.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return int(self.data.shape[0])


def load_adsb_dataset(
    mat_path: os.PathLike[str] | str,
    *,
    feature_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the ADS-B dataset stored inside a MATLAB v7.3 ``.mat`` file.

    The function tries to infer the dataset keys automatically.  When a custom
    layout is used, set ``feature_key`` and ``label_key`` explicitly.  The
    returned data is shaped as ``(num_samples, 2, 4800)`` using the channel-first
    convention expected by :class:`torch.nn.Conv1d`.
    """

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    with h5py.File(mat_path, "r") as mat_file:
        keys = list(mat_file.keys())
        if len(keys) < 2 and not (feature_key and label_key):
            raise RuntimeError(
                "The .mat file must contain at least two datasets (features and "
                "labels)."
            )

        # ``feature_key``/``label_key`` override the automatic detection.  When
        # not provided we heuristically pick the array with the highest number of
        # dimensions as the feature tensor and the remaining one as labels.
        if feature_key is None or label_key is None:
            arrays = {k: np.array(mat_file[k]) for k in keys}
            if feature_key is None:
                feature_key = max(arrays, key=lambda k: arrays[k].ndim)
            if label_key is None:
                remaining = [k for k in arrays if k != feature_key]
                if not remaining:
                    raise RuntimeError("Unable to infer label dataset from file.")
                # Prefer the array with the smallest dimensionality for labels.
                label_key = min(remaining, key=lambda k: arrays[k].ndim)
            features = arrays[feature_key]
            labels = arrays[label_key]
        else:
            features = np.array(mat_file[feature_key])
            labels = np.array(mat_file[label_key])

    data = _ensure_channel_first(features)
    labels = _prepare_labels(labels)

    if data.shape[0] != labels.shape[0]:
        raise RuntimeError(
            "Mismatched number of samples: "
            f"features={data.shape[0]} vs labels={labels.shape[0]}"
        )

    _log_dataset_statistics(mat_path, data, labels, feature_key, label_key)
    return data, labels


def create_datasets(
    train_mat_path: os.PathLike[str] | str,
    *,
    test_mat_path: os.PathLike[str] | str,
    val_ratio: float = 0.1,
    random_state: int = 42,
    feature_key: Optional[str] = None,
    label_key: Optional[str] = None,
    test_feature_key: Optional[str] = None,
    test_label_key: Optional[str] = None,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, dict[str, list[float]]]:
    """Create train/validation/test splits from dedicated dataset files.

    Parameters
    ----------
    train_mat_path:
        Path to the MATLAB ``.mat`` file containing the combined training
        samples.  The function will internally split this data into training and
        validation subsets using ``val_ratio``.
    test_mat_path:
        Path to the MATLAB ``.mat`` file containing the held-out test set.
    val_ratio:
        Fraction of the training samples used for validation.  The remainder is
        used for training.
    random_state:
        Seed controlling the deterministic behaviour of ``train_test_split``.
    feature_key / label_key:
        Optional overrides selecting the arrays that contain the raw signals and
        the labels inside the MATLAB file.  When unset the function will attempt
        to infer them automatically.  ``test_feature_key`` and ``test_label_key``
        provide the same overrides for the test dataset (falling back to the
        training keys when omitted).
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("`val_ratio` must be between 0 and 1 (exclusive).")

    data, labels = load_adsb_dataset(
        train_mat_path,
        feature_key=feature_key,
        label_key=label_key,
    )

    channel_stats = _compute_channelwise_stats(data)
    data = _normalise_signals(data, channel_stats)

    stratify = labels if len(np.unique(labels)) > 1 else None
    data_train, data_val, label_train, label_val = train_test_split(
        data,
        labels,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify,
    )

    test_data, test_labels = load_adsb_dataset(
        test_mat_path,
        feature_key=test_feature_key or feature_key,
        label_key=test_label_key or label_key,
    )

    test_data = _normalise_signals(test_data, channel_stats)

    return (
        DatasetSplit(data_train, label_train, metadata={"split": "train"}),
        DatasetSplit(data_val, label_val, metadata={"split": "valid"}),
        DatasetSplit(test_data, test_labels, metadata={"split": "test"}),
        {
            "mean": channel_stats["mean"].flatten().tolist(),
            "std": channel_stats["std"].flatten().tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_channel_first(array: np.ndarray) -> np.ndarray:
    """Rearrange the ``.mat`` feature tensor to ``(N, 2, 4800)`` format."""

    array = np.asarray(array)
    if array.ndim != 3:
        raise RuntimeError(
            "Expected a 3-D tensor containing (samples, channels, length). "
            f"Got {array.shape}."
        )

    axes = list(range(3))
    try:
        channel_axis = next(ax for ax, size in enumerate(array.shape) if size == 2)
        length_axis = next(ax for ax, size in enumerate(array.shape) if size == 4800)
    except StopIteration as exc:  # pragma: no cover - defensive programming
        raise RuntimeError(
            "Unable to locate channel/length dimensions in feature array."
        ) from exc

    sample_axis = next(ax for ax in axes if ax not in (channel_axis, length_axis))
    ordered = np.transpose(array, (sample_axis, channel_axis, length_axis))
    return ordered.astype(np.float32, copy=False)


def _prepare_labels(array: np.ndarray) -> np.ndarray:
    """Convert labels to a flat array of zero-based ``int64`` indices."""

    labels = np.asarray(array)
    labels = labels.squeeze()
    if labels.ndim != 1:
        raise RuntimeError("Labels must reduce to a one-dimensional array.")

    labels = labels.astype(np.int64)
    min_label = labels.min()
    if min_label == 1:
        # MATLAB-style one-based labels.  Shift to zero-based for PyTorch.
        labels -= 1
    return labels


def _log_dataset_statistics(
    mat_path: Path,
    data: np.ndarray,
    labels: np.ndarray,
    feature_key: Optional[str],
    label_key: Optional[str],
) -> None:
    """Print a short JSON-formatted summary for transparency."""

    info = {
        "file": str(mat_path.resolve()),
        "feature_key": feature_key,
        "label_key": label_key,
        "num_samples": int(data.shape[0]),
        "signal_shape": list(map(int, data.shape[1:])),
        "num_classes": int(len(np.unique(labels))),
    }
    print(json.dumps(info, ensure_ascii=False))


def _compute_channelwise_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-channel mean and standard deviation for normalisation."""

    mean = data.mean(axis=(0, 2), keepdims=True)
    std = data.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32, copy=False), "std": std.astype(np.float32, copy=False)}


def _normalise_signals(data: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Apply channel-wise normalisation using the provided statistics."""

    normalised = (data - stats["mean"]) / stats["std"]
    return normalised.astype(np.float32, copy=False)
