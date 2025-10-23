"""Utility function for drawing confusion matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

__all__ = ["plot_confusion_matrix"]


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    class_names: Optional[Iterable[str]] = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
    save_path: Optional[Path | str] = None,
) -> plt.Figure:
    """Render a confusion matrix with optional normalisation.

    Parameters
    ----------
    y_true, y_pred:
        Iterables containing the ground-truth and predicted labels.
    class_names:
        Optional sequence providing human readable class names.  When omitted
        the class indices are used directly.
    normalize:
        Normalise the confusion matrix row-wise when ``True``.  This highlights
        per-class accuracy and is often easier to interpret when the dataset is
        imbalanced.
    title:
        Title displayed at the top of the plot.
    save_path:
        When provided, the figure is stored as a PNG file at the given path.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure so callers can further customise or embed it in
        reports.
    """

    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    if class_names is None:
        labels = np.sort(np.unique(np.concatenate((y_true, y_pred))))
        display_labels = labels
    else:
        display_labels = list(class_names)
        labels = np.arange(len(display_labels))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(np.float32)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cmap = plt.cm.Blues
    display.plot(ax=ax, cmap=cmap, values_format=".2f" if normalize else "d", colorbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
