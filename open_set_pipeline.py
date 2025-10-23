#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""High-level open-set evaluation workflow for ADS-B experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from open_set import (
    OpenMaxCalibrator,
    OpenSetSplit,
    evaluate_open_set_methods,
    extract_features,
)

__all__ = [
    "SignalGenerator",
    "SignalDiscriminator",
    "train_gan",
    "save_gan_samples",
    "plot_tsne_embeddings",
    "run_open_set_pipeline",
]


@dataclass
class OpenSetOutputs:
    """Container for the open-set metrics computed during evaluation."""

    open_split: Dict[str, Any]
    real_unknown: Dict[str, Dict[str, float]]
    augmented_with_gan: Dict[str, Dict[str, float]]


class SignalGenerator(nn.Module):
    """Generator network used for synthesising unknown ADS-B signals."""

    def __init__(self, latent_dim: int, hidden_dim: int, signal_shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.signal_shape = signal_shape
        output_dim = int(np.prod(signal_shape))
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        output = self.net(z)
        return output.view(z.size(0), *self.signal_shape)


class SignalDiscriminator(nn.Module):
    """Discriminator network distinguishing real vs synthetic unknown samples."""

    def __init__(self, hidden_dim: int, signal_shape: Tuple[int, ...]) -> None:
        super().__init__()
        input_dim = int(np.prod(signal_shape))
        mid_dim = max(hidden_dim // 2, 128)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, mid_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mid_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.net(flat)


def train_gan(
    unknown_data: np.ndarray,
    config: Any,
    device: torch.device,
    output_dir: Path,
) -> Tuple[SignalGenerator, float]:
    """Train the GAN on unknown samples and persist qualitative outputs."""

    if unknown_data.size == 0:
        raise ValueError("GAN training requires at least one unknown sample.")

    signal_shape = unknown_data.shape[1:]
    scale = float(np.max(np.abs(unknown_data)))
    if scale == 0.0:
        scale = 1.0

    normalised = (unknown_data / scale).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(normalised))
    dataloader = DataLoader(
        dataset,
        batch_size=config.gan_batch_size,
        shuffle=True,
        drop_last=False,
    )

    generator = SignalGenerator(config.gan_latent_dim, config.gan_hidden_dim, signal_shape).to(device)
    discriminator = SignalDiscriminator(config.gan_hidden_dim, signal_shape).to(device)

    criterion = nn.BCELoss()
    optim_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(1, config.gan_epochs + 1):
        for (real_batch,) in dataloader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train discriminator
            optim_d.zero_grad()
            real_pred = discriminator(real_batch)
            loss_real = criterion(real_pred, valid)

            noise = torch.randn(batch_size, config.gan_latent_dim, device=device)
            generated = generator(noise).detach()
            fake_pred = discriminator(generated)
            loss_fake = criterion(fake_pred, fake)

            loss_d = (loss_real + loss_fake) * 0.5
            loss_d.backward()
            optim_d.step()

            # Train generator
            optim_g.zero_grad()
            noise = torch.randn(batch_size, config.gan_latent_dim, device=device)
            generated = generator(noise)
            pred = discriminator(generated)
            loss_g = criterion(pred, valid)
            loss_g.backward()
            optim_g.step()

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[GAN] Epoch {epoch:03d}/{config.gan_epochs} | "
                f"loss_d={loss_d.item():.4f} loss_g={loss_g.item():.4f}"
            )

    save_gan_samples(generator, scale, config, device, output_dir)
    return generator, scale


def save_gan_samples(
    generator: SignalGenerator,
    scale: float,
    config: Any,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 16,
) -> None:
    """Render and store qualitative GAN outputs for inspection."""

    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, config.gan_latent_dim, device=device)
        samples = generator(z).cpu().numpy() * scale

    rows = int(math.sqrt(num_samples))
    cols = int(math.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
    axes = np.array(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes.flat):
        if idx >= num_samples:
            ax.axis("off")
            continue
        signal = samples[idx]
        ax.plot(signal[0], label="I", linewidth=0.8)
        ax.plot(signal[1], label="Q", linewidth=0.8)
        ax.set_title(f"Sample {idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "gan_generated_samples.png", dpi=300)
    plt.close(fig)


def plot_tsne_embeddings(
    known_features: np.ndarray,
    unknown_features: np.ndarray,
    gan_features: np.ndarray | None,
    output_dir: Path,
    seed: int,
    max_samples: int,
) -> None:
    """Visualise known, unknown and GAN features using t-SNE."""

    all_features = []
    labels: list[str] = []

    def add(features: np.ndarray | None, label: str) -> None:
        if features is None or features.size == 0:
            return
        all_features.append(features)
        labels.extend([label] * features.shape[0])

    add(known_features, "Known")
    add(unknown_features, "Unknown")
    add(gan_features, "GAN")

    if not all_features:
        return

    feature_array = np.concatenate(all_features, axis=0)
    label_array = np.array(labels)

    if feature_array.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(feature_array.shape[0], max_samples, replace=False)
        feature_array = feature_array[indices]
        label_array = label_array[indices]

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca")
    embedding = tsne.fit_transform(feature_array)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in [("Known", "tab:blue"), ("Unknown", "tab:orange"), ("GAN", "tab:green")]:
        mask = label_array == label
        if np.sum(mask) == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=label,
            s=20,
            alpha=0.7,
        )
    ax.set_title("t-SNE of known vs unknown feature embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "tsne_open_set.png", dpi=300)
    plt.close(fig)


def _compute_class_statistics(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-class mean vectors and shared precision matrix."""

    class_means = []
    for cls in range(num_classes):
        cls_mask = train_labels == cls
        if not np.any(cls_mask):
            raise ValueError(f"No samples found for class index {cls} when computing statistics.")
        class_means.append(train_features[cls_mask].mean(axis=0))
    class_means = np.stack(class_means, axis=0)

    centered = train_features - class_means[train_labels]
    covariance = np.cov(centered, rowvar=False) + 1e-6 * np.eye(centered.shape[1])
    precision = np.linalg.pinv(covariance)
    return class_means, precision


def run_open_set_pipeline(
    model: torch.nn.Module,
    train_loader: DataLoader,
    known_loader: DataLoader,
    unknown_loader: DataLoader,
    *,
    unknown_pool: np.ndarray,
    num_classes: int,
    open_split: OpenSetSplit,
    config: Any,
    device: torch.device,
    output_dir: Path,
) -> OpenSetOutputs:
    """Execute the OpenMax + GAN workflow and persist evaluation artefacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    gan_dir = output_dir / "gan"

    train_logits, train_features, train_labels = extract_features(model, train_loader, device)
    known_logits, known_features, known_labels = extract_features(model, known_loader, device)
    unknown_logits, unknown_features, _ = extract_features(model, unknown_loader, device)

    calibrator = OpenMaxCalibrator(
        tail_size=config.openmax_tail_size,
        alpha=config.openmax_alpha,
    )
    calibrator.fit(train_features, train_logits, train_labels)

    class_means, precision = _compute_class_statistics(train_features, train_labels, num_classes)

    open_set_results = evaluate_open_set_methods(
        known_logits,
        known_features,
        known_labels,
        unknown_logits,
        unknown_features,
        calibrator,
        class_means,
        precision,
    )

    gan_generator, gan_scale = train_gan(unknown_pool, config, device, gan_dir)

    gan_samples = min(unknown_pool.shape[0], 2000)
    if gan_samples == 0:
        raise RuntimeError("Unable to train GAN without unknown samples.")
    with torch.no_grad():
        z = torch.randn(gan_samples, config.gan_latent_dim, device=device)
        gan_signals = (gan_generator(z).cpu().numpy() * gan_scale).astype(np.float32)

    gan_dataset = TensorDataset(
        torch.from_numpy(gan_signals),
        torch.zeros(gan_samples, dtype=torch.int64),
    )
    gan_loader = DataLoader(
        gan_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    gan_logits, gan_features, _ = extract_features(model, gan_loader, device)

    combined_unknown_logits = np.concatenate([unknown_logits, gan_logits], axis=0)
    combined_unknown_features = np.concatenate([unknown_features, gan_features], axis=0)

    open_set_results_with_gan = evaluate_open_set_methods(
        known_logits,
        known_features,
        known_labels,
        combined_unknown_logits,
        combined_unknown_features,
        calibrator,
        class_means,
        precision,
    )

    plot_tsne_embeddings(
        known_features,
        unknown_features,
        gan_features,
        output_dir,
        seed=config.seed,
        max_samples=config.tsne_samples,
    )

    summary = OpenSetOutputs(
        open_split=open_split.as_dict(),
        real_unknown=open_set_results,
        augmented_with_gan=open_set_results_with_gan,
    )

    with open(output_dir / "open_set_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "open_split": summary.open_split,
                "real_unknown": summary.real_unknown,
                "augmented_with_gan": summary.augmented_with_gan,
            },
            fp,
            indent=2,
        )

    return summary
