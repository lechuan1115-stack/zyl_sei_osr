# ADS-B Signal Classification

This repository provides a full training pipeline for recognising 10 ADS-B
signal categories from complex baseband I/Q samples stored in MATLAB v7.3
(``.mat``) files.  Each sample is represented by 4 800 time steps with two
channels (in-phase and quadrature).

## Training the model

1. Prepare the dataset in a `.mat` file. The file must contain at least two
datasets: the complex I/Q tensor of shape `(num_samples, 4800, 2)` (the loader
also accepts equivalent permutations such as `(2, 4800, num_samples)`) and the
corresponding labels.
2. Launch the training script:

```bash
python train_adsb.py \
    --data path/to/dataset.mat \
    --output-dir outputs/adsb_run \
    --epochs 80 \
    --batch-size 128
```

Additional flags are available for specifying dataset keys inside the `.mat`
file (`--feature-key` and `--label-key`), adjusting optimiser hyper-parameters
and controlling early stopping.

The script automatically splits the dataset into training/validation/test sets
(70 %/15 %/15 %) with stratification, trains the CNN–Transformer architecture
and saves:

- The best model weights (`best_model.pt`).
- Training curves for loss and accuracy (`training_curves.png`).
- Normalised and raw confusion matrices (`confusion_matrix.png` and
  `confusion_matrix_counts.png`).
- JSON files containing the training history, final metrics and a detailed
  classification report.

All outputs are placed in the directory specified via `--output-dir`.
