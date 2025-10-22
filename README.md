# ADS-B Signal Classification

This repository provides a full training pipeline for recognising 10 ADS-B
signal categories from complex baseband I/Q samples stored in MATLAB v7.3
(``.mat``) files.  Each sample is represented by 4 800 time steps with two
channels (in-phase and quadrature).

## Training the model

1. Prepare the dataset in two `.mat` files: one containing the combined
   training/validation samples (e.g. `ADS-B_Train_10X360-2_5-10-15-20dB.mat`
   with 32 400 labelled examples) and another containing the held-out test set
   (e.g. `ADS-B_Test_10X360-2_5-10-15-20dB.mat` with 3 600 labelled examples).
   Each file must expose a complex I/Q tensor named `signal` (shape
   `(num_samples, 4800, 2)` or an equivalent permutation such as
   `(2, 4800, num_samples)`) and a `label` array.  Custom keys can be provided
   through command-line arguments if required.
2. Launch the training script:

```bash
python train_adsb.py \
    --train-data path/to/ADS-B_Train_10X360-2_5-10-15-20dB.mat \
    --test-data path/to/ADS-B_Test_10X360-2_5-10-15-20dB.mat \
    --output-dir outputs/adsb_run \
    --epochs 80 \
    --batch-size 128 \
    --log-interval 50
```

Additional flags are available for specifying dataset keys inside the `.mat`
files (`--feature-key`/`--label-key` for the training file and
`--test-feature-key`/`--test-label-key` for the test file), selecting the
validation ratio (`--val-ratio`), adjusting optimiser and scheduler
hyper-parameters (`--lr-factor`, `--lr-patience`, `--lr-threshold`, `--min-lr`)
and controlling early stopping sensitivity (`--patience`, `--early-stop-delta`).
Use `--log-interval` to display intra-epoch progress updates including the
running recognition rate.

The script automatically splits the training file into training and validation
subsets (90 %/10 % by default with stratification), evaluates on the dedicated
test file, trains the lightweight CNN–Transformer architecture and saves:

- The best model weights (`best_model.pt`).
- Training curves for loss and accuracy (`training_curves.png`).
- Normalised and raw confusion matrices (`confusion_matrix.png` and
  `confusion_matrix_counts.png`).
- JSON files containing the training history, final metrics and a detailed
  classification report.

All outputs are placed in the directory specified via `--output-dir`.
