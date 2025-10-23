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
   `(2, 4800, num_samples)`) and a `label` array.  If the tensors are stored
   under different keys, simply edit the configuration file generated in the
   next step.
2. Launch the training script once with `python train_adsb.py`.  If a
   `training_config.json` file does not yet exist, the script will generate a
   template alongside helpful defaults (including the Windows paths mentioned
   above) and exit with a message prompting you to edit the dataset locations.
   Subsequent runs will read the JSON configuration directly and start the
   training loop without requiring any command-line options.

All runtime options are driven by `training_config.json`.  The notable entries
include:

- `train_data` / `test_data`: Absolute or relative paths to the MATLAB files.
- `epochs`, `warmup_epochs`, `min_epochs`: Control the total training duration,
  the length of the linear warm-up (12 epochs by default) and the minimum
  number of epochs to run (70) before early stopping is allowed to trigger.
- `lr` / `min_lr`: Define the cosine learning-rate schedule that decays from the
  peak learning rate (`8e-4`) towards the floor (`5e-5`).
- `batch_size`, `weight_decay`, `max_grad_norm`: Batch configuration, L2
  regularisation and gradient clipping, respectively.
- `patience`, `early_stop_delta`: Early-stopping parameters—thanks to the
  higher patience (60 epochs) and minimum duration the optimiser now explores
  more of the schedule before halting.
- `label_smoothing`: Adds a small amount (0.03) of smoothing to the
  cross-entropy targets, which often stabilises convergence on noisy radio
  datasets.
- `log_interval`: Optional intra-epoch logging cadence.

The loader standardises each channel (zero mean, unit variance) using
statistics estimated from the training split and reuses the same parameters for
validation/testing, minimising covariate shift.  The script automatically
splits the training file into training and validation subsets (90 %/10 % by
default with stratification), evaluates on the dedicated test file, trains the
lightweight CNN–Transformer architecture (now capped at 192 hidden units to
mitigate overfitting) and saves:

- The best model weights (`best_model.pt`).
- Training curves for loss, accuracy and the epoch-wise learning rate
  (`training_curves.png`).
- Normalised and raw confusion matrices (`confusion_matrix.png` and
  `confusion_matrix_counts.png`).
- A per-class accuracy bar chart (`per_class_accuracy.png`) to quickly spot
  under-performing categories.
- A test-set t-SNE projection (`tsne_embeddings.png`) capturing the spatial
  organisation of the learnt representation and a feature distance histogram
  contrasting intra- and inter-class separations (`feature_distance_distribution.png`).
- A prediction confidence histogram (`confidence_histogram.png`) highlighting
  calibration differences between correct and incorrect predictions.
- JSON files containing the training history, final metrics and a detailed
  classification report.

All outputs are placed in the directory specified via `output_dir` in the
configuration file.
