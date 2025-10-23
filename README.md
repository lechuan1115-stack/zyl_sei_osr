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
2. Launch the training script once with `python train_adsb.py`.  If a
   `training_config.json` file does not yet exist, the script will generate a
   template alongside helpful defaults (including the Windows paths mentioned
   above) and exit with a message prompting you to edit the dataset locations.
   Subsequent runs will read the JSON configuration directly and start the
   training loop without requiring any command-line options.

All training options previously exposed via CLI flags remain available inside
`training_config.json`.  Each key mirrors the argument name (for example,
`"val_ratio"`, `"lr_factor"`, `"log_interval"`), so you can tailor the
validation split, optimiser schedule, early-stopping patience and logging
frequency from that single file.  Remove any entry to fall back to the defaults
defined in `train_adsb.py`.

The script automatically splits the training file into training and validation
subsets (90 %/10 % by default with stratification), evaluates on the dedicated
test file, trains the lightweight CNN–Transformer architecture and saves:

- The best model weights (`best_model.pt`).
- Training curves for loss and accuracy (`training_curves.png`).
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
