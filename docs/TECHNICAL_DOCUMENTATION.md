# Technical Documentation

This document outlines the main modules, data flow, and key conventions in BoltzmaNN9.

## CLI entrypoint (`boltzmann.py`)
- Subcommands: `new_project`, `preprocess_raw`, `train`, `evaluate`, `list`.
- `train`/`evaluate` adjust relative `data.csv_path` against the project directory; `preprocess_raw` passes the config directory to the preprocessor for path resolution.
- Runs are created under `<project>/output/run_<timestamp>[/<suffix>]` with model, config copy, logs, history, plots, and metrics.

## Configuration (`src/boltzmann/config.py`, `src/templates/config.py`)
- `load_config` imports a Python file exposing a `config` dict.
- Sections: `device`, `data`, `model`, `preprocess`, `dataloader`, `train`, `eval`, `conditional`.
- Relative paths in preprocessing are resolved against the config file; training makes `data.csv_path` relative to the project directory when not absolute.

## Preprocessing (`src/boltzmann/preprocessor.py`)
- `DataPreprocessor(config, config_dir=...)` turns raw CSV columns into binary features for RBMs.
- `model.visible_blocks` maps `"path/to.csv/column"` → `K` bits. Paths are resolved with `_resolve_path` using `config_dir`.
- Type handling:
  - Binary columns pass through with optional missing-bit.
  - Float columns are quantized to Gray code with `K` bits using quantiles (`q_low`/`q_high`).
  - Object/categorical columns become one-hot with UNK/MISSING handling.
- Outputs: processed CSV at `data.csv_path` (resolved) and metadata JSON (`.meta.json`) describing specs and produced columns.

## Data loading (`src/boltzmann/data.py`, `src/boltzmann/pipeline.py`)
- `BMDataset` loads a binary matrix CSV and optional `drop_cols`.
- `split_rbm_loaders` builds train/val/test loaders with configurable splits, shuffling, workers, and pinning.
- `pipeline.create_dataloaders` orchestrates dataset + loaders + device resolution; used by `train_rbm` and `evaluate_rbm`.

## Model (`src/boltzmann/model.py`)
- RBM with visible/hidden block definitions and cross-block masking (`cross_block_restrictions`).
- Training uses persistent contrastive divergence with optional sparsity, momentum, weight decay, gradient clipping, and learning-rate schedules (`constant`, `exponential`, `step`, `cosine`, `plateau`).
- Evaluation: free energy, reconstruction MSE, bit error; sampling and clamped sampling helpers; visualization of block structure.

## Evaluation (`src/boltzmann/tester.py`)
- `RBMTester.conditional_nll` computes conditional negative log-likelihood over specified clamp/target indices.
- Used by CLI `evaluate` and `pipeline.evaluate_rbm`.

## Project and run management (`src/boltzmann/project.py`, `src/boltzmann/run_utils.py`)
- `new_project` scaffolds `config.py`, `data/synthetic_generator.py`, and `output/`.
- `create_run`/`save_run_config`/`get_run_paths` manage output directories and config copies.
- `RunLogger` and helpers in `run_utils` handle logging, history, plots, and metrics persistence.

## Synthetic data template (`src/templates/synthetic_generator.py`, `src/boltzmann/data_generator.py`)
- `GeneratorConfig` and `SyntheticDataGenerator` produce a binary feature matrix and visualization for demo experiments.
- The template script writes `data.csv` and `data_visualization.png` inside a project’s `data/` directory.

## File layout expectations
- Training expects `config["data"]["csv_path"]` to point to a 0/1 feature matrix whose column counts match `model.visible_blocks`.
- Preprocessing can be used to build that matrix from raw CSVs; otherwise supply your own ready-to-train data.
