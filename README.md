# BoltzmaNN9

BoltzmaNN9 is a small toolkit for training Restricted Boltzmann Machines (RBMs) with a block-aware architecture, optional preprocessing, and a simple CLI for managing experiments.

## Installation
- Requires Python 3.10+ and PyTorch (CPU or CUDA).
- From the repo root: `python -m venv .venv && .venv/Scripts/Activate.ps1` (PowerShell) then `pip install -e .` or `pip install -e ".[dev]"` for linting/test extras.

## CLI commands
- `python boltzmann.py new_project <path>` – scaffold a project with `config.py`, `data/`, `output/`, and a synthetic data generator.
- `python boltzmann.py preprocess_raw --config <config.py>` – run `DataPreprocessor` to turn raw CSV columns into binary features; relative paths in the config are resolved against the config file location.
- `python boltzmann.py train --project <project_dir>` – train an RBM using the project’s `config.py`; creates a timestamped run under `output/`.
- `python boltzmann.py evaluate --run <run_dir or project_dir>` – evaluate a saved run (defaults to the latest run if a project path is given).
- `python boltzmann.py list --project <project_dir>` – list available runs.

## Configuration basics
`config.py` is a plain Python dict (see `src/templates/config.py` for a template):
- `device`: `"auto"`/`cpu`/`cuda:0`/`mps`.
- `data`: `csv_path` (training data), optional `drop_cols`.
- `model`: visible/hidden block sizes, cross-block restrictions, initialization.
- `preprocess`: quantile bounds, category limits, missing-bit toggles.
- `dataloader` / `train` / `eval` / `conditional`: loader sizes, training hyperparameters, evaluation options.
All relative paths in preprocessing are resolved relative to the config file; training also adjusts `data.csv_path` relative to the project directory.

## Project template
`python boltzmann.py new_project demo_project` creates:
- `demo_project/config.py` – prefilled with the data path `data/data.csv`.
- `demo_project/data/synthetic_generator.py` – generates a demo dataset and plot.
- `demo_project/output/` – destination for runs (`run_<timestamp>`).

## Development
- Run tests (if added): `python -m pytest`.
- Lint (if installed): `ruff check .` and `black --check .`.
- Key sources: `boltzmann.py` (CLI), `src/boltzmann/model.py` (RBM), `src/boltzmann/preprocessor.py` (raw → binary), `src/boltzmann/pipeline.py` (training/eval flow).

See `docs/QUICKSTART.md` for a step-by-step guide and `docs/TECHNICAL_DOCUMENTATION.md` for module-level details.
