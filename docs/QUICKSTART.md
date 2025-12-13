# Quickstart

Get up and running with BoltzmaNN9 using the built-in project template and synthetic data generator.

## 1) Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
# or include dev tools
pip install -e ".[dev]"
```

## 2) Scaffold a project
```powershell
python boltzmann.py new_project demo_project
```
Created layout:
- `demo_project/config.py` – experiment settings (edit as needed).
- `demo_project/data/synthetic_generator.py` – produces `data.csv` and a plot.
- `demo_project/output/` – training runs will be stored here.

## 3) Generate demo data
```powershell
cd demo_project/data
python synthetic_generator.py
cd ../..
```
This writes `demo_project/data/data.csv` and `data_visualization.png`.

## 4) Train
```powershell
python boltzmann.py train --project demo_project
```
This loads `demo_project/config.py`, trains an RBM, and writes a run under `demo_project/output/run_<timestamp>/` containing `model.pt`, `training.log`, `history.json`, and plots.

## 5) Evaluate
Evaluate the latest run in the project:
```powershell
python boltzmann.py evaluate --run demo_project
```
Or a specific run:
```powershell
python boltzmann.py evaluate --run demo_project/output/run_<timestamp>
```

## 6) List runs
```powershell
python boltzmann.py list --project demo_project
```

## 7) Preprocess external raw data (optional)
If you have your own raw CSVs and a `visible_blocks` mapping, place a `config.py` alongside them and run:
```powershell
python boltzmann.py preprocess_raw --config path/to/config.py
```
Relative paths inside the config are resolved against the config file’s directory. The preprocessor writes a processed binary CSV and a matching `.meta.json`.

## Notes
- `config.py` controls devices, data paths, block sizes, dataloader splits, and training hyperparameters; see `src/templates/config.py` for a reference structure.
- Training automatically treats `data.csv_path` as relative to the project directory when not absolute.
- For custom datasets without preprocessing, ensure `config["data"]["csv_path"]` points to a binary (0/1) feature matrix matching your visible block sizes.
