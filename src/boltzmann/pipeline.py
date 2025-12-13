"""RBM training and evaluation pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .config import load_config
from .data import BMDataset, split_rbm_loaders
from .model import RBM
from .tester import RBMTester
from .utils import resolve_device, resolve_pin_memory


def create_dataloaders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create dataset and dataloaders from config.

    Args:
        cfg: Configuration dictionary with 'data' and 'dataloader' sections.

    Returns:
        Dictionary with 'dataset', 'train', 'val', 'test' loaders, and 'device'.
    """
    device = resolve_device(cfg.get("device", "auto"))

    data_cfg = cfg.get("data", {})
    csv_path = data_cfg.get("csv_path")
    if not csv_path:
        raise ValueError("csv_path must be specified in config['data']")

    drop_cols = data_cfg.get("drop_cols", [])
    dataset = BMDataset(csv_path, drop_cols=drop_cols)

    dl = cfg.get("dataloader", {})
    pin_memory = resolve_pin_memory(dl.get("pin_memory", "auto"), device)

    loaders = split_rbm_loaders(
        dataset,
        batch_size=dl.get("batch_size", 256),
        split=tuple(dl.get("split", (0.8, 0.1, 0.1))),
        seed=dl.get("seed", 42),
        shuffle_train=dl.get("shuffle_train", True),
        num_workers=dl.get("num_workers", 0),
        pin_memory=pin_memory,
        drop_last_train=dl.get("drop_last_train", True),
    )

    return {
        "dataset": dataset,
        "train": loaders["train"],
        "val": loaders["val"],
        "test": loaders["test"],
        "device": device,
    }


def train_rbm(config_path: str | Path) -> Dict[str, Any]:
    """Train an RBM model.

    Args:
        config_path: Path to configuration .py file.

    Returns:
        Dictionary containing:
            - model: Trained RBM model.
            - history: Training history.
            - device: Device used for training.
            - config: The loaded configuration dictionary.
    """
    cfg = load_config(config_path)

    # Setup
    loaders = create_dataloaders(cfg)
    device = loaders["device"]

    print(f"Using device: {device}")
    print(f"Loaded dataset: {len(loaders['dataset'])} samples")
    print(f"  Columns: {loaders['dataset'].columns}")

    # Model
    model_cfg = dict(cfg.get("model", {}))
    model = RBM(model_cfg).to(device)

    # Train
    train_cfg = dict(cfg.get("train", {}))
    history = model.fit(
        loaders["train"],
        val_loader=loaders["val"],
        **train_cfg,
    )

    return {
        "model": model,
        "history": history,
        "device": device,
        "config": cfg,
    }


def evaluate_rbm(
    model: RBM,
    config_path: str | Path,
) -> Dict[str, Any]:
    """Evaluate a trained RBM model.

    Args:
        model: Trained RBM model.
        config_path: Path to configuration .py file.

    Returns:
        Dictionary containing:
            - test_metrics: Basic test metrics (free energy, reconstruction).
            - conditional_results: Conditional NLL evaluation results.
    """
    cfg = load_config(config_path)

    loaders = create_dataloaders(cfg)
    device = loaders["device"]
    model = model.to(device)

    print(f"Using device: {device}")
    print(f"Evaluating on {len(loaders['test'].dataset)} test samples")

    # Basic metrics
    eval_cfg = cfg.get("eval", {})
    test_metrics = model.evaluate(loaders["test"], recon_k=eval_cfg.get("recon_k", 1))
    print("Test metrics:", test_metrics)

    # Conditional NLL
    cond_cfg = cfg.get("conditional", {})
    tester = RBMTester(
        model=model,
        test_dataloader=loaders["test"],
        clamp_idx=cond_cfg["clamp_idx"],
        target_idx=cond_cfg["target_idx"],
    )
    conditional_results = tester.conditional_nll(
        n_samples=cond_cfg.get("n_samples", 100),
        burn_in=cond_cfg.get("burn_in", 500),
        thin=cond_cfg.get("thin", 10),
    )

    return {
        "test_metrics": test_metrics,
        "conditional_results": conditional_results,
    }


def save_model(
    model: RBM,
    path: str | Path,
    config_path: str | Path | None = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: RBM model to save.
        path: Path to save checkpoint.
        config_path: Optional path to config file (will be saved as reference).
    """
    path = Path(path)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "nv": model.nv,
        "nh": model.nh,
    }

    if config_path is not None:
        # Load and save the actual config dict for portability
        cfg = load_config(config_path)
        checkpoint["config"] = cfg
        checkpoint["config_path"] = str(config_path)

    torch.save(checkpoint, path)
    print(f"Model saved to: {path}")


def load_model(
    path: str | Path,
    device: str | None = None,
) -> tuple[RBM, Dict[str, Any] | None]:
    """Load model from checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to load model to. If None, uses CPU.

    Returns:
        Tuple of (model, config) where config may be None if not saved.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Reconstruct config for model initialization
    cfg = checkpoint.get("config")
    if cfg is not None:
        model_cfg = cfg.get("model", {})
    else:
        # Fallback: create minimal config from saved dimensions
        nv = checkpoint.get("nv", checkpoint["model_state_dict"]["W"].shape[0])
        nh = checkpoint.get("nh", checkpoint["model_state_dict"]["W"].shape[1])
        model_cfg = {
            "visible_blocks": {"v": nv},
            "hidden_blocks": {"h": nh},
        }

    model = RBM(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device:
        model = model.to(device)

    print(f"Model loaded from: {path}")
    return model, cfg
