"""Utilities for run logging and visualization."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TextIO


class RunLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_path: Path):
        """Initialize logger.

        Args:
            log_path: Path to log file.
        """
        self.log_path = log_path
        self.log_file: TextIO | None = None
        self._original_stdout = sys.stdout

    def __enter__(self):
        self.log_file = open(self.log_path, "w")
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        if self.log_file:
            self.log_file.close()
        return False

    def write(self, message: str):
        self._original_stdout.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self._original_stdout.flush()
        if self.log_file:
            self.log_file.flush()


def save_history(history: Dict[str, list], path: Path) -> None:
    """Save training history to JSON.

    Args:
        history: Training history dictionary.
        path: Path to save JSON file.
    """
    # Convert any non-serializable types
    serializable = {}
    for key, values in history.items():
        serializable[key] = [
            float(v) if isinstance(v, (int, float)) else v
            for v in values
        ]

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_history(path: Path) -> Dict[str, list]:
    """Load training history from JSON.

    Args:
        path: Path to history JSON file.

    Returns:
        Training history dictionary.
    """
    with open(path) as f:
        return json.load(f)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save evaluation metrics to JSON.

    Args:
        metrics: Metrics dictionary.
        path: Path to save JSON file.
    """
    # Convert nested structures
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (int, float)):
            return float(obj)
        else:
            return obj

    with open(path, "w") as f:
        json.dump(convert(metrics), f, indent=2)


def load_metrics(path: Path) -> Dict[str, Any]:
    """Load evaluation metrics from JSON.

    Args:
        path: Path to metrics JSON file.

    Returns:
        Metrics dictionary.
    """
    with open(path) as f:
        return json.load(f)


def save_plots(history: Dict[str, list], plots_dir: Path, model: Any = None) -> None:
    """Save training plots to directory.

    Args:
        history: Training history dictionary.
        plots_dir: Directory to save plots.
        model: Optional RBM model instance. If provided, saves block diagram.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_plots] matplotlib not available, skipping plots")
        return

    import math

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save RBM block diagram if model is provided
    if model is not None and hasattr(model, "draw_blocks"):
        try:
            model.draw_blocks(save_path=str(plots_dir / "RBM_blocks.png"), show=False)
        except Exception as e:
            print(f"[save_plots] Failed to save block diagram: {e}")

    epochs = history.get(
        "epoch", list(range(1, len(history.get("train_free_energy", [])) + 1))
    )

    def _is_all_nan(xs):
        if not xs:
            return True
        return all(
            (x is None) or (isinstance(x, float) and math.isnan(x)) for x in xs
        )

    # Plot 1: Free Energy
    fig, ax = plt.subplots(figsize=(10, 6))
    y_tr = history.get("train_free_energy", [])
    y_va = history.get("val_free_energy", [])

    ax.plot(epochs[:len(y_tr)], y_tr, label="train", linewidth=2)
    if not _is_all_nan(y_va):
        ax.plot(epochs[:len(y_va)], y_va, label="val", linewidth=2)

    ax.set_title("Free Energy", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Mean FE (lower is better)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "free_energy.png", dpi=150)
    plt.close(fig)

    # Plot 2: Reconstruction MSE
    fig, ax = plt.subplots(figsize=(10, 6))
    y_tr = history.get("train_recon_mse", [])
    y_va = history.get("val_recon_mse", [])

    ax.plot(epochs[:len(y_tr)], y_tr, label="train", linewidth=2)
    if not _is_all_nan(y_va):
        ax.plot(epochs[:len(y_va)], y_va, label="val", linewidth=2)

    ax.set_title("Reconstruction MSE", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "reconstruction_mse.png", dpi=150)
    plt.close(fig)

    # Plot 3: Reconstruction Bit Error
    fig, ax = plt.subplots(figsize=(10, 6))
    y_tr = history.get("train_recon_bit_error", [])
    y_va = history.get("val_recon_bit_error", [])

    ax.plot(epochs[:len(y_tr)], y_tr, label="train", linewidth=2)
    if not _is_all_nan(y_va):
        ax.plot(epochs[:len(y_va)], y_va, label="val", linewidth=2)

    ax.set_title("Reconstruction Bit Error", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Fraction Mismatched", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "bit_error.png", dpi=150)
    plt.close(fig)

    # Plot 4: Learning Rate
    if "lr" in history and history["lr"]:
        fig, ax = plt.subplots(figsize=(10, 6))
        lr_vals = history["lr"]
        ax.plot(epochs[:len(lr_vals)], lr_vals, linewidth=2, color="green")
        ax.set_title("Learning Rate Schedule", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "learning_rate.png", dpi=150)
        plt.close(fig)

    # Combined plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Free Energy
    y_tr = history.get("train_free_energy", [])
    y_va = history.get("val_free_energy", [])
    axes[0].plot(epochs[:len(y_tr)], y_tr, label="train")
    if not _is_all_nan(y_va):
        axes[0].plot(epochs[:len(y_va)], y_va, label="val")
    axes[0].set_title("Free Energy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean FE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MSE
    y_tr = history.get("train_recon_mse", [])
    y_va = history.get("val_recon_mse", [])
    axes[1].plot(epochs[:len(y_tr)], y_tr, label="train")
    if not _is_all_nan(y_va):
        axes[1].plot(epochs[:len(y_va)], y_va, label="val")
    axes[1].set_title("Reconstruction MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Bit Error
    y_tr = history.get("train_recon_bit_error", [])
    y_va = history.get("val_recon_bit_error", [])
    axes[2].plot(epochs[:len(y_tr)], y_tr, label="train")
    if not _is_all_nan(y_va):
        axes[2].plot(epochs[:len(y_va)], y_va, label="val")
    axes[2].set_title("Bit Error")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Fraction")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "training_summary.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to: {plots_dir}")
