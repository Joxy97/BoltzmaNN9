"""Configuration loading utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load configuration from a Python file.

    The config file should contain a dictionary named `config`.

    Args:
        config_path: Path to the configuration .py file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file doesn't contain a 'config' dictionary.

    Example config.py:
        config = {
            "device": "auto",
            "data": {"csv_path": "data.csv", ...},
            ...
        }
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load config from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract the config dictionary
    if not hasattr(module, "config"):
        raise ValueError(
            f"Config file must contain a 'config' dictionary: {config_path}"
        )

    config = getattr(module, "config")

    if not isinstance(config, dict):
        raise ValueError(
            f"'config' must be a dictionary, got {type(config).__name__}"
        )

    return config
