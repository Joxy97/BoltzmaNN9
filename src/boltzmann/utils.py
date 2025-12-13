"""Utility functions for device resolution and memory management."""

from __future__ import annotations

import torch


def resolve_device(device_cfg: str | None) -> torch.device:
    """Resolve device configuration to a torch.device.

    Args:
        device_cfg: Device configuration string. Can be:
            - None or "auto": Auto-detect best available device
            - "cpu": Use CPU
            - "cuda:0", "cuda:1", etc.: Use specific CUDA device
            - "mps": Use Apple Metal Performance Shaders

    Returns:
        torch.device instance for the resolved device.
    """
    if device_cfg is None or device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def resolve_pin_memory(pin_cfg, device: torch.device) -> bool:
    """Resolve pin_memory configuration based on device type.

    Args:
        pin_cfg: Pin memory configuration. Can be "auto", True, or False.
        device: The target torch device.

    Returns:
        Boolean indicating whether to pin memory.
    """
    if pin_cfg == "auto":
        return device.type == "cuda"
    return bool(pin_cfg)
