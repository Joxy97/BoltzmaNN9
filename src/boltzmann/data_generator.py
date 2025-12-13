"""Synthetic data generation for Boltzmann Machine experiments.

This module generates synthetic time-series data using a discretized
Langevin equation (stochastic harmonic oscillator) for testing RBMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation.

    Attributes:
        n_samples: Number of time steps/samples to generate.
        dt: Time step size for discretization.
        r_min: Minimum allowed value for R.
        r_max: Maximum allowed value for R.
        k_bins: Number of bins for discretizing R.
        spring_k: Spring strength (how strongly R is pulled to equilibrium).
        sigma: Noise strength (stochastic forcing).
        eq_interval: Steps between equilibrium position updates.
        m0: Initial equilibrium value.
        sigma_eq: Size of random shift when equilibrium jumps.
        lookahead: Steps ahead to look for decision variable.
    """

    n_samples: int = 5000
    dt: float = 0.1
    r_min: float = -2.0
    r_max: float = 2.0
    k_bins: int = 16
    spring_k: float = 5.0
    sigma: float = 1.0
    eq_interval: int = 100
    m0: float = 0.25
    sigma_eq: float = 0.0
    lookahead: int = 10


class SyntheticDataGenerator:
    """Generate synthetic stochastic time-series data.

    Uses a discretized Langevin equation for a harmonic oscillator:
        dR/dt = k * (m_t - R) + sigma * noise

    The continuous values are discretized into K bins and encoded as binary.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """Initialize generator with configuration.

        Args:
            config: Generator configuration. Uses defaults if None.
        """
        self.config = config or GeneratorConfig()
        self._setup_bins()

    def _setup_bins(self) -> None:
        """Set up bin edges and centers for discretization."""
        cfg = self.config
        self.bin_edges = np.linspace(cfg.r_min, cfg.r_max, cfg.k_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.n_bits = int(np.ceil(np.log2(cfg.k_bins)))

    def round_to_nearest_bin(self, r_continuous: float) -> tuple[int, float]:
        """Round a continuous R value to the nearest bin.

        Args:
            r_continuous: Continuous R value.

        Returns:
            Tuple of (bin_index, bin_center_value).
        """
        idx = np.argmin(np.abs(self.bin_centers - r_continuous))
        return idx, self.bin_centers[idx]

    @staticmethod
    def bin_index_to_binary(idx: int, n_bits: int) -> str:
        """Convert bin index to binary string (MSB first).

        Args:
            idx: Bin index.
            n_bits: Number of bits to use.

        Returns:
            Binary string representation.
        """
        return format(idx, f"0{n_bits}b")

    @staticmethod
    def binary_to_list(binary_str: str) -> list[int]:
        """Convert binary string to list of integers.

        Args:
            binary_str: Binary string representation.

        Returns:
            List of bit values (0 or 1).
        """
        return [int(bit) for bit in binary_str]

    def _update_equilibrium(self, m_prev: float) -> float:
        """Randomly move equilibrium and clip to valid range."""
        cfg = self.config
        m_new = m_prev + np.random.normal(0.0, cfg.sigma_eq)
        return np.clip(m_new, cfg.r_min, cfg.r_max)

    def _step_r(self, r_prev: float, m_t: float) -> float:
        """One step of discretized Langevin equation."""
        cfg = self.config
        drift = cfg.spring_k * (m_t - r_prev) * cfg.dt
        diffusion = cfg.sigma * np.sqrt(cfg.dt) * np.random.normal()
        r_new = r_prev + drift + diffusion
        return np.clip(r_new, cfg.r_min, cfg.r_max)

    @staticmethod
    def _forward_looking_decision(r_current: float, r_future: float) -> int:
        """Decision rule based on forward return.

        Returns:
            1 if future >= current, else 0.
        """
        return 1 if (r_future - r_current) >= 0 else 0

    def generate(self, seed: Optional[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic data.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (full_dataframe, simplified_dataframe):
                - full_dataframe: Complete simulation data with all columns.
                - simplified_dataframe: Only binary R_t, R_t+lookahead, and x.
        """
        if seed is not None:
            np.random.seed(seed)

        cfg = self.config
        n_bits = self.n_bits

        # Storage
        r_values = []
        r_discretized = []
        r_bin_indices = []
        r_binary_strings = []
        r_binary_lists = []
        m_values = []

        r_t = 0.0
        m_t = cfg.m0

        # Simulation loop
        for t in range(cfg.n_samples):
            if t % cfg.eq_interval == 0 and t > 0:
                m_t = self._update_equilibrium(m_t)

            r_t = self._step_r(r_t, m_t)
            bin_idx, r_disc = self.round_to_nearest_bin(r_t)
            binary_str = self.bin_index_to_binary(bin_idx, n_bits)
            binary_list = self.binary_to_list(binary_str)

            r_values.append(r_t)
            r_discretized.append(r_disc)
            r_bin_indices.append(bin_idx)
            r_binary_strings.append(binary_str)
            r_binary_lists.append(binary_list)
            m_values.append(m_t)

        # Compute decision variable
        x_values = []
        for t in range(cfg.n_samples):
            if t + cfg.lookahead < cfg.n_samples:
                x_t = self._forward_looking_decision(
                    r_discretized[t], r_discretized[t + cfg.lookahead]
                )
            else:
                x_t = np.nan
            x_values.append(x_t)

        # Build full dataframe
        df = pd.DataFrame(
            {
                "t": np.arange(cfg.n_samples),
                "R_continuous": r_values,
                "R": r_discretized,
                "R_bin_index": r_bin_indices,
                "R_binary": r_binary_strings,
                "equilibrium": m_values,
                "x": x_values,
            }
        )

        for i in range(n_bits):
            df[f"R_bit_{i}"] = [bits[i] for bits in r_binary_lists]

        # Build simplified dataframe (binary only)
        dataframe_rows = []
        for t in range(cfg.n_samples - cfg.lookahead):
            row = {}
            for i in range(n_bits):
                row[f"R_t_bit_{i}"] = r_binary_lists[t][i]
            for i in range(n_bits):
                row[f"R_t+10_bit_{i}"] = r_binary_lists[t + cfg.lookahead][i]
            row["x"] = x_values[t]
            dataframe_rows.append(row)

        simplified_df = pd.DataFrame(dataframe_rows)

        return df, simplified_df

    def print_info(self) -> None:
        """Print information about the binary encoding."""
        cfg = self.config
        n_bits = self.n_bits

        print("=" * 60)
        print("BINARY ENCODING INFO")
        print("=" * 60)
        print(f"Number of bins (K): {cfg.k_bins}")
        print(f"Number of bits needed: {n_bits}")
        print(f"\nBin index to binary mapping:")
        for i in range(cfg.k_bins):
            print(
                f"  Bin {i}: {self.bin_index_to_binary(i, n_bits)} -> "
                f"R = {self.bin_centers[i]:.4f}"
            )
