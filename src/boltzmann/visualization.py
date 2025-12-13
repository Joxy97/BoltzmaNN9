"""Visualization utilities for data exploration."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def plot_simulation_data(
    df: pd.DataFrame,
    *,
    n_bits: int,
    k_bins: int,
    start: int = 0,
    stop: int = 100,
    figsize: tuple[int, int] = (15, 10),
) -> None:
    """Plot simulation data visualization.

    Creates a 6-panel figure showing:
    1. Continuous vs discretized R values
    2. Distribution histogram
    3. Bin indices over time
    4. Binary representation heatmap
    5. Bin index distribution
    6. Decision variable by bin

    Args:
        df: DataFrame with simulation data.
        n_bits: Number of bits used for binary encoding.
        k_bins: Number of discretization bins.
        start: Start index for time series plots.
        stop: End index for time series plots.
        figsize: Figure size tuple.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"matplotlib not available: {e}")
        return

    fig = plt.figure(figsize=figsize)

    # Plot 1: Continuous vs Discretized
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(
        df["t"][start:stop],
        df["R_continuous"][start:stop],
        alpha=0.5,
        label="Continuous",
        linewidth=1,
    )
    plt.xlabel("Time")
    plt.ylabel("R")
    plt.legend()
    plt.title("Continuous vs Discretized R")
    plt.grid(True, alpha=0.3)

    # Plot 2: Histogram
    ax2 = plt.subplot(3, 2, 2)
    plt.hist(df["R"], bins=k_bins, edgecolor="black", alpha=0.7)
    plt.xlabel("R (discretized)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution across {k_bins} bins")
    plt.grid(True, alpha=0.3)

    # Plot 3: Bin indices over time
    ax3 = plt.subplot(3, 2, 3)
    plt.plot(
        df["t"][start:stop], df["R_bin_index"][start:stop], drawstyle="steps-post"
    )
    plt.xlabel("Time")
    plt.ylabel("Bin Index")
    plt.title("Bin Index over Time")
    plt.yticks(range(k_bins))
    plt.grid(True, alpha=0.3)

    # Plot 4: Binary bits over time (heatmap style)
    ax4 = plt.subplot(3, 2, 4)
    binary_matrix = np.array(
        [df[f"R_bit_{i}"][start:stop].values for i in range(n_bits)]
    )
    plt.imshow(binary_matrix, aspect="auto", cmap="binary", interpolation="nearest")
    plt.xlabel("Time")
    plt.ylabel("Bit Position")
    plt.title(f"Binary Representation over Time ({n_bits} bits)")
    plt.yticks(range(n_bits), [f"Bit {i}" for i in range(n_bits)])
    plt.colorbar(label="Bit Value")

    # Plot 5: Bin index distribution
    ax5 = plt.subplot(3, 2, 5)
    bin_counts = df["R_bin_index"].value_counts().sort_index()
    plt.bar(bin_counts.index, bin_counts.values, alpha=0.7, edgecolor="black")
    plt.xlabel("Bin Index")
    plt.ylabel("Frequency")
    plt.title("Bin Index Distribution")
    plt.xticks(range(k_bins))
    plt.grid(True, alpha=0.3)

    # Plot 6: Decision variable vs bin index
    ax6 = plt.subplot(3, 2, 6)
    df_valid = df[df["x"].notna()]
    decision_by_bin = df_valid.groupby("R_bin_index")["x"].mean()
    plt.bar(decision_by_bin.index, decision_by_bin.values, alpha=0.7, edgecolor="black")
    plt.xlabel("Bin Index")
    plt.ylabel("Mean Decision (x)")
    plt.title("Average Decision by Bin")
    plt.xticks(range(k_bins))
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
