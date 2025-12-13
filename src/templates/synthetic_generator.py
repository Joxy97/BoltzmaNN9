#!/usr/bin/env python
"""Generate synthetic data for this project.

This script is auto-generated from the BoltzmaNN9 template.
Modify the GeneratorConfig parameters below to customize data generation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.boltzmann.data_generator import SyntheticDataGenerator, GeneratorConfig


def visualize_data(full_df, generator, save_path, show_from=0, show_to=500):
    """Create and save visualization of the generated data."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"[visualize_data] matplotlib/numpy import failed: {e}")
        return

    cfg = generator.config
    K = cfg.k_bins
    n_bits = generator.n_bits

    # Clamp range to valid indices
    show_from = max(0, show_from)
    show_to = min(len(full_df), show_to)

    fig = plt.figure(figsize=(15, 10))

    # Plot 1: Continuous vs Discretized R
    plt.subplot(3, 2, 1)
    plt.plot(full_df["t"][show_from:show_to], full_df["R_continuous"][show_from:show_to],
             alpha=0.5, label="Continuous", linewidth=1)
    plt.plot(full_df["t"][show_from:show_to], full_df["R"][show_from:show_to],
             alpha=0.8, label="Discretized", linewidth=1, drawstyle="steps-post")
    plt.xlabel("Time")
    plt.ylabel("R")
    plt.legend()
    plt.title("Continuous vs Discretized R")
    plt.grid(True, alpha=0.3)

    # Plot 2: Histogram of discretized R
    plt.subplot(3, 2, 2)
    plt.hist(full_df["R"], bins=K, edgecolor="black", alpha=0.7)
    plt.xlabel("R (discretized)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution across {K} bins")
    plt.grid(True, alpha=0.3)

    # Plot 3: Bin indices over time
    plt.subplot(3, 2, 3)
    plt.plot(full_df["t"][show_from:show_to], full_df["R_bin_index"][show_from:show_to],
             drawstyle="steps-post")
    plt.xlabel("Time")
    plt.ylabel("Bin Index")
    plt.title("Bin Index over Time")
    plt.yticks(range(K))
    plt.grid(True, alpha=0.3)

    # Plot 4: Binary bits over time (heatmap style)
    plt.subplot(3, 2, 4)
    binary_matrix = np.array([full_df[f"R_bit_{i}"][show_from:show_to].values for i in range(n_bits)])
    im = plt.imshow(binary_matrix, aspect="auto", cmap="binary", interpolation="nearest")
    plt.xlabel("Time")
    plt.ylabel("Bit Position")
    plt.title(f"Binary Representation over Time ({n_bits} bits)")
    plt.yticks(range(n_bits), [f"Bit {i}" for i in range(n_bits)])
    plt.colorbar(im, label="Bit Value")

    # Plot 5: Bin index distribution
    plt.subplot(3, 2, 5)
    bin_counts = full_df["R_bin_index"].value_counts().sort_index()
    plt.bar(bin_counts.index, bin_counts.values, alpha=0.7, edgecolor="black")
    plt.xlabel("Bin Index")
    plt.ylabel("Frequency")
    plt.title("Bin Index Distribution")
    plt.xticks(range(K))
    plt.grid(True, alpha=0.3)

    # Plot 6: Decision variable vs bin index
    plt.subplot(3, 2, 6)
    df_valid = full_df[full_df["x"].notna()]
    decision_by_bin = df_valid.groupby("R_bin_index")["x"].mean()
    plt.bar(decision_by_bin.index, decision_by_bin.values, alpha=0.7, edgecolor="black")
    plt.xlabel("Bin Index")
    plt.ylabel("Mean Decision (x)")
    plt.title("Average Decision by Bin")
    plt.xticks(range(K))
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")


def main():
    """Generate synthetic data and save to data.csv."""
    # ============================================================
    # MODIFY THESE PARAMETERS TO CUSTOMIZE DATA GENERATION
    # ============================================================
    gen_config = GeneratorConfig(
        n_samples=5000,
        dt=0.1,
        r_min=-2.0,
        r_max=2.0,
        k_bins=16,
        spring_k=5.0,
        sigma=1.0,
        eq_interval=100,
        m0=0.25,
        sigma_eq=0.0,
        lookahead=10,
    )
    seed = 42
    show_from = 0
    show_to = 100
    # ============================================================

    generator = SyntheticDataGenerator(gen_config)
    generator.print_info()

    full_df, simplified_df = generator.generate(seed=seed)

    # Save to data folder
    output_path = Path(__file__).parent / "data.csv"
    simplified_df.to_csv(output_path, index=False)

    print(f"\nSaved training data to: {output_path}")
    print(f"  Columns: {list(simplified_df.columns)}")
    print(f"  Rows: {len(simplified_df)}")

    # Generate visualization
    plot_path = Path(__file__).parent / "data_visualization.png"
    visualize_data(full_df, generator, plot_path, show_from=show_from, show_to=show_to)


if __name__ == "__main__":
    main()
