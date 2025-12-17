"""Testing utilities for RBM conditional probability evaluation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Sequence

import torch
from tqdm.auto import tqdm


class RBMTester:
    """Test RBM conditional probability estimation.

    Evaluates how well the RBM can predict target visible units
    given clamped (observed) visible units.

    Args:
        model: Trained RBM model.
        test_dataloader: DataLoader for test data.
        clamp_idx: Indices of visible units to clamp (condition on).
        target_idx: Indices of visible units to predict.
        device: Target device for computations.
    """

    def __init__(
        self,
        model,
        test_dataloader,
        clamp_idx: Sequence[int],
        target_idx: Sequence[int],
        *,
        device=None,
    ):
        self.model = model
        self.test_dataloader = test_dataloader
        self.clamp_idx = list(clamp_idx)
        self.target_idx = list(target_idx)
        self.device = device or model.W.device

    @staticmethod
    def _bits_to_int(bits: torch.Tensor) -> int:
        """Convert binary bits to integer (LSB-first)."""
        weights = 2 ** torch.arange(bits.numel(), device=bits.device)
        return int((bits * weights).sum().item())

    @torch.no_grad()
    def conditional_nll(
        self,
        *,
        n_samples: int = 500,
        burn_in: int = 300,
        thin: int = 10,
        laplace_alpha: float = 1.0,
        log_every: int = 50,
    ) -> Dict[str, Any]:
        """Compute conditional negative log-likelihood over test set.

        For each test sample, clamps the specified visible units and
        samples from the conditional distribution p(target | clamp).

        Computes per-bit NLL using empirical bit frequencies, which scales
        to large target dimensions.

        Args:
            n_samples: Number of MCMC samples per test point.
            burn_in: Burn-in steps for MCMC.
            thin: Thinning interval between samples.
            laplace_alpha: Laplace smoothing parameter for per-bit probabilities.
            log_every: Log progress every N samples.

        Returns:
            Dictionary with:
                - mean_nll_nats: Mean NLL in nats (sum over target bits).
                - mean_nll_bits: Mean NLL in bits.
                - mean_nll_per_bit: Mean NLL per target bit.
                - nll_nats_per_sample: List of per-sample NLL in nats.
                - nll_bits_per_sample: List of per-sample NLL in bits.
        """
        self.model.eval()

        ln2 = math.log(2.0)

        nlls_nats = []
        nlls_bits = []

        total_points = len(self.test_dataloader.dataset)
        n_target_bits = len(self.target_idx)

        outer_pbar = tqdm(
            total=total_points,
            desc="RBM conditional NLL",
            leave=True,
        )

        for batch_idx, v in enumerate(self.test_dataloader):
            v = v.to(self.device)

            for i in range(v.size(0)):
                # Clamp visible units
                v_clamp = torch.zeros(
                    self.model.nv,
                    device=self.device,
                    dtype=self.model.W.dtype,
                )
                v_clamp[self.clamp_idx] = v[i, self.clamp_idx]

                # True target bits
                true_bits = v[i, self.target_idx]

                # Sample conditional
                samples = self.model.sample_clamped(
                    v_clamp=v_clamp,
                    clamp_idx=self.clamp_idx,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    thin=thin,
                )

                # Get target bits from samples: (n_samples, n_target_bits)
                sampled_bits = samples[:, self.target_idx]

                # Compute per-bit empirical probability with Laplace smoothing
                # For each target bit, estimate P(bit=true_value)
                bit_matches = (sampled_bits == true_bits.unsqueeze(0)).float()
                bit_probs = (bit_matches.sum(dim=0) + laplace_alpha) / (n_samples + 2 * laplace_alpha)

                # NLL = -sum(log(p_i)) for each bit
                # Clamp to avoid log(0)
                bit_probs = bit_probs.clamp(min=1e-10, max=1 - 1e-10)
                nll_nats = -torch.log(bit_probs).sum().item()
                nll_bits = nll_nats / ln2

                nlls_nats.append(nll_nats)
                nlls_bits.append(nll_bits)

                # Logging
                if len(nlls_nats) % log_every == 0:
                    mean_nats = sum(nlls_nats) / len(nlls_nats)
                    mean_bits = sum(nlls_bits) / len(nlls_bits)
                    outer_pbar.set_postfix(
                        nll_nats=f"{mean_nats:.3f}",
                        nll_bits=f"{mean_bits:.3f}",
                        per_bit=f"{mean_nats / n_target_bits:.4f}",
                    )

                outer_pbar.update(1)

        outer_pbar.close()

        mean_nats = (
            float(sum(nlls_nats) / len(nlls_nats)) if nlls_nats else float("nan")
        )
        mean_bits = (
            float(sum(nlls_bits) / len(nlls_bits)) if nlls_bits else float("nan")
        )
        mean_per_bit = mean_nats / n_target_bits if n_target_bits > 0 else float("nan")

        return {
            "mean_nll_nats": mean_nats,
            "mean_nll_bits": mean_bits,
            "mean_nll_per_bit": mean_per_bit,
            "n_target_bits": n_target_bits,
            "nll_nats_per_sample": nlls_nats,
            "nll_bits_per_sample": nlls_bits,
        }
