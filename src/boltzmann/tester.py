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
        Computes NLL based on empirical frequency of true target values.

        Args:
            n_samples: Number of MCMC samples per test point.
            burn_in: Burn-in steps for MCMC.
            thin: Thinning interval between samples.
            laplace_alpha: Laplace smoothing parameter.
            log_every: Log progress every N samples.

        Returns:
            Dictionary with:
                - mean_nll_nats: Mean NLL in nats.
                - mean_nll_bits: Mean NLL in bits.
                - nll_nats_per_sample: List of per-sample NLL in nats.
                - nll_bits_per_sample: List of per-sample NLL in bits.
        """
        self.model.eval()

        ln2 = math.log(2.0)

        nlls_nats = []
        nlls_bits = []

        total_points = len(self.test_dataloader.dataset)

        outer_pbar = tqdm(
            total=total_points,
            desc="RBM conditional NLL",
            leave=True,
        )

        for batch_idx, v in enumerate(self.test_dataloader):
            v = v.to(self.device)

            for i in range(v.size(0)):
                # Clamp R_t
                v_clamp = torch.zeros(
                    self.model.nv,
                    device=self.device,
                    dtype=self.model.W.dtype,
                )
                v_clamp[self.clamp_idx] = v[i, self.clamp_idx]

                # True future value
                true_bits = v[i, self.target_idx]
                true_val = self._bits_to_int(true_bits)

                # Sample conditional
                samples = self.model.sample_clamped(
                    v_clamp=v_clamp,
                    clamp_idx=self.clamp_idx,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    thin=thin,
                )

                future_bits = samples[:, self.target_idx]
                future_vals = [
                    self._bits_to_int(future_bits[j])
                    for j in range(future_bits.size(0))
                ]

                # Empirical pmf with Laplace smoothing
                counts = Counter(future_vals)
                K = 2 ** len(self.target_idx)

                prob = (counts.get(true_val, 0) + laplace_alpha) / (
                    n_samples + laplace_alpha * K
                )

                # NLL in nats and bits
                nll_nats = -math.log(prob)
                nll_bits = nll_nats / ln2

                nlls_nats.append(nll_nats)
                nlls_bits.append(nll_bits)

                # Logging
                if len(nlls_nats) % log_every == 0:
                    mean_nats = sum(nlls_nats) / len(nlls_nats)
                    mean_bits = sum(nlls_bits) / len(nlls_bits)
                    outer_pbar.set_postfix(
                        mean_nats=f"{mean_nats:.3f}",
                        mean_bits=f"{mean_bits:.3f}",
                        samples=len(nlls_nats),
                    )

                outer_pbar.update(1)

        outer_pbar.close()

        mean_nats = (
            float(sum(nlls_nats) / len(nlls_nats)) if nlls_nats else float("nan")
        )
        mean_bits = (
            float(sum(nlls_bits) / len(nlls_bits)) if nlls_bits else float("nan")
        )

        return {
            "mean_nll_nats": mean_nats,
            "mean_nll_bits": mean_bits,
            "nll_nats_per_sample": nlls_nats,
            "nll_bits_per_sample": nlls_bits,
        }
