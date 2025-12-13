"""
Restricted Boltzmann Machine (RBM) implementation in PyTorch
with cross-block restrictions (weight masking).

- Binary visible/hidden units
- PCD (persistent contrastive divergence)
- Momentum updates, weight decay, gradient clipping
- LR schedules (constant/exponential/step/cosine/plateau)
- Optional hidden sparsity regularization
- Optional early stopping with validation monitoring

Cross-block restrictions:
    config["model"]["cross_block_restrictions"] = [("v_block", "h_block"), ...]

These pairs indicate which V-block × H-block submatrices in W must be forced to 0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class RBM(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # Accept either a full app config with "model" key, or a model-only dict.
        model_cfg = config.get("model", config)

        visible_blocks: Dict[str, Any] = model_cfg["visible_blocks"]
        hidden_blocks: Dict[str, Any] = model_cfg["hidden_blocks"]
        restrictions = model_cfg.get("cross_block_restrictions", []) or []

        self.visible_blocks = {k: int(v) for k, v in visible_blocks.items()}
        self.hidden_blocks = {k: int(v) for k, v in hidden_blocks.items()}

        self.nv = sum(self.visible_blocks.values())
        self.nh = sum(self.hidden_blocks.values())

        # --------- build block ranges (name -> (start, end)) ----------
        self._v_block_ranges: Dict[str, Tuple[int, int]] = {}
        off = 0
        for name, size in self.visible_blocks.items():
            if size <= 0:
                raise ValueError(f"Visible block {name!r} must have positive size, got {size}.")
            self._v_block_ranges[name] = (off, off + size)
            off += size

        self._h_block_ranges: Dict[str, Tuple[int, int]] = {}
        off = 0
        for name, size in self.hidden_blocks.items():
            if size <= 0:
                raise ValueError(f"Hidden block {name!r} must have positive size, got {size}.")
            self._h_block_ranges[name] = (off, off + size)
            off += size

        # --------- parameters ----------
        self.W = nn.Parameter(torch.empty(self.nv, self.nh))
        self.bv = nn.Parameter(torch.zeros(self.nv))
        self.bh = nn.Parameter(torch.zeros(self.nh))

        nn.init.xavier_uniform_(self.W)

        # --------- mask construction ----------
        # Mask is float tensor with 1.0 for allowed edges and 0.0 for forbidden edges.
        mask = torch.ones(self.nv, self.nh, dtype=self.W.dtype)

        for pair in restrictions:
            if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
                raise ValueError(
                    "Each cross_block_restrictions entry must be a pair (v_block, h_block). "
                    f"Got: {pair!r}"
                )
            v_block, h_block = pair
            if v_block not in self._v_block_ranges:
                raise KeyError(
                    f"Unknown visible block {v_block!r} in cross_block_restrictions. "
                    f"Known: {list(self._v_block_ranges.keys())}"
                )
            if h_block not in self._h_block_ranges:
                raise KeyError(
                    f"Unknown hidden block {h_block!r} in cross_block_restrictions. "
                    f"Known: {list(self._h_block_ranges.keys())}"
                )

            vs, ve = self._v_block_ranges[v_block]
            hs, he = self._h_block_ranges[h_block]
            mask[vs:ve, hs:he] = 0.0

        # register_buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer("mask", mask)

        # enforce mask at init
        with torch.no_grad():
            self.W.mul_(self.mask)

        # Persistent chain for PCD
        self.v_chain: Optional[torch.Tensor] = None

        # Momentum buffers (registered so they move with .to(device))
        self.register_buffer("_vW", torch.zeros_like(self.W))
        self.register_buffer("_vbv", torch.zeros_like(self.bv))
        self.register_buffer("_vbh", torch.zeros_like(self.bh))

        # Plateau scheduler state
        self._plateau_best: Optional[float] = None
        self._plateau_bad_count: int = 0

    # --------------------------------------------------
    # Core distributions
    # --------------------------------------------------

    def hidden_prob(self, v: torch.Tensor) -> torch.Tensor:
        """Compute P(h=1 | v)."""
        return torch.sigmoid(v @ self.W + self.bh)

    def visible_prob(self, h: torch.Tensor) -> torch.Tensor:
        """Compute P(v=1 | h)."""
        return torch.sigmoid(h @ self.W.T + self.bv)

    def _bernoulli(self, p: torch.Tensor) -> torch.Tensor:
        """Sample from Bernoulli distribution."""
        return torch.bernoulli(p)

    # --------------------------------------------------
    # Forward (semantic: inference, NOT training)
    # --------------------------------------------------

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Return P(h=1 | v)."""
        return self.hidden_prob(v.to(self.W.dtype))

    # --------------------------------------------------
    # Phases
    # --------------------------------------------------

    def positive_phase(self, v: torch.Tensor, kind: str = "mean-field"):
        """Compute positive phase statistics."""
        v = v.to(self.W.dtype)
        ph = self.hidden_prob(v)
        h_used = ph if kind == "mean-field" else self._bernoulli(ph)

        pos_W = v.T @ h_used
        pos_bv = v.sum(dim=0)
        pos_bh = h_used.sum(dim=0)
        return pos_W, pos_bv, pos_bh, ph

    @torch.no_grad()
    def negative_phase(
        self,
        batch_size: int,
        k: int = 1,
        kind: str = "mean-field",
        device: Optional[torch.device] = None,
    ):
        """Compute negative phase statistics using PCD."""
        device = device or self.W.device

        if self.v_chain is None or self.v_chain.shape[0] != batch_size:
            self.v_chain = self._bernoulli(
                torch.full((batch_size, self.nv), 0.5, device=device, dtype=self.W.dtype)
            )

        v = self.v_chain

        for _ in range(k):
            h = self._bernoulli(self.hidden_prob(v))
            v = self._bernoulli(self.visible_prob(h))

        self.v_chain = v.detach()

        phk = self.hidden_prob(v)
        h_used = phk if kind == "mean-field" else self._bernoulli(phk)

        neg_W = v.T @ h_used
        neg_bv = v.sum(dim=0)
        neg_bh = h_used.sum(dim=0)
        return neg_W, neg_bv, neg_bh

    # --------------------------------------------------
    # Update helpers (momentum, clipping, regularization)
    # --------------------------------------------------

    @staticmethod
    def _clip_by_value(x: torch.Tensor, clip_value: Optional[float]) -> torch.Tensor:
        if clip_value is None:
            return x
        return x.clamp(min=-clip_value, max=clip_value)

    @staticmethod
    def _clip_by_global_norm(
        dW: torch.Tensor,
        dbv: torch.Tensor,
        dbh: torch.Tensor,
        max_norm: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if max_norm is None:
            return dW, dbv, dbh
        norm = torch.sqrt((dW * dW).sum() + (dbv * dbv).sum() + (dbh * dbh).sum())
        if norm > max_norm:
            scale = max_norm / (norm + 1e-12)
            dW = dW * scale
            dbv = dbv * scale
            dbh = dbh * scale
        return dW, dbv, dbh

    def _apply_update(
        self,
        *,
        lr: float,
        dW: torch.Tensor,
        dbv: torch.Tensor,
        dbh: torch.Tensor,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None,
    ) -> None:
        """Apply parameter updates with momentum, weight decay, clipping, and weight masking."""
        # Mask gradients early (avoids momentum accumulating on forbidden edges)
        if hasattr(self, "mask") and self.mask is not None:
            dW = dW * self.mask

        # L2 weight decay
        if weight_decay and weight_decay > 0.0:
            dW = dW - weight_decay * self.W

        # Clip by value
        dW = self._clip_by_value(dW, clip_value)
        dbv = self._clip_by_value(dbv, clip_value)
        dbh = self._clip_by_value(dbh, clip_value)

        # Clip by global norm
        dW, dbv, dbh = self._clip_by_global_norm(dW, dbv, dbh, clip_norm)

        # Momentum update
        with torch.no_grad():
            if momentum and momentum > 0.0:
                self._vW.mul_(momentum).add_(dW, alpha=lr)
                self._vbv.mul_(momentum).add_(dbv, alpha=lr)
                self._vbh.mul_(momentum).add_(dbh, alpha=lr)

                # keep momentum buffer masked too (optional but good)
                if hasattr(self, "mask") and self.mask is not None:
                    self._vW.mul_(self.mask)

                self.W.add_(self._vW)
                self.bv.add_(self._vbv)
                self.bh.add_(self._vbh)
            else:
                self.W.add_(dW, alpha=lr)
                self.bv.add_(dbv, alpha=lr)
                self.bh.add_(dbh, alpha=lr)

            # Re-apply mask to ensure restricted weights stay zero
            if hasattr(self, "mask") and self.mask is not None:
                self.W.mul_(self.mask)

    # --------------------------------------------------
    # LR scheduling
    # --------------------------------------------------

    def _lr_at_epoch(
        self,
        *,
        base_lr: float,
        epoch: int,
        epochs: int,
        schedule: Optional[Dict[str, Any]] = None,
        current_val_metric: Optional[float] = None,
    ) -> float:
        if not schedule:
            return float(base_lr)

        mode = schedule.get("mode", "constant")
        lr0 = float(base_lr)

        if mode == "constant":
            return lr0

        if mode == "exponential":
            gamma = float(schedule.get("gamma", 0.99))
            return lr0 * (gamma ** (epoch - 1))

        if mode == "step":
            step_size = int(schedule.get("step_size", 10))
            gamma = float(schedule.get("gamma", 0.5))
            n_steps = (epoch - 1) // step_size
            return lr0 * (gamma ** n_steps)

        if mode == "cosine":
            min_lr = float(schedule.get("min_lr", 0.0))
            t = (epoch - 1) / max(1, (epochs - 1))
            return min_lr + 0.5 * (lr0 - min_lr) * (1.0 + math.cos(math.pi * t))

        if mode == "plateau":
            factor = float(schedule.get("factor", 0.5))
            patience = int(schedule.get("patience", 3))
            min_lr = float(schedule.get("min_lr", 1e-6))
            threshold = float(schedule.get("threshold", 1e-4))

            if current_val_metric is None:
                return float(schedule.get("__current_lr", lr0))

            current_lr = float(schedule.get("__current_lr", lr0))

            if self._plateau_best is None or (self._plateau_best - current_val_metric) > threshold:
                self._plateau_best = float(current_val_metric)
                self._plateau_bad_count = 0
                return current_lr

            self._plateau_bad_count += 1
            if self._plateau_bad_count >= patience:
                new_lr = max(min_lr, current_lr * factor)
                schedule["__current_lr"] = new_lr
                self._plateau_bad_count = 0
                return new_lr

            return current_lr

        raise ValueError(f"Unknown lr schedule mode={mode!r}")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    @torch.no_grad()
    def cd_step(
        self,
        v: torch.Tensor,
        *,
        lr: float,
        k: int = 1,
        kind: str = "mean-field",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None,
        sparse_hidden: bool = False,
        rho: float = 0.1,
        lambda_sparse: float = 0.0,
    ) -> None:
        B = v.size(0)

        pos_W, pos_bv, pos_bh, ph = self.positive_phase(v, kind)
        neg_W, neg_bv, neg_bh = self.negative_phase(batch_size=B, k=k, kind=kind, device=v.device)

        dW = (pos_W - neg_W) / B
        dbv = (pos_bv - neg_bv) / B
        dbh = (pos_bh - neg_bh) / B

        if sparse_hidden and lambda_sparse > 0.0:
            err = ph.mean(dim=0) - rho
            dbh = dbh - lambda_sparse * err
            v_ = v.to(self.W.dtype)
            dW = dW - lambda_sparse * (v_.mean(dim=0).unsqueeze(1) * err.unsqueeze(0))

        self._apply_update(
            lr=lr,
            dW=dW,
            dbv=dbv,
            dbh=dbh,
            momentum=momentum,
            weight_decay=weight_decay,
            clip_value=clip_value,
            clip_norm=clip_norm,
        )

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def reconstruct(self, v: torch.Tensor, k: int = 1) -> torch.Tensor:
        v = v.to(self.W.dtype)
        for _ in range(k):
            h = self._bernoulli(self.hidden_prob(v))
            v = self._bernoulli(self.visible_prob(h))
        return v

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.to(self.W.dtype)
        wx_b = v @ self.W + self.bh
        return -v @ self.bv - torch.log1p(torch.exp(wx_b)).sum(dim=1)

    @torch.no_grad()
    def evaluate(self, dataloader, *, recon_k: int = 1) -> Dict[str, float]:
        device = self.W.device

        fe_sum = 0.0
        mse_sum = 0.0
        ber_sum = 0.0
        n_samples = 0

        for v in dataloader:
            v = v.to(device, non_blocking=True)
            B = v.size(0)
            n_samples += B

            fe = self.free_energy(v).mean().item()
            v_rec = self.reconstruct(v, k=recon_k)

            mse = torch.mean((v - v_rec) ** 2).item()
            ber = torch.mean((v != v_rec).to(torch.float32)).item()

            fe_sum += fe * B
            mse_sum += mse * B
            ber_sum += ber * B

        return {
            "free_energy_mean": fe_sum / n_samples,
            "recon_mse_mean": mse_sum / n_samples,
            "recon_bit_error": ber_sum / n_samples,
        }

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    def fit(
        self,
        train_loader,
        *,
        val_loader: Optional[object] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        k: int = 1,
        kind: str = "mean-field",
        eval_every: int = 1,
        recon_k: int = 1,
        lr_schedule: Optional[Dict[str, Any]] = None,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None,
        sparse_hidden: bool = False,
        rho: float = 0.1,
        lambda_sparse: float = 0.0,
        early_stopping: bool = False,
        es_patience: int = 10,
        es_min_delta: float = 1e-4,
    ) -> Dict[str, list]:
        device = self.W.device
        history: Dict[str, list] = {
            "epoch": [],
            "train_free_energy": [],
            "train_recon_mse": [],
            "train_recon_bit_error": [],
            "val_free_energy": [],
            "val_recon_mse": [],
            "val_recon_bit_error": [],
            "lr": [],
        }

        best_val: Optional[float] = None
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        if lr_schedule and lr_schedule.get("mode") == "plateau":
            lr_schedule = dict(lr_schedule)
            lr_schedule["__current_lr"] = float(lr)

        for epoch in range(1, epochs + 1):
            self.train()

            current_lr = self._lr_at_epoch(
                base_lr=float(lr),
                epoch=epoch,
                epochs=epochs,
                schedule=lr_schedule,
                current_val_metric=None,
            )

            for v in train_loader:
                v = v.to(device, non_blocking=True)
                self.cd_step(
                    v,
                    lr=current_lr,
                    k=k,
                    kind=kind,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    clip_value=clip_value,
                    clip_norm=clip_norm,
                    sparse_hidden=sparse_hidden,
                    rho=rho,
                    lambda_sparse=lambda_sparse,
                )

            if epoch % eval_every == 0:
                self.eval()

                train_metrics = self.evaluate(train_loader, recon_k=recon_k)
                history["epoch"].append(epoch)
                history["train_free_energy"].append(train_metrics["free_energy_mean"])
                history["train_recon_mse"].append(train_metrics["recon_mse_mean"])
                history["train_recon_bit_error"].append(train_metrics["recon_bit_error"])

                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader, recon_k=recon_k)
                    val_fe = float(val_metrics["free_energy_mean"])

                    history["val_free_energy"].append(val_fe)
                    history["val_recon_mse"].append(val_metrics["recon_mse_mean"])
                    history["val_recon_bit_error"].append(val_metrics["recon_bit_error"])

                    if lr_schedule and lr_schedule.get("mode") == "plateau":
                        # plateau uses its internal "__current_lr"
                        _ = self._lr_at_epoch(
                            base_lr=float(lr),
                            epoch=epoch,
                            epochs=epochs,
                            schedule=lr_schedule,
                            current_val_metric=val_fe,
                        )
                        current_lr = float(lr_schedule.get("__current_lr", current_lr))

                    if early_stopping:
                        if best_val is None or (best_val - val_fe) > es_min_delta:
                            best_val = val_fe
                            best_state = {k: t.detach().clone() for k, t in self.state_dict().items()}
                            bad_epochs = 0
                        else:
                            bad_epochs += 1
                            if bad_epochs >= es_patience:
                                if best_state is not None:
                                    self.load_state_dict(best_state)
                                history["lr"].append(current_lr)
                                print(
                                    f"Early stopping at epoch {epoch} "
                                    f"(best val FE={best_val:.6f})"
                                )
                                self.visualize_history(history)
                                return history

                    print(
                        f"Epoch {epoch:04d} | lr={current_lr:.3e} | "
                        f"train FE={train_metrics['free_energy_mean']:.4f} "
                        f"val FE={val_fe:.4f} | "
                        f"train recon_mse={train_metrics['recon_mse_mean']:.4f} "
                        f"val recon_mse={val_metrics['recon_mse_mean']:.4f}"
                    )
                else:
                    history["val_free_energy"].append(float("nan"))
                    history["val_recon_mse"].append(float("nan"))
                    history["val_recon_bit_error"].append(float("nan"))
                    print(
                        f"Epoch {epoch:04d} | lr={current_lr:.3e} | "
                        f"train FE={train_metrics['free_energy_mean']:.4f} | "
                        f"train recon_mse={train_metrics['recon_mse_mean']:.4f}"
                    )

                history["lr"].append(current_lr)

        self.visualize_history(history)
        return history

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    def visualize_history(self, history: dict) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[visualize_history] matplotlib import failed: {e}")
            return

        epochs = history.get("epoch", list(range(1, len(history.get("train_free_energy", [])) + 1)))

        def _is_all_nan(xs):
            if not xs:
                return True
            return all((x is None) or (isinstance(x, float) and math.isnan(x)) for x in xs)

        def _plot(ax, y_train_key, y_val_key, title, ylabel):
            y_tr = history.get(y_train_key, [])
            y_va = history.get(y_val_key, [])

            ax.plot(epochs[: len(y_tr)], y_tr, label="train")
            if not _is_all_nan(y_va):
                ax.plot(epochs[: len(y_va)], y_va, label="val")

            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.legend()

        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            _plot(axes[0], "train_free_energy", "val_free_energy", "Free Energy", "mean FE (lower is better)")
            _plot(axes[1], "train_recon_mse", "val_recon_mse", "Reconstruction MSE", "MSE")
            _plot(axes[2], "train_recon_bit_error", "val_recon_bit_error", "Reconstruction Bit Error", "fraction mismatched")

            fig.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[visualize_history] plotting failed: {e}")

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        *,
        burn_in: int = 200,
        thin: int = 10,
        init: str = "random",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or self.W.device
        dtype = self.W.dtype

        if init == "chain" and self.v_chain is not None:
            v = self.v_chain[:1].to(device=device, dtype=dtype)
        else:
            v = torch.bernoulli(torch.full((1, self.nv), 0.5, device=device, dtype=dtype))

        for _ in range(burn_in):
            h = torch.bernoulli(self.hidden_prob(v))
            v = torch.bernoulli(self.visible_prob(h))

        samples = []
        steps_needed = n_samples * thin
        for t in range(steps_needed):
            h = torch.bernoulli(self.hidden_prob(v))
            v = torch.bernoulli(self.visible_prob(h))
            if (t + 1) % thin == 0:
                samples.append(v.squeeze(0).clone())

        self.v_chain = v.detach()
        return torch.stack(samples, dim=0)

    def draw_blocks(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """Visualize the RBM block structure and weight mask.

        Draws a bipartite graph showing visible blocks (bottom) and hidden blocks (top),
        with connections colored by the mask (allowed vs restricted).

        Args:
            save_path: If provided, save the figure to this path.
            show: If True, display the plot interactively.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.collections import LineCollection
        except ImportError as e:
            print(f"[draw_blocks] matplotlib import failed: {e}")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Layout parameters
        v_y = 0.0  # visible layer y-coordinate
        h_y = 1.0  # hidden layer y-coordinate
        node_radius = 0.02
        block_gap = 0.15  # gap between blocks

        # Colors
        v_colors = plt.cm.Set2.colors  # visible block colors
        h_colors = plt.cm.Set3.colors  # hidden block colors
        allowed_color = "#4CAF50"  # green for allowed connections
        restricted_color = "#FFCDD2"  # light red for restricted (shown faintly)

        # Calculate positions for visible units (grouped by block)
        v_positions = []
        v_block_colors = []
        v_block_labels = []
        x_offset = 0.0

        for i, (block_name, (start, end)) in enumerate(self._v_block_ranges.items()):
            block_size = end - start
            color = v_colors[i % len(v_colors)]
            for j in range(block_size):
                x = x_offset + j * 0.08
                v_positions.append((x, v_y))
                v_block_colors.append(color)
            # Store label position (center of block)
            center_x = x_offset + (block_size - 1) * 0.08 / 2
            v_block_labels.append((center_x, block_name, color))
            x_offset += block_size * 0.08 + block_gap

        # Normalize visible positions to [0, 1] range
        if v_positions:
            v_max_x = max(p[0] for p in v_positions)
            v_min_x = min(p[0] for p in v_positions)
            v_range = v_max_x - v_min_x if v_max_x > v_min_x else 1.0
            v_positions = [((p[0] - v_min_x) / v_range * 0.8 + 0.1, p[1]) for p in v_positions]
            v_block_labels = [((x - v_min_x) / v_range * 0.8 + 0.1, name, c) for x, name, c in v_block_labels]

        # Calculate positions for hidden units (grouped by block)
        h_positions = []
        h_block_colors = []
        h_block_labels = []
        x_offset = 0.0

        for i, (block_name, (start, end)) in enumerate(self._h_block_ranges.items()):
            block_size = end - start
            color = h_colors[i % len(h_colors)]
            for j in range(block_size):
                x = x_offset + j * 0.08
                h_positions.append((x, h_y))
                h_block_colors.append(color)
            center_x = x_offset + (block_size - 1) * 0.08 / 2
            h_block_labels.append((center_x, block_name, color))
            x_offset += block_size * 0.08 + block_gap

        # Normalize hidden positions
        if h_positions:
            h_max_x = max(p[0] for p in h_positions)
            h_min_x = min(p[0] for p in h_positions)
            h_range = h_max_x - h_min_x if h_max_x > h_min_x else 1.0
            h_positions = [((p[0] - h_min_x) / h_range * 0.8 + 0.1, p[1]) for p in h_positions]
            h_block_labels = [((x - h_min_x) / h_range * 0.8 + 0.1, name, c) for x, name, c in h_block_labels]

        # Draw connections (edges) based on mask
        mask_np = self.mask.detach().cpu().numpy()
        allowed_lines = []
        restricted_lines = []

        for vi in range(self.nv):
            for hi in range(self.nh):
                vx, vy = v_positions[vi]
                hx, hy = h_positions[hi]
                if mask_np[vi, hi] > 0.5:
                    allowed_lines.append([(vx, vy), (hx, hy)])
                else:
                    restricted_lines.append([(vx, vy), (hx, hy)])

        # Draw restricted connections (faint, behind)
        if restricted_lines:
            lc_restricted = LineCollection(
                restricted_lines, colors=restricted_color, linewidths=0.3, alpha=0.3
            )
            ax.add_collection(lc_restricted)

        # Draw allowed connections
        if allowed_lines:
            lc_allowed = LineCollection(
                allowed_lines, colors=allowed_color, linewidths=0.5, alpha=0.6
            )
            ax.add_collection(lc_allowed)

        # Draw visible nodes
        for (x, y), color in zip(v_positions, v_block_colors):
            circle = plt.Circle((x, y), node_radius, color=color, ec="black", linewidth=0.5, zorder=3)
            ax.add_patch(circle)

        # Draw hidden nodes
        for (x, y), color in zip(h_positions, h_block_colors):
            circle = plt.Circle((x, y), node_radius, color=color, ec="black", linewidth=0.5, zorder=3)
            ax.add_patch(circle)

        # Draw block labels
        for x, name, color in v_block_labels:
            ax.text(x, v_y - 0.08, name, ha="center", va="top", fontsize=10, fontweight="bold",
                    color="black", bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

        for x, name, color in h_block_labels:
            ax.text(x, h_y + 0.08, name, ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color="black", bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

        # Layer labels
        ax.text(0.02, v_y, "Visible", ha="left", va="center", fontsize=12, fontweight="bold")
        ax.text(0.02, h_y, "Hidden", ha="left", va="center", fontsize=12, fontweight="bold")

        # Legend
        allowed_patch = mpatches.Patch(color=allowed_color, label="Allowed connections")
        restricted_patch = mpatches.Patch(color=restricted_color, label="Restricted (masked)")
        ax.legend(handles=[allowed_patch, restricted_patch], loc="upper right", fontsize=9)

        # Title with summary
        n_allowed = int(mask_np.sum())
        n_total = self.nv * self.nh
        n_restricted = n_total - n_allowed
        ax.set_title(
            f"RBM Block Structure\n"
            f"Visible: {self.nv} units ({len(self._v_block_ranges)} blocks) | "
            f"Hidden: {self.nh} units ({len(self._h_block_ranges)} blocks)\n"
            f"Connections: {n_allowed}/{n_total} allowed, {n_restricted} restricted",
            fontsize=11
        )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Block diagram saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    @torch.no_grad()
    def sample_clamped(
        self,
        v_clamp: torch.Tensor,
        clamp_idx: Sequence[int],
        *,
        n_samples: int = 1000,
        burn_in: int = 200,
        thin: int = 10,
        init: str = "random",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or self.W.device
        dtype = self.W.dtype

        if v_clamp.dim() == 1:
            v_clamp = v_clamp.unsqueeze(0)
        v_clamp = v_clamp.to(device=device, dtype=dtype)

        clamp_idx_t = torch.as_tensor(clamp_idx, device=device, dtype=torch.long)

        v = torch.bernoulli(torch.full((1, self.nv), 0.5, device=device, dtype=dtype))
        v[:, clamp_idx_t] = v_clamp[:, clamp_idx_t]

        for _ in range(burn_in):
            h = torch.bernoulli(self.hidden_prob(v))
            v = torch.bernoulli(self.visible_prob(h))
            v[:, clamp_idx_t] = v_clamp[:, clamp_idx_t]

        samples = []
        for t in range(n_samples * thin):
            h = torch.bernoulli(self.hidden_prob(v))
            v = torch.bernoulli(self.visible_prob(h))
            v[:, clamp_idx_t] = v_clamp[:, clamp_idx_t]
            if (t + 1) % thin == 0:
                samples.append(v.squeeze(0).clone())

        return torch.stack(samples, dim=0)
