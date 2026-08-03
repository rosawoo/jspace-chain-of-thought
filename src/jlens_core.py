"""Shared J-lens utilities: lens loading, model adapter, scores and vectors.

Conventions (locked in NOTES.md):
- A J-lens vector for token t at layer l is row t of ``W_U @ diag(gamma) @ J_l``,
  where ``gamma`` is the final RMSNorm scale (folded so vectors match the
  model's readout path) and ``W_U`` is the unembedding matrix (tied to the
  input embeddings on Qwen3-4B).
- "Activation strength" of vector t on residual h is ``<v_t, h> / ||v_t||``.
  The RMS division of the final norm is a per-vector positive scalar and does
  not change per-position rankings, so it is omitted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


class record_residuals:
    """Context manager capturing block outputs (residual stream) at layers."""

    def __init__(self, layers_module, at: list[int]):
        self.layers_module = layers_module
        self.at = at
        self.acts: dict[int, torch.Tensor] = {}

    def __enter__(self):
        self.handles = []
        for layer in self.at:
            def hook(mod, args, output, layer=layer):
                h = output[0] if isinstance(output, tuple) else output
                self.acts[layer] = h.detach()
            self.handles.append(
                self.layers_module[layer].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def load_lens_matrices(path: str, device: str | torch.device = "cpu",
                       dtype: torch.dtype = torch.float32) -> dict[int, torch.Tensor]:
    """Load ``{layer: J_l}`` from a JacobianLens ``save()`` file or a raw dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    jac = ckpt["J"] if isinstance(ckpt, dict) and "J" in ckpt else ckpt
    if not isinstance(jac, dict) or not all(torch.is_tensor(v) for v in jac.values()):
        raise ValueError(f"{path} does not contain per-layer Jacobian matrices")
    return {int(l): J.to(device=device, dtype=dtype) for l, J in jac.items()}


@dataclass
class ModelAdapter:
    """Handles to the pieces of a HF Qwen3-style model the lens math needs."""

    unembed_weight: torch.Tensor  # [vocab, d_model], final-norm gamma NOT applied
    final_norm_weight: torch.Tensor  # [d_model] RMSNorm scale gamma
    layers: torch.nn.ModuleList
    d_model: int
    vocab_size: int

    @classmethod
    def from_hf(cls, model) -> "ModelAdapter":
        inner = model.model  # Qwen3ForCausalLM -> Qwen3Model
        w_u = model.get_output_embeddings().weight  # tied on Qwen3-4B
        gamma = inner.norm.weight
        return cls(
            unembed_weight=w_u,
            final_norm_weight=gamma,
            layers=inner.layers,
            d_model=w_u.shape[1],
            vocab_size=w_u.shape[0],
        )


class LensSpace:
    """Per-layer J-lens score/vector computations without materializing the
    full [vocab, d_model] vector matrix.

    scores(h, l)   = (W_U * gamma) @ (J_l @ h)          -> [..., vocab]
    strengths      = scores / row_norms[l]              (unit-vector projections)
    vectors(l, ids)= (W_U[ids] * gamma) @ J_l           -> [n, d_model]
    """

    def __init__(self, adapter: ModelAdapter, jacobians: dict[int, torch.Tensor],
                 row_norm_cache: str | None = None):
        self.adapter = adapter
        self.jacobians = jacobians
        self._scaled_wu = None  # lazily built [vocab, d] = W_U * gamma
        self._row_norms: dict[int, torch.Tensor] = {}
        self._row_norm_cache = row_norm_cache
        if row_norm_cache and os.path.exists(row_norm_cache):
            cached = torch.load(row_norm_cache, map_location="cpu", weights_only=True)
            self._row_norms = {int(l): v for l, v in cached.items()}

    @property
    def layers(self) -> list[int]:
        return sorted(self.jacobians)

    def _wu_gamma(self, device, dtype) -> torch.Tensor:
        if (self._scaled_wu is None or self._scaled_wu.device != device
                or self._scaled_wu.dtype != dtype):
            w = self.adapter.unembed_weight.to(device=device, dtype=dtype)
            g = self.adapter.final_norm_weight.to(device=device, dtype=dtype)
            self._scaled_wu = w * g  # rowwise scale of columns by gamma
        return self._scaled_wu

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """J_l @ h for h of shape [..., d_model]."""
        J = self.jacobians[layer].to(device=h.device, dtype=h.dtype)
        return h @ J.T

    def scores(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Unnormalized lens scores <v_t, h> for every vocab token."""
        wu = self._wu_gamma(h.device, h.dtype)
        return self.transport(h, layer) @ wu.T

    def row_norms(self, layer: int, device, dtype=torch.float32,
                  chunk: int = 8192) -> torch.Tensor:
        """||v_t|| for every vocab token at ``layer`` (computed once, cached)."""
        if layer not in self._row_norms:
            wu = self._wu_gamma(device, dtype)
            J = self.jacobians[layer].to(device=device, dtype=dtype)
            norms = torch.empty(wu.shape[0], dtype=torch.float32)
            for i in range(0, wu.shape[0], chunk):
                norms[i:i + chunk] = (wu[i:i + chunk] @ J).norm(dim=1).float().cpu()
            self._row_norms[layer] = norms
            if self._row_norm_cache:
                torch.save({str(l): v for l, v in self._row_norms.items()},
                           self._row_norm_cache)
        return self._row_norms[layer].to(device)

    def strengths(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Projections of h onto unit-normed J-lens vectors: [..., vocab]."""
        norms = self.row_norms(layer, h.device).to(h.dtype).clamp_min(1e-8)
        return self.scores(h, layer) / norms

    def vectors(self, layer: int, token_ids: torch.Tensor,
                device=None, dtype=torch.float32) -> torch.Tensor:
        """J-lens vectors (unnormalized) for the given token ids: [n, d_model]."""
        device = device or self.adapter.unembed_weight.device
        wu = self._wu_gamma(device, dtype)
        J = self.jacobians[layer].to(device=device, dtype=dtype)
        return wu[token_ids] @ J
