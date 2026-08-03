"""J-space ablation during generation, following the paper's recipe:

  "At each token position, across a band of layers, we identify the k=10 most
   strongly activated J-lens vectors and zero out the residual stream's
   projection onto each. [...] we do not ablate any tokens that appear in the
   top-10 tokens of a clean forward pass."

Implementation decisions (locked in NOTES.md, referenced by HYPOTHESES.md):
- Selection: top-k by |projection onto unit-normed lens vector|, per
  (position, band layer), spare token ids masked out before top-k.
- Zeroing: QR-orthonormalize the k selected (correlated) directions and
  subtract the projection onto their span — exactly zeroes every selected
  vector's projection.
- Sparing: per decode step, a paired clean forward on the *identical current
  context* (separate KV cache fed the same token sequence) provides the clean
  top-10 next-token ids at the new position; prefill positions get their
  per-position clean top-10 from the clean prefill. `spare=False` disables.
- Random control: at each (position, layer) compute the norm the J-ablation
  would remove (with sparing, identically to the J arm), then remove exactly
  that norm along the residual's projection direction within a fixed random
  k-dimensional subspace. Matched per (position, layer) by construction.
- KV cache: hooks edit block outputs, so downstream K/V for the edited
  position derive from the edited residual; later steps read the frozen cache
  and only the new position is edited — "ablate at every position" holds
  without recomputing history.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from .jlens_core import LensSpace


@dataclass
class AblationConfig:
    band_layers: list[int]
    k: int = 10
    mode: str = "jspace"  # "jspace" | "random" | "none"
    spare: bool = True
    spare_top_n: int = 10
    seed: int = 0
    record_selected: bool = False  # aggregate Counter of ablated token ids


@dataclass
class AblationStats:
    removed_norm_sum: float = 0.0
    removed_norm_count: int = 0
    resid_norm_sum: float = 0.0
    positions_ablated: int = 0
    selected_counter: Counter = field(default_factory=Counter)

    @property
    def mean_removed_norm(self) -> float:
        return self.removed_norm_sum / max(1, self.removed_norm_count)

    @property
    def mean_removed_fraction(self) -> float:
        return self.removed_norm_sum / max(1e-9, self.resid_norm_sum)

    def as_dict(self) -> dict:
        return {
            "mean_removed_norm": self.mean_removed_norm,
            "mean_removed_fraction": self.mean_removed_fraction,
            "positions_ablated": self.positions_ablated,
            "top_selected": self.selected_counter.most_common(50),
        }


class JSpaceAblator:
    """Stateful ablation engine. Attach via `hooks()` around a forward pass.

    Before each hooked forward, call `set_spare_ids(ids)` with the clean
    top-N token ids for the positions about to be processed
    (shape [batch, n_positions, spare_top_n]) — or None when spare=False.
    """

    def __init__(self, lens: LensSpace, config: AblationConfig):
        self.lens = lens
        self.cfg = config
        self.stats = AblationStats()
        self._spare_ids: torch.Tensor | None = None
        self._rand_basis: dict[int, torch.Tensor] = {}  # layer -> [k, d] orthonormal
        self._layer_index = {m: l for l, m in
                             ((l, lens.adapter.layers[l]) for l in config.band_layers)}

    def set_spare_ids(self, spare_ids: torch.Tensor | None) -> None:
        self._spare_ids = spare_ids

    def _random_basis(self, layer: int, d: int, device) -> torch.Tensor:
        if layer not in self._rand_basis:
            gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed * 1000 + layer)
            raw = torch.randn(self.cfg.k, d, generator=gen)
            q, _ = torch.linalg.qr(raw.T.float())  # [d, k], orthonormal columns
            self._rand_basis[layer] = q.T.contiguous()  # [k, d]
        return self._rand_basis[layer].to(device)

    def _edit(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Ablate h of shape [batch, n_pos, d] in fp32; returns same dtype."""
        if self.cfg.mode == "none":
            return h
        cfg, lens = self.cfg, self.lens
        orig_dtype = h.dtype
        hf = h.detach().float()  # ablation is never backpropagated through
        B, P, d = hf.shape

        magnitudes = lens.strengths(hf, layer).abs()  # [B, P, vocab]
        if cfg.spare and self._spare_ids is not None:
            spare = self._spare_ids
            if spare.shape[1] != P:  # prefill hooks see full seq; decode sees 1
                spare = spare[:, -P:]
            # zero (not -inf: selection is by absolute magnitude) so spared
            # tokens can never be selected
            magnitudes = magnitudes.scatter(
                -1, spare.to(magnitudes.device), 0.0)

        top_ids = magnitudes.topk(cfg.k, dim=-1).indices  # [B, P, k]

        out = hf.clone()
        for b in range(B):
            for p in range(P):
                ids = top_ids[b, p]
                V = lens.vectors(layer, ids, device=hf.device)  # [k, d] fp32
                Q, _ = torch.linalg.qr(V.T)  # [d, k] orthonormal columns
                coeffs = hf[b, p] @ Q  # [k]
                removed = coeffs.norm()
                if cfg.mode == "jspace":
                    out[b, p] = hf[b, p] - Q @ coeffs
                    if cfg.record_selected:
                        self.stats.selected_counter.update(ids.tolist())
                else:  # random: remove `removed` norm inside a fixed random span
                    Qr = self._random_basis(layer, d, hf.device)  # [k, d]
                    proj = Qr @ hf[b, p]  # [k]
                    direction = (Qr.T @ proj)
                    dnorm = direction.norm().clamp_min(1e-8)
                    out[b, p] = hf[b, p] - removed * direction / dnorm
                self.stats.removed_norm_sum += float(removed)
                self.stats.resid_norm_sum += float(hf[b, p].norm())
                self.stats.removed_norm_count += 1
        self.stats.positions_ablated += B * P
        return out.to(orig_dtype)

    def hooks(self):
        """Context manager registering forward hooks on the band layers."""
        ablator = self

        class _Ctx:
            def __enter__(ctx):
                ctx.handles = []
                for module, layer in ablator._layer_index.items():
                    def hook(mod, args, output, layer=layer):
                        h = output[0] if isinstance(output, tuple) else output
                        h2 = ablator._edit(h, layer)
                        return ((h2, *output[1:]) if isinstance(output, tuple)
                                else h2)
                    ctx.handles.append(module.register_forward_hook(hook))
                return ctx

            def __exit__(ctx, *exc):
                for handle in ctx.handles:
                    handle.remove()

        return _Ctx()


@torch.no_grad()
def generate_with_ablation(
    model,
    input_ids: torch.Tensor,  # [1, T] prompt
    ablator: JSpaceAblator,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 20,
    seed: int = 0,
    eos_token_ids: list[int] | None = None,
) -> dict:
    """Sampled generation under per-step ablation with the paired-clean sparing
    rule. Maintains two KV caches fed the identical (ablated-trajectory) token
    sequence: the clean cache supplies each step's spare set; the hooked cache
    produces the logits that are sampled from.

    Returns dict(sequence, n_new_tokens, hit_eos, stats).
    """
    from transformers import DynamicCache

    device = input_ids.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cfg = ablator.cfg
    eos = set(eos_token_ids or [])

    clean_past, abl_past = DynamicCache(), DynamicCache()

    # ---- prefill ----
    spare_ids = None
    if cfg.spare and cfg.mode != "none":
        clean_out = model(input_ids=input_ids, past_key_values=clean_past,
                          use_cache=True)
        clean_past = clean_out.past_key_values
        spare_ids = clean_out.logits.topk(cfg.spare_top_n, dim=-1).indices  # [1,T,N]
    ablator.set_spare_ids(spare_ids)
    with ablator.hooks():
        abl_out = model(input_ids=input_ids, past_key_values=abl_past,
                        use_cache=True)
    abl_past = abl_out.past_key_values

    tokens: list[int] = []
    hit_eos = False
    next_logits = abl_out.logits[:, -1, :]

    for _ in range(max_new_tokens):
        next_id = _sample(next_logits, temperature, top_p, top_k, gen)
        tokens.append(next_id)
        if next_id in eos:
            hit_eos = True
            break
        step = torch.tensor([[next_id]], device=device)

        spare_ids = None
        if cfg.spare and cfg.mode != "none":
            clean_out = model(input_ids=step, past_key_values=clean_past,
                              use_cache=True)
            clean_past = clean_out.past_key_values
            spare_ids = clean_out.logits.topk(cfg.spare_top_n, dim=-1).indices
        elif cfg.mode != "none" and not cfg.spare:
            # exemption-OFF arm still needs the clean cache advanced? No:
            # without sparing the clean pass is unnecessary; skip it entirely.
            pass
        ablator.set_spare_ids(spare_ids)
        with ablator.hooks():
            abl_out = model(input_ids=step, past_key_values=abl_past,
                            use_cache=True)
        abl_past = abl_out.past_key_values
        next_logits = abl_out.logits[:, -1, :]

    return {
        "sequence": tokens,
        "n_new_tokens": len(tokens),
        "hit_eos": hit_eos,
        "stats": ablator.stats.as_dict(),
    }


def _sample(logits: torch.Tensor, temperature: float, top_p: float,
            top_k: int, gen: torch.Generator) -> int:
    """Temperature + top-k + top-p (nucleus) sampling; CPU generator for
    cross-arm seed pairing."""
    logits = logits[0].float().cpu()
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    if 0 < top_p < 1:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(0)
        cutoff = int((cum < top_p).sum()) + 1
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[sorted_idx[:cutoff]] = True
        probs = probs * mask
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=gen))
