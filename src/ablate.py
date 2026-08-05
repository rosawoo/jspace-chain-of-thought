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
    # Selection semantics. "positive" matches the paper's "most strongly
    # activated" (nonneg J-space combinations); "abs" is the pre-2026-08-03
    # behavior, kept as a labeled variant — zeroing a NEGATIVE projection
    # injects that token's disposition rather than removing content.
    select: str = "positive"  # "positive" | "abs"
    # Removal operator. "span" = QR-exact zeroing of every selected
    # projection (minimal-norm edit achieving that); "pervector" = one-shot
    # per-vector subtraction (milder when vectors correlate; third-party
    # reference behavior).
    projection: str = "span"  # "span" | "pervector"
    # Skip ablation at the first N absolute positions: the official fitting
    # code excludes positions < 16 as attention sinks, so lens vectors were
    # never fit to be valid there.
    skip_first_positions: int = 0
    # Partial-strength removal: out = h - alpha * delta. 1.0 = full zeroing.
    alpha: float = 1.0
    # Restore each position's residual to its pre-ablation norm afterwards
    # (tests whether coherence damage is norm-shock vs content removal).
    renorm: bool = False
    # mode == "random_tokens": operator-matched control — span-remove the
    # J-lens vectors of k RANDOM vocab tokens (fixed per layer by seed)
    # instead of the top-k activated ones. Same operator class/geometry;
    # only the selection is random.
    # fixed_token_ids: ablate exactly these tokens' lens vectors at every
    # position (the paper's fixed-semantic-set recipe); overrides top-k.
    fixed_token_ids: list[int] | None = None


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
        self._position_offset = 0  # absolute index of first position in hook
        self._rand_basis: dict[int, torch.Tensor] = {}  # layer -> [k, d] orthonormal
        self._layer_index = {m: l for l, m in
                             ((l, lens.adapter.layers[l]) for l in config.band_layers)}

    def set_spare_ids(self, spare_ids: torch.Tensor | None) -> None:
        self._spare_ids = spare_ids

    def set_position_offset(self, offset: int) -> None:
        self._position_offset = offset

    def _random_basis(self, layer: int, d: int, device) -> torch.Tensor:
        if layer not in self._rand_basis:
            gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed * 1000 + layer)
            raw = torch.randn(self.cfg.k, d, generator=gen)
            q, _ = torch.linalg.qr(raw.T.float())  # [d, k], orthonormal columns
            self._rand_basis[layer] = q.T.contiguous()  # [k, d]
        return self._rand_basis[layer].to(device)

    def _fixed_vectors(self, layer: int, token_ids, device) -> torch.Tensor:
        """Cache the [k, d] lens-vector matrix for a fixed token list
        (deduplicated: duplicates never change a span)."""
        token_ids = list(dict.fromkeys(token_ids))
        key = (layer, tuple(token_ids))
        if key not in self._rand_basis:
            ids = torch.tensor(token_ids, dtype=torch.long)
            self._rand_basis[key] = self.lens.vectors(layer, ids, device=device)
        return self._rand_basis[key].to(device)

    @staticmethod
    def _span_project(flat: torch.Tensor, V_all: torch.Tensor) -> torch.Tensor:
        """Exact projection of each row of ``flat`` [N, d] onto the span of
        its vector set ``V_all`` [N, k, d], via the Gram pseudo-inverse:
        ``delta = V^T (V V^T)^+ V h``. Unlike batched QR, this is exact for
        rank-deficient sets — duplicate or zeroed rows simply drop out —
        which the positive-selection fallback and spare-masking rely on.
        """
        gram = torch.einsum("nkd,njd->nkj", V_all, V_all)  # [N, k, k]
        vh = torch.einsum("nkd,nd->nk", V_all, flat)  # [N, k]
        coeffs = torch.linalg.pinv(gram, rcond=1e-6) @ vh.unsqueeze(-1)
        return torch.einsum("nkd,nk->nd", V_all, coeffs.squeeze(-1))

    def _edit(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Ablate h of shape [batch, n_pos, d] in fp32; returns same dtype.

        Vectorized: selection, gather, batched QR, and span projection run
        for all (batch, position) pairs at once. Semantics identical to the
        original per-position loop (verified by the parity unit tests).
        """
        if self.cfg.mode == "none":
            return h
        cfg, lens = self.cfg, self.lens
        orig_dtype = h.dtype
        hf = h.detach().float()  # ablation is never backpropagated through
        B, P, d = hf.shape
        flat = hf.reshape(B * P, d)  # [N, d]
        N = flat.shape[0]

        # ---- select the k vectors to remove, per position ----
        fixed_mode = (cfg.fixed_token_ids is not None
                      or cfg.mode == "random_tokens")
        if fixed_mode:
            if cfg.fixed_token_ids is not None:
                token_ids = list(dict.fromkeys(cfg.fixed_token_ids))
            else:  # random tokens, fixed per layer by seed (deduplicated)
                gen = torch.Generator().manual_seed(cfg.seed * 7919 + layer)
                token_ids = list(dict.fromkeys(torch.randint(
                    0, lens.adapter.vocab_size, (cfg.k,),
                    generator=gen).tolist()))
            V = self._fixed_vectors(layer, token_ids, hf.device)  # [k', d]
            V_all = V.unsqueeze(0).expand(N, -1, -1)  # [N, k', d]
            if cfg.spare and self._spare_ids is not None:
                # spare-match the control arms: zero the rows whose token is
                # in that position's spare set (harmless under Gram-pinv)
                spare = self._spare_ids
                if spare.shape[1] != P:
                    spare = spare[:, -P:]
                spare_flat = spare.reshape(N, -1).to(hf.device)  # [N, n_spare]
                ids_t = torch.tensor(token_ids, device=hf.device)
                keep = ~(spare_flat.unsqueeze(1) == ids_t.view(1, -1, 1)
                         ).any(-1)  # [N, k']
                V_all = V_all * keep.unsqueeze(-1)
        else:
            strengths = lens.strengths(hf, layer)  # [B, P, vocab]
            magnitudes = strengths.abs() if cfg.select == "abs" else strengths
            if cfg.spare and self._spare_ids is not None:
                spare = self._spare_ids
                if spare.shape[1] != P:  # prefill sees full seq; decode sees 1
                    spare = spare[:, -P:]
                magnitudes = magnitudes.scatter(
                    -1, spare.to(magnitudes.device),
                    torch.finfo(magnitudes.dtype).min)
            top_ids = magnitudes.topk(cfg.k, dim=-1).indices  # [B, P, k]
            flat_ids = top_ids.reshape(N, cfg.k)  # [N, k]
            pos_mask = torch.ones_like(flat_ids, dtype=torch.bool)
            if cfg.select == "positive":
                # never "remove" a non-active concept: zero the V rows of
                # negative-strength picks (exact no-op under Gram-pinv)
                sel = strengths.reshape(N, -1).gather(-1, flat_ids)  # [N, k]
                pos_mask = sel > 0
            # gather lens vectors ((W_U*gamma)[ids] @ J, batched; kept inline
            # rather than LensSpace.vectors because ids are [N, k] not [n])
            wu = lens._wu_gamma(hf.device, torch.float32)
            J = lens.jacobians[layer].to(device=hf.device, dtype=torch.float32)
            V_all = (wu[flat_ids] @ J) * pos_mask.unsqueeze(-1)  # [N, k, d]

        # ---- position skipping (absolute index < skip_first_positions) ----
        abs_pos = (self._position_offset
                   + torch.arange(P, device=hf.device)).repeat_interleave(1
                   ).unsqueeze(0).expand(B, -1).reshape(N)  # [N]
        active = abs_pos >= cfg.skip_first_positions
        if not fixed_mode and cfg.select == "positive":
            active = active & pos_mask.any(dim=-1)

        # ---- removal delta ----
        if cfg.projection == "span":
            delta = self._span_project(flat, V_all)
        else:  # pervector one-shot on unit-normed vectors
            Vh = V_all / V_all.norm(dim=2, keepdim=True).clamp_min(1e-8)
            delta = torch.einsum(
                "nk,nkd->nd", torch.einsum("nd,nkd->nk", flat, Vh), Vh)
        delta = delta * active.unsqueeze(-1)

        removed = delta.norm(dim=-1)  # [N]
        if cfg.mode == "random":  # remove matched norm inside a random span
            Qr = self._random_basis(layer, d, hf.device)  # [k, d]
            proj = flat @ Qr.T  # [N, k]
            direction = proj @ Qr  # [N, d]
            dnorm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            new_flat = flat - (cfg.alpha * removed).unsqueeze(-1) * direction / dnorm
        else:
            new_flat = flat - cfg.alpha * delta
        if cfg.renorm:
            orig_norm = flat.norm(dim=-1, keepdim=True)
            new_norm = new_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scaled = new_flat * orig_norm / new_norm
            # only rescale positions that were actually edited
            new_flat = torch.where(active.unsqueeze(-1), scaled, new_flat)

        # ---- stats: only positions actually edited; dose = actual net edit ----
        net = (flat - new_flat).norm(dim=-1)  # equals alpha*removed unless renorm
        n_active = int(active.sum())
        self.stats.removed_norm_sum += float(net[active].sum())
        self.stats.resid_norm_sum += float(flat.norm(dim=-1)[active].sum())
        self.stats.removed_norm_count += n_active
        self.stats.positions_ablated += n_active
        if cfg.record_selected and cfg.mode != "random":
            if fixed_mode:
                self.stats.selected_counter.update(
                    {t: n_active for t in token_ids})
            else:
                genuine = pos_mask & active.unsqueeze(-1)
                self.stats.selected_counter.update(
                    flat_ids[genuine].flatten().tolist())
        return new_flat.reshape(B, P, d).to(orig_dtype)

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
    ablator.set_position_offset(0)
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
        ablator.set_position_offset(input_ids.shape[1] + len(tokens) - 1)
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
