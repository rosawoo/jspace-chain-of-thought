"""Phase-0 gate measurements (see README table and HYPOTHESES.md).

Gate A: workspace-band statistics (readout kurtosis, next-token agreement,
        top-1 autocorrelation vs shuffled null) + occupancy-vs-depth curve.
Gate B: two-hop positive control on the paper's multihop eval set.
Gate C: numeric-loading audit (share of would-be-ablated slots that are
        numeric during direct math).
Gate D: operating-point sweep is driven by scripts/gates.py using the
        coherence battery here plus generation via src.ablate.

All functions take an already-loaded HF model + tokenizer + LensSpace.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request

import torch

from .jlens_core import LensSpace, record_residuals

OFFICIAL_RAW = "https://raw.githubusercontent.com/anthropics/jacobian-lens/main/data"

NUMBER_WORDS = set(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand".split())


def fetch_official_eval(name: str, data_dir: str = "data") -> dict:
    """Download one of the paper's eval sets from the official repo (cached)."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"lens-eval-{name}.json")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            f"{OFFICIAL_RAW}/evaluations/lens-eval-{name}.json", path)
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def readout_logits(model, tok, lens: LensSpace, prompt: str,
                   layers: list[int], max_len: int = 512,
                   device=None) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Lens logits per layer at every position, plus model final logits."""
    ids = tok(prompt.rstrip(), return_tensors="pt",
              truncation=True, max_length=max_len).input_ids
    ids = ids.to(device or model.device)
    with record_residuals(lens.adapter.layers, at=layers) as rec:
        out = model(input_ids=ids)
    lens_logits = {}
    for l in layers:
        h = rec.acts[l][0].float()  # [T, d]
        lens_logits[l] = lens.scores(h, l).cpu()  # [T, vocab]
    return lens_logits, out.logits[0].float().cpu()


@torch.no_grad()
def band_metrics(model, tok, lens: LensSpace, prompts: list[str],
                 layers: list[int], top_k: int = 10) -> dict[int, dict]:
    """Per-layer workspace signatures (paper section 'which layers')."""
    stats = {l: {"kurt": [], "agree": [], "auto": [], "auto0": []}
             for l in layers}
    for p in prompts:
        outs, model_logits = readout_logits(model, tok, lens, p, layers)
        next_tok = model_logits.argmax(dim=-1)
        for l, lg in outs.items():
            z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
            stats[l]["kurt"].append(((z ** 4).mean(-1) - 3).mean().item())
            top = lg.topk(top_k, dim=-1).indices
            stats[l]["agree"].append(
                (top == next_tok[:, None]).any(-1).float().mean().item())
            t1 = lg.argmax(dim=-1)
            stats[l]["auto"].append((t1[:-1] == t1[1:]).float().mean().item())
            perm = t1[torch.randperm(len(t1))]
            stats[l]["auto0"].append((perm[:-1] == perm[1:]).float().mean().item())
    mean = lambda v: sum(v) / len(v)
    return {l: {"kurtosis": mean(s["kurt"]),
                "next_token_agree": mean(s["agree"]),
                "autocorr_excess": mean(s["auto"]) - mean(s["auto0"])}
            for l, s in stats.items()}


@torch.no_grad()
def occupancy_curve(model, tok, lens: LensSpace, prompts: list[str],
                    layers: list[int], k_max: int = 40,
                    positions_per_prompt: int = 16,
                    seed: int = 0) -> dict[int, dict]:
    """Greedy nonneg matching pursuit marginal gains, J-dictionary vs a
    random-direction control; occupancy = first K where the J marginal gain
    falls below the random control's."""
    gen = torch.Generator().manual_seed(seed)
    results = {}
    for l in layers:
        gains_j, gains_r = [], []
        for p in prompts:
            ids = tok(p.rstrip(), return_tensors="pt", truncation=True,
                      max_length=256).input_ids.to(model.device)
            with record_residuals(lens.adapter.layers, at=[l]) as rec:
                model(input_ids=ids)
            h_all = rec.acts[l][0].float()  # [T, d]
            T = h_all.shape[0]
            pos = torch.randperm(T - 16, generator=gen)[:positions_per_prompt] + 16
            for t in pos.tolist():
                h = h_all[min(t, T - 1)]
                gains_j.append(_pursuit_gains(h, lens, l, k_max))
                gains_r.append(_pursuit_gains(h, lens, l, k_max, random_dict=True,
                                              gen=gen))
        gj = torch.tensor(gains_j).mean(0)  # [k_max]
        gr = torch.tensor(gains_r).mean(0)
        below = (gj <= gr).nonzero()
        occ = int(below[0]) + 1 if len(below) else k_max
        results[l] = {"occupancy": occ,
                      "gain_j": gj.tolist(), "gain_random": gr.tolist()}
    return results


def _pursuit_gains(h: torch.Tensor, lens: LensSpace, layer: int, k_max: int,
                   random_dict: bool = False, gen=None) -> list[float]:
    """Marginal squared-error reduction at each greedy nonneg pursuit step."""
    d = h.shape[0]
    residual = h.clone()
    gains = []
    chosen: list[torch.Tensor] = []
    for _ in range(k_max):
        if random_dict:
            cand = torch.randn(256, d, generator=gen).to(h.device)
            cand = cand / cand.norm(dim=1, keepdim=True)
            scores = cand @ residual
            best = scores.argmax()
            v = cand[best] if scores[best] > 0 else -cand[best]
        else:
            strengths = lens.strengths(residual.unsqueeze(0), layer)[0]
            best = strengths.argmax()  # nonneg: positive projections only
            if strengths[best] <= 0:
                gains.append(0.0)
                continue
            v = lens.vectors(layer, best.unsqueeze(0), device=h.device)[0]
            v = v / v.norm().clamp_min(1e-8)
        chosen.append(v)
        before = residual.norm() ** 2
        coeff = torch.dot(v, residual).clamp_min(0)
        residual = residual - coeff * v
        gains.append(float(before - residual.norm() ** 2))
    return gains


@torch.no_grad()
def two_hop_readout(model, tok, lens: LensSpace, band: list[int],
                    n_items: int | None = 20, top_k: int = 10) -> dict:
    """Gate B: does the bridge entity appear in the top-k mid-band readout?"""
    items = fetch_official_eval("multihop")["items"][:n_items]
    hits, details = 0, []
    for item in items:
        prompt = item["prompt"].rstrip()
        outs, _ = readout_logits(model, tok, lens, prompt, band)
        found, best_rank = False, None
        for word in item["intermediates"]:
            ids = _single_token_ids(tok, word)
            if not ids:
                continue
            rank = min(_rank_of(outs[l][-1], ids) for l in band)
            best_rank = rank if best_rank is None else min(best_rank, rank)
            if rank <= top_k:
                found = True
        hits += found
        details.append({"prompt": prompt[:80], "found": found,
                        "best_rank": best_rank})
    return {"hits": hits, "total": len(items), "details": details}


def _single_token_ids(tok, word: str) -> list[int]:
    ids = set()
    for form in {word, word.lower(), word.capitalize(),
                 " " + word, " " + word.lower(), " " + word.capitalize()}:
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


def _rank_of(logits: torch.Tensor, ids: list[int]) -> int:
    best = logits[ids].max()
    return int((logits > best).sum().item()) + 1


def token_is_numeric(tok, token_id: int) -> bool:
    s = tok.decode([token_id]).strip().lower()
    return bool(re.fullmatch(r"[0-9]+([.,][0-9]+)?", s)) or s in NUMBER_WORDS


@torch.no_grad()
def numeric_loading_audit(model, tok, lens: LensSpace, band: list[int],
                          prompts: list[str], k: int = 10,
                          spare_top_n: int = 10) -> dict:
    """Gate C: on direct-math prompts, what fraction of the would-be-ablated
    (position, layer) slots are numeric tokens? Mirrors the ablation's
    selection rule exactly, including sparing."""
    numeric_slots, total_slots = 0, 0
    per_prompt = []
    for p in prompts:
        ids = tok(p.rstrip(), return_tensors="pt", truncation=True,
                  max_length=1024).input_ids.to(model.device)
        with record_residuals(lens.adapter.layers, at=band) as rec:
            out = model(input_ids=ids)
        spare = out.logits[0].topk(spare_top_n, dim=-1).indices  # [T, N]
        n_num, n_tot = 0, 0
        for l in band:
            h = rec.acts[l][0].float()  # [T, d]
            mags = lens.strengths(h, l).abs()
            mags = mags.scatter(-1, spare.to(mags.device), 0.0)
            top = mags.topk(k, dim=-1).indices  # [T, k]
            for tid in top.flatten().tolist():
                n_num += token_is_numeric(tok, tid)
                n_tot += 1
        numeric_slots += n_num
        total_slots += n_tot
        per_prompt.append(n_num / max(1, n_tot))
    return {"numeric_fraction": numeric_slots / max(1, total_slots),
            "per_prompt": per_prompt}


@torch.no_grad()
def pretraining_top1_match(model, tok, lens: LensSpace, band: list[int],
                           passages: list[str], k: int,
                           mode: str = "jspace", spare: bool = True,
                           max_len: int = 512, **ablation_kwargs) -> float:
    """Coherence-gate statistic: fraction of positions where the hooked
    (teacher-forced) model's top-1 next-token prediction matches clean."""
    from .ablate import AblationConfig, JSpaceAblator

    matches, total = 0, 0
    for p in passages:
        ids = tok(p.rstrip(), return_tensors="pt", truncation=True,
                  max_length=max_len).input_ids.to(model.device)
        clean_logits = model(input_ids=ids).logits[0]
        ablator = JSpaceAblator(lens, AblationConfig(
            band_layers=band, k=k, mode=mode, spare=spare, **ablation_kwargs))
        spare_ids = (clean_logits.topk(10, dim=-1).indices.unsqueeze(0)
                     if spare else None)
        ablator.set_spare_ids(spare_ids)
        with ablator.hooks():
            abl_logits = model(input_ids=ids).logits[0]
        matches += int((clean_logits.argmax(-1) == abl_logits.argmax(-1)).sum())
        total += ids.shape[1]
    return matches / max(1, total)
