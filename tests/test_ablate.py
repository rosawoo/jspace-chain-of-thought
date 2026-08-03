"""Unit tests for the J-space ablation engine (Mac/CPU, tiny random Qwen3).

These must pass before any cloud spend (plan: Verification section).
"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.ablate import AblationConfig, JSpaceAblator, generate_with_ablation
from src.jlens_core import LensSpace, ModelAdapter

VOCAB, D, LAYERS = 97, 64, 4
BAND = [1, 2]


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=D, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=128, max_position_embeddings=256,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    adapter = ModelAdapter.from_hf(model)
    jac = {l: torch.randn(D, D) / D**0.5 + torch.eye(D) for l in BAND}
    lens = LensSpace(adapter, jac)
    return model, adapter, lens


def make_ablator(lens, **kw):
    defaults = dict(band_layers=BAND, k=5, mode="jspace", spare=False, seed=0)
    defaults.update(kw)
    return JSpaceAblator(lens, AblationConfig(**defaults))


def test_topk_selection_matches_bruteforce(tiny):
    _, adapter, lens = tiny
    h = torch.randn(1, 3, D)
    layer = BAND[0]
    # brute force: materialize all lens vectors, unit-norm projections
    wu = adapter.unembed_weight * adapter.final_norm_weight
    V = wu @ lens.jacobians[layer]  # [vocab, d]
    strengths_bf = (h @ V.T) / V.norm(dim=1).clamp_min(1e-8)
    got = lens.strengths(h, layer)
    assert torch.allclose(got, strengths_bf, atol=1e-4)
    top_bf = strengths_bf.abs().topk(5, dim=-1).indices
    top_got = got.abs().topk(5, dim=-1).indices
    assert torch.equal(top_bf, top_got)


def test_projection_exactly_zeroed(tiny):
    _, _, lens = tiny
    ablator = make_ablator(lens, record_selected=True)
    h = torch.randn(2, 4, D)
    layer = BAND[0]
    selected_before = lens.strengths(h, layer).abs().topk(5, dim=-1).indices
    out = ablator._edit(h, layer)
    strengths_after = lens.strengths(out, layer)
    picked = strengths_after.gather(-1, selected_before)
    assert picked.abs().max() < 1e-3, "selected projections must be zeroed"


def test_removed_norm_accounting(tiny):
    _, _, lens = tiny
    for mode in ("jspace", "random"):
        ablator = make_ablator(lens, mode=mode)
        h = torch.randn(1, 5, D)
        out = ablator._edit(h.clone(), BAND[1])
        actual = (h - out).norm(dim=-1).sum().item()
        recorded = ablator.stats.removed_norm_sum
        assert actual == pytest.approx(recorded, rel=1e-3), mode


def test_random_arm_norm_matched_but_different_direction(tiny):
    _, _, lens = tiny
    h = torch.randn(1, 4, D)
    layer = BAND[0]
    ab_j = make_ablator(lens, mode="jspace")
    ab_r = make_ablator(lens, mode="random")
    out_j = ab_j._edit(h.clone(), layer)
    out_r = ab_r._edit(h.clone(), layer)
    # removed norms match per position
    dj = (h - out_j).norm(dim=-1)
    dr = (h - out_r).norm(dim=-1)
    assert torch.allclose(dj, dr, rtol=1e-3)
    # but the perturbation directions differ substantially
    cos = torch.nn.functional.cosine_similarity(
        (h - out_j).flatten(0, 1), (h - out_r).flatten(0, 1), dim=-1)
    assert cos.abs().max() < 0.9


def test_sparing_excludes_clean_top_tokens(tiny):
    _, _, lens = tiny
    # sparing is per-position (paper: clean top-10 at each position), so test
    # a single position where the spared/selected distinction is exact
    ablator = make_ablator(lens, spare=True, record_selected=True, k=5)
    h = torch.randn(1, 1, D)
    layer = BAND[0]
    # construct spare set = the tokens that WOULD be selected without sparing
    would_select = lens.strengths(h, layer).abs().topk(5, dim=-1).indices
    ablator.set_spare_ids(would_select)
    ablator._edit(h, layer)
    selected = set(ablator.stats.selected_counter)
    spared = set(would_select.flatten().tolist())
    assert len(selected) == 5
    assert selected.isdisjoint(spared), "spared ids must never be ablated"


def test_mode_none_is_identity_generation(tiny):
    model, _, lens = tiny
    prompt = torch.randint(0, VOCAB, (1, 8))
    ab = make_ablator(lens, mode="none")
    out = generate_with_ablation(
        model, prompt, ab, max_new_tokens=6, temperature=0.0, top_p=1.0)
    ref = model.generate(prompt, max_new_tokens=6, do_sample=False,
                         pad_token_id=0)
    assert out["sequence"] == ref[0, 8:].tolist()


def test_kv_cache_parity_with_full_recompute(tiny):
    """Cached step-by-step ablated generation must equal ablating the full
    sequence in one forward pass (the critical correctness property)."""
    model, _, lens = tiny
    prompt = torch.randint(0, VOCAB, (1, 8))
    ab = make_ablator(lens, mode="jspace", spare=False, k=5)
    out = generate_with_ablation(
        model, prompt, ab, max_new_tokens=5, temperature=0.0, top_p=1.0)
    seq = torch.cat([prompt, torch.tensor([out["sequence"]])], dim=1)

    ab2 = make_ablator(lens, mode="jspace", spare=False, k=5)
    ab2.set_spare_ids(None)
    with ab2.hooks():
        full = model(input_ids=seq).logits
    # each generated token must be the argmax of the full-recompute logits
    for i, tok in enumerate(out["sequence"]):
        pos = prompt.shape[1] + i - 1
        assert int(full[0, pos].argmax()) == tok, f"divergence at step {i}"


def test_paired_seeds_reproduce(tiny):
    model, _, lens = tiny
    prompt = torch.randint(0, VOCAB, (1, 8))
    outs = []
    for _ in range(2):
        ab = make_ablator(lens, mode="jspace")
        outs.append(generate_with_ablation(
            model, prompt, ab, max_new_tokens=8,
            temperature=0.7, top_p=0.8, seed=123)["sequence"])
    assert outs[0] == outs[1]
