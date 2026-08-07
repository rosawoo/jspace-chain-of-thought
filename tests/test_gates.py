"""Shape/sanity smoke tests for Phase-0 measurements on the tiny model."""

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from src.jlens_core import LensSpace, ModelAdapter
from src.gates import (band_metrics, numeric_loading_audit, occupancy_curve,
                        pretraining_top1_match, token_is_numeric)

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

    class FakeTok:
        """Minimal tokenizer over a 97-token vocab for smoke tests."""
        def __call__(self, text, return_tensors=None, truncation=True,
                     max_length=64):
            ids = [(ord(c) * 7) % VOCAB for c in text[:max_length]]
            import types
            return types.SimpleNamespace(
                input_ids=torch.tensor([ids or [1]]))

        def encode(self, text, add_special_tokens=False):
            return [(ord(c) * 7) % VOCAB for c in text][:2]

        def decode(self, ids):
            return "7" if ids[0] % 3 == 0 else "x"

    return model, lens, FakeTok()


def test_band_metrics_shapes(tiny):
    model, lens, tok = tiny
    out = band_metrics(model, tok, lens, ["hello world this is a test"], BAND)
    assert set(out) == set(BAND)
    for l in BAND:
        assert {"kurtosis", "next_token_agree", "autocorr_excess"} <= set(out[l])


def test_occupancy_runs(tiny):
    model, lens, tok = tiny
    prompts = ["a reasonably long test passage " * 4]
    out = occupancy_curve(model, tok, lens, prompts, [BAND[0]], k_max=5,
                          positions_per_prompt=3)
    assert 1 <= out[BAND[0]]["occupancy"] <= 5
    assert len(out[BAND[0]]["gain_j"]) == 5


def test_numeric_audit_and_tokencheck(tiny):
    model, lens, tok = tiny
    out = numeric_loading_audit(model, tok, lens, BAND,
                                ["what is 17 times 23"], k=3)
    assert 0.0 <= out["numeric_fraction"] <= 1.0
    real_tok = None  # token_is_numeric with the fake tokenizer:
    assert token_is_numeric(tok, 0) in (True, False)


def test_pretraining_top1_match_bounds(tiny):
    model, lens, tok = tiny
    m_none = pretraining_top1_match(model, tok, lens, BAND,
                                    ["another test passage " * 3], k=3,
                                    mode="jspace", spare=False)
    assert 0.0 <= m_none <= 1.0
