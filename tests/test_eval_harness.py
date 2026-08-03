"""Tests for scoring, extraction, pairing, and the cell runner (tiny model)."""

import json

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.eval_harness import (Problem, extract_boxed, paired_bootstrap,
                              paired_seed, score_answer, summarize_cell)


def test_extract_boxed_variants():
    assert extract_boxed(r"the answer is \boxed{42}") == "42"
    assert extract_boxed(r"\boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"<think>\boxed{9}</think> final \boxed{7}") == "7"
    assert extract_boxed(r"<think>\boxed{9}</think> no box after") is None
    assert extract_boxed("nothing here") is None


def test_score_answer_aime_and_math():
    aime = Problem("a", "q", "042", "aime")
    assert score_answer("42", aime)
    assert score_answer(r"042", aime)
    assert not score_answer("41", aime)
    math = Problem("m", "q", "1/2", "math500")
    assert score_answer("0.5", math) or score_answer(r"\frac{1}{2}", math)
    gsm = Problem("g", "q", "1300", "gsm8k")
    assert score_answer("1,300", gsm)


def test_paired_seed_is_arm_independent_and_stable():
    assert paired_seed("gsm8k-3", 0) == paired_seed("gsm8k-3", 0)
    assert paired_seed("gsm8k-3", 0) != paired_seed("gsm8k-3", 1)
    assert paired_seed("gsm8k-3", 0) != paired_seed("gsm8k-4", 0)


def test_paired_bootstrap_direction():
    rows_a = [{"problem_id": f"p{i}", "sample_idx": 0, "correct": True}
              for i in range(40)]
    rows_b = [{"problem_id": f"p{i}", "sample_idx": 0, "correct": i < 10}
              for i in range(40)]
    out = paired_bootstrap(rows_a, rows_b, n_boot=500)
    assert out["n"] == 40
    assert out["diff"] == pytest.approx(0.75)
    assert out["ci95"][0] > 0  # significantly positive


def test_run_cell_roundtrip(tmp_path):
    from src.ablate import AblationConfig
    from src.eval_harness import run_cell
    from src.jlens_core import LensSpace, ModelAdapter

    torch.manual_seed(0)
    VOCAB, D = 97, 64
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=D, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, intermediate_size=128,
                      max_position_embeddings=256, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).eval()
    adapter = ModelAdapter.from_hf(model)
    lens = LensSpace(adapter, {1: torch.eye(D), 2: torch.eye(D)})

    class FakeTok:
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False):
            return messages[0]["content"]

        def __call__(self, text, return_tensors="pt"):
            import types
            ids = [(ord(c) * 7) % 96 + 1 for c in text[:32]]
            return types.SimpleNamespace(input_ids=torch.tensor([ids]))

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    problems = [Problem(f"t-{i}", f"question {i}", "42", "gsm8k")
                for i in range(2)]
    out = tmp_path / "cell.jsonl"
    run_cell(model, FakeTok(), lens, problems, answer_mode="direct",
             arm="jspace", band=[1, 2], k=3, max_new_tokens=5,
             out_path=str(out))
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 2
    assert all(r["ok"] and r["arm"] == "jspace" for r in rows)
    # resumability: second call adds nothing
    run_cell(model, FakeTok(), lens, problems, answer_mode="direct",
             arm="jspace", band=[1, 2], k=3, max_new_tokens=5,
             out_path=str(out))
    assert len(open(out).read().splitlines()) == 2
    s = summarize_cell(str(out))
    assert s["n"] == 2 and 0 <= s["accuracy"] <= 1
