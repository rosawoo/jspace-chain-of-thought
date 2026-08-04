"""Evaluation harness for the core experiments C1-C4.

Cells are (dataset, answer_mode, arm) triples. Every generation is logged as
one JSONL row with its transcript, so runs are resumable and every analysis
is recomputable offline. Sampling params follow the Qwen3 model card
(never greedy); seeds are paired across arms: seed = hash(problem_id, sample_idx),
independent of arm, so arm contrasts are within-problem-within-seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass

import torch

from .ablate import AblationConfig, JSpaceAblator, generate_with_ablation
from .jlens_core import LensSpace

THINK_END_ID = 151668  # </think> for Qwen3 tokenizers

SAMPLING = {
    # model-card params; "thinking" also used for its warning: never greedy
    "thinking": dict(temperature=0.6, top_p=0.95, top_k=20),
    "non_thinking": dict(temperature=0.7, top_p=0.8, top_k=20),
}


@dataclass
class Problem:
    problem_id: str
    question: str
    gold_answer: str
    dataset: str
    level: int | None = None  # MATH-500 difficulty stratum


def load_problems(dataset: str, n: int | None = None, seed: int = 0
                  ) -> list[Problem]:
    from datasets import load_dataset

    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        probs = [Problem(f"gsm8k-{i}", r["question"],
                         r["answer"].split("####")[-1].strip().replace(",", ""),
                         "gsm8k")
                 for i, r in enumerate(ds)]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        probs = [Problem(f"math500-{i}", r["problem"], r["answer"], "math500",
                         level=int(r["level"]))
                 for i, r in enumerate(ds)]
    elif dataset == "aime":
        rows = []
        d24 = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        rows += [(f"aime24-{i}", r["Problem"], str(r["Answer"]))
                 for i, r in enumerate(d24)]
        d25 = load_dataset("yentinglin/aime_2025", "default", split="train")
        rows += [(f"aime25-{i}", r["problem"], str(r["answer"]))
                 for i, r in enumerate(d25)]
        probs = [Problem(pid, q, a, "aime") for pid, q, a in rows]
    else:
        raise ValueError(dataset)

    if n is not None and n < len(probs):
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(probs), generator=gen)[:n].tolist()
        probs = [probs[i] for i in sorted(idx)]
    return probs


def stratified_math500(per_level: int = 60, seed: int = 0) -> list[Problem]:
    probs = load_problems("math500")
    out = []
    gen = torch.Generator().manual_seed(seed)
    for level in (1, 2, 3, 4, 5):
        pool = [p for p in probs if p.level == level]
        idx = torch.randperm(len(pool), generator=gen)[:per_level].tolist()
        out += [pool[i] for i in sorted(idx)]
    return out


def build_prompt(tok, problem: Problem, answer_mode: str) -> torch.Tensor:
    """answer_mode: 'direct' | 'cot' (instructed, non-thinking) | 'thinking'."""
    if answer_mode == "direct":
        content = (problem.question +
                   "\n\nGive only the final answer, inside \\boxed{}. "
                   "Do not show any working.")
        thinking = False
    elif answer_mode == "cot":
        content = (problem.question +
                   "\n\nThink step by step, then give the final answer "
                   "inside \\boxed{}.")
        thinking = False
    elif answer_mode == "thinking":
        content = (problem.question +
                   "\n\nPut your final answer inside \\boxed{}.")
        thinking = True
    else:
        raise ValueError(answer_mode)
    text = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=thinking)
    return tok(text, return_tensors="pt").input_ids


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} after </think> if present (balanced braces)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    starts = [m.end() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None
    start = starts[-1]
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def score_answer(pred: str | None, problem: Problem) -> bool:
    if pred is None:
        return False
    if problem.dataset == "aime":
        digits = re.sub(r"[^\d]", "", pred)
        return digits != "" and int(digits) == int(problem.gold_answer)
    try:
        from math_verify import parse, verify
        return bool(verify(parse(problem.gold_answer), parse(pred)))
    except Exception:
        return pred.strip().replace(",", "") == problem.gold_answer.strip()


def paired_seed(problem_id: str, sample_idx: int) -> int:
    h = hashlib.sha256(f"{problem_id}/{sample_idx}".encode()).hexdigest()
    return int(h[:8], 16)


ARMS = {
    "clean": dict(mode="none", spare=True),
    "jspace": dict(mode="jspace", spare=True),
    "random": dict(mode="random", spare=True),
    "jspace-nospare": dict(mode="jspace", spare=False),
}


@torch.no_grad()
def run_cell(model, tok, lens: LensSpace, problems: list[Problem], *,
             answer_mode: str, arm: str, band: list[int], k: int,
             max_new_tokens: int, samples_per_problem: int = 1,
             out_path: str, ablation_seed: int = 0) -> None:
    """Run one experiment cell, appending JSONL rows; resumable."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {(r["problem_id"], r["sample_idx"])
                    for r in map(json.loads, f) if r.get("ok")}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sampling = SAMPLING["thinking" if answer_mode == "thinking"
                        else "non_thinking"]

    with open(out_path, "a") as f:
        for problem in problems:
            for s in range(samples_per_problem):
                if (problem.problem_id, s) in done:
                    continue
                ids = build_prompt(tok, problem, answer_mode).to(model.device)
                ablator = JSpaceAblator(lens, AblationConfig(
                    band_layers=band, k=k, seed=ablation_seed,
                    **ARMS[arm]))
                out = generate_with_ablation(
                    model, ids, ablator,
                    max_new_tokens=max_new_tokens,
                    seed=paired_seed(problem.problem_id, s),
                    eos_token_ids=[tok.eos_token_id],
                    **sampling)
                text = tok.decode(out["sequence"])
                pred = extract_boxed(text)
                row = {
                    "problem_id": problem.problem_id,
                    "sample_idx": s,
                    "dataset": problem.dataset,
                    "level": problem.level,
                    "answer_mode": answer_mode,
                    "arm": arm,
                    "band": band, "k": k,
                    "pred": pred,
                    "correct": score_answer(pred, problem),
                    "truncated": not out["hit_eos"]
                                 and out["n_new_tokens"] >= max_new_tokens,
                    "format_break": pred is None,
                    "n_new_tokens": out["n_new_tokens"],
                    "ablation_stats": out["stats"],
                    "text": text,
                    "ok": True,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict],
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the accuracy difference (a - b), clustered by problem_id.
    Rows must cover the same problems (paired design)."""
    import numpy as np

    by_problem: dict[str, list[tuple[bool, bool]]] = {}
    b_index = {(r["problem_id"], r["sample_idx"]): r["correct"] for r in rows_b}
    for r in rows_a:
        key = (r["problem_id"], r["sample_idx"])
        if key in b_index:
            by_problem.setdefault(r["problem_id"], []).append(
                (r["correct"], b_index[key]))
    problems = list(by_problem)
    if not problems:
        return {"n": 0}
    per_prob = np.array([
        [np.mean([a for a, _ in v]), np.mean([b for _, b in v])]
        for v in by_problem.values()])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(problems), len(problems))
        sample = per_prob[idx]
        diffs.append(sample[:, 0].mean() - sample[:, 1].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": len(problems),
            "acc_a": float(per_prob[:, 0].mean()),
            "acc_b": float(per_prob[:, 1].mean()),
            "diff": float(per_prob[:, 0].mean() - per_prob[:, 1].mean()),
            "ci95": [float(lo), float(hi)]}


def summarize_cell(path: str) -> dict:
    rows = [json.loads(line) for line in open(path)]
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"n": 0}
    frac = lambda key: sum(r[key] for r in ok) / len(ok)
    return {
        "n": len(ok),
        "accuracy": frac("correct"),
        "truncation_rate": frac("truncated"),
        "format_break_rate": frac("format_break"),
        "mean_new_tokens": sum(r["n_new_tokens"] for r in ok) / len(ok),
        "mean_removed_norm": (
            sum(r["ablation_stats"].get("mean_removed_norm", 0) for r in ok)
            / len(ok)),
    }
