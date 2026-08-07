"""Core experiment driver (C1-C4). Every cell writes
results/<experiment>/<dataset>_<mode>_<arm>.jsonl and is resumable.

The band and k come from Phase-0 Gate D (results/gates/gate_d.json).

    python scripts/run_core.py c1 --lens LENS --band ... --k 10
    python scripts/run_core.py c2 --lens LENS --band ... --k 10
    python scripts/run_core.py c3 --lens LENS --band ... --k 10
    python scripts/run_core.py c4 --lens LENS --band ... --k 10
    python scripts/run_core.py summarize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# Pre-registered cell definitions (HYPOTHESES.md incl. Amendment 1).
# M1 runs three arms; "jspace-nospare" (H2) is a deferred fourth arm.
C1_ARMS = ["clean", "jspace", "random"]
C2_ARMS = ["clean", "jspace", "random"]
C3_ARMS = ["clean", "jspace", "random"]
CAPS = {"gsm8k_direct": 64, "gsm8k_cot": 1024, "math500_cot": 3072,
        "math500_direct": 64, "aime_thinking": 16384}


def load_everything(args):
    from scripts.gates import load_everything as _le
    return _le(args.lens, args.model)


def c1(args) -> None:
    """GSM8K replication + artifact battery: 200 problems, direct + CoT."""
    from src.eval_harness import load_problems, run_cell

    model, tok, lens = load_everything(args)
    problems = load_problems("gsm8k", n=args.n or 200, seed=7)
    arms = args.arms or C1_ARMS
    for mode, cap_key in (("direct", "gsm8k_direct"), ("cot", "gsm8k_cot")):
        for arm in arms:
            out = os.path.join(RESULTS, "c1", f"gsm8k_{mode}_{arm}.jsonl")
            print(f"C1 {mode}/{arm} -> {out}")
            run_cell(model, tok, lens, problems, answer_mode=mode, arm=arm,
                     band=args.band, k=args.k, max_new_tokens=CAPS[cap_key],
                     out_path=out)


def c2(args) -> None:
    """MATH-500 difficulty gradient: 60/level, CoT (+ direct on L1-2 only)."""
    from src.eval_harness import run_cell, stratified_math500

    model, tok, lens = load_everything(args)
    problems = stratified_math500(per_level=args.n or 60, seed=7)
    for arm in C2_ARMS:
        out = os.path.join(RESULTS, "c2", f"math500_cot_{arm}.jsonl")
        print(f"C2 cot/{arm} -> {out}")
        run_cell(model, tok, lens, problems, answer_mode="cot", arm=arm,
                 band=args.band, k=args.k,
                 max_new_tokens=CAPS["math500_cot"], out_path=out)
    # direct arms only where clean direct accuracy can clear the 30% floor
    easy = [p for p in problems if p.level in (1, 2)]
    for arm in C2_ARMS:
        out = os.path.join(RESULTS, "c2", f"math500_direct_{arm}.jsonl")
        print(f"C2 direct/{arm} (L1-2 only) -> {out}")
        run_cell(model, tok, lens, easy, answer_mode="direct", arm=arm,
                 band=args.band, k=args.k,
                 max_new_tokens=CAPS["math500_direct"], out_path=out)


def c3(args) -> None:
    """AIME anchor: 60 problems, thinking mode, 4 samples/problem, no direct."""
    from src.eval_harness import load_problems, run_cell

    model, tok, lens = load_everything(args)
    problems = load_problems("aime")
    samples = {"clean": 4, "jspace": 4, "random": args.random_samples}
    for arm in C3_ARMS:
        out = os.path.join(RESULTS, "c3", f"aime_thinking_{arm}.jsonl")
        print(f"C3 thinking/{arm} x{samples[arm]} -> {out}")
        run_cell(model, tok, lens, problems, answer_mode="thinking", arm=arm,
                 band=args.band, k=args.k,
                 max_new_tokens=CAPS["aime_thinking"],
                 samples_per_problem=samples[arm], out_path=out)


def c4(args) -> None:
    """Broad-degradation battery per arm: pretraining top-1 match, MMLU
    subset, copy task."""
    import torch
    from datasets import load_dataset

    from scripts.gates import held_out_passages
    from src.ablate import AblationConfig, JSpaceAblator, generate_with_ablation
    from src.eval_harness import ARMS
    from src.gates import pretraining_top1_match

    model, tok, lens = load_everything(args)
    battery: dict = {}

    passages = held_out_passages(50)
    mmlu = load_dataset("cais/mmlu", "all", split="validation").select(range(200))
    copy_text = "The quick brown fox jumps over the lazy dog near the riverbank."

    for arm in ("clean", "jspace", "random"):
        entry = {}
        if arm == "clean":
            entry["pretraining_top1_match"] = 1.0
        else:
            entry["pretraining_top1_match"] = pretraining_top1_match(
                model, tok, lens, args.band, passages, k=args.k,
                mode=ARMS[arm]["mode"], spare=ARMS[arm]["spare"])

        correct = 0
        for i, row in enumerate(mmlu):
            letters = ["A", "B", "C", "D"]
            q = row["question"] + "\n" + "\n".join(
                f"{l}. {c}" for l, c in zip(letters, row["choices"]))
            text = tok.apply_chat_template(
                [{"role": "user",
                  "content": q + "\n\nAnswer with just the letter."}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            ablator = JSpaceAblator(lens, AblationConfig(
                band_layers=args.band, k=args.k, **ARMS[arm]))
            out = generate_with_ablation(
                model, ids, ablator, max_new_tokens=4,
                temperature=0.7, top_p=0.8, seed=i,
                eos_token_ids=[tok.eos_token_id])
            answer = tok.decode(out["sequence"]).strip()[:1].upper()
            correct += answer == letters[row["answer"]]
        entry["mmlu200"] = correct / len(mmlu)

        copied = 0
        for i in range(20):
            text = tok.apply_chat_template(
                [{"role": "user",
                  "content": f"Copy this sentence exactly: {copy_text}"}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            ablator = JSpaceAblator(lens, AblationConfig(
                band_layers=args.band, k=args.k, **ARMS[arm]))
            out = generate_with_ablation(
                model, ids, ablator, max_new_tokens=40,
                temperature=0.7, top_p=0.8, seed=i,
                eos_token_ids=[tok.eos_token_id])
            copied += copy_text in tok.decode(out["sequence"])
        entry["copy_task"] = copied / 20
        battery[arm] = entry
        print(arm, entry)

    os.makedirs(os.path.join(RESULTS, "c4"), exist_ok=True)
    with open(os.path.join(RESULTS, "c4", "battery.json"), "w") as f:
        json.dump(battery, f, indent=1)


def summarize(args) -> None:
    from src.eval_harness import summarize_cell

    for path in sorted(glob.glob(os.path.join(RESULTS, "c*", "*.jsonl"))):
        s = summarize_cell(path)
        rel = os.path.relpath(path, RESULTS)
        print(f"{rel:>45}  n={s['n']:>4}  acc={s.get('accuracy', 0):.3f}  "
              f"trunc={s.get('truncation_rate', 0):.2f}  "
              f"fmt-break={s.get('format_break_rate', 0):.2f}  "
              f"len={s.get('mean_new_tokens', 0):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("c1", "c2", "c3", "c4"):
        s = sub.add_parser(name)
        s.add_argument("--lens", required=True)
        s.add_argument("--model", default="Qwen/Qwen3-4B")
        s.add_argument("--band", type=int, nargs="+", required=True)
        s.add_argument("--k", type=int, default=10)
        s.add_argument("--n", type=int, default=None)
        s.add_argument("--random-samples", type=int, default=4)
        s.add_argument("--arms", nargs="+", default=None,
                       help="override arms, e.g. --arms jspace-nospare "
                            "to add the deferred H2 arm later")
    sub.add_parser("summarize")
    args = parser.parse_args()
    {"c1": c1, "c2": c2, "c3": c3, "c4": c4,
     "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
