"""M2a — the internal-external trade-off, lesion-differential form (H4).

GSM8K x {direct, cot} x {clean, jspace, randtok} at full-band k=10, honest
semantics. H4: relative drop under J-ablation is >=1.5x larger in direct
mode than CoT mode; asymmetry absent in randtok arm. The pre-registered
test statistic is the bootstrap distribution of
    asym = rel_drop(direct) - 1.5 * rel_drop(cot)
(clustered by problem, jointly resampled across all four cells): H4 is
supported when its CI95 lies above 0.

    python -u scripts/m2a.py run --lens LENS [--n 150] [--smoke]
    python -u scripts/m2a.py analyze
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "m2a")
FULL = list(range(12, 29))
ARMS = ["clean", "jspace", "randtok"]
CAPS = {"direct": 64, "cot": 768}


def run(args) -> None:
    from scripts.gates import load_everything
    from src.eval_harness import load_problems, run_cell, summarize_cell

    model, tok, lens = load_everything(args.lens, args.model)
    lens_id = os.path.basename(args.lens)
    n = 6 if args.smoke else args.n
    problems = load_problems("gsm8k", n=n, seed=7)
    for mode in ("direct", "cot"):
        for arm in ARMS:
            tag = "smoke_" if args.smoke else ""
            out = os.path.join(RESULTS, f"{tag}gsm8k_{mode}_{arm}.jsonl")
            print(f"M2a {mode}/{arm} -> {out}", flush=True)
            run_cell(model, tok, lens, problems, answer_mode=mode, arm=arm,
                     band=FULL, k=10, max_new_tokens=CAPS[mode],
                     out_path=out, lens_id=lens_id)
            s = summarize_cell(out)
            print(f"  acc={s['accuracy']:.3f} trunc={s['truncation_rate']:.2f} "
                  f"fmt_break={s['format_break_rate']:.2f} "
                  f"len={s['mean_new_tokens']:.0f}", flush=True)


def _cells(prefix: str = "") -> dict:
    from src.eval_harness import _read_rows_lenient

    cells = {}
    for mode in ("direct", "cot"):
        for arm in ["clean"] + [a for a in ARMS if a != "clean"]:
            path = os.path.join(RESULTS, f"{prefix}gsm8k_{mode}_{arm}.jsonl")
            if os.path.exists(path):
                cells[(mode, arm)] = [
                    r for r in _read_rows_lenient(path) if r.get("ok")]
    return cells


def analyze(args) -> None:
    import numpy as np

    from src.eval_harness import summarize_cell

    prefix = "smoke_" if args.smoke else ""
    cells = _cells(prefix)
    if len(cells) < 6:
        print(f"missing cells (have {sorted(cells)}); run M2a first")
        return

    # per-cell descriptives incl. the frozen confound covariates (H4 report
    # must show truncation/format effects next to every drop)
    for (mode, arm), rows in sorted(cells.items()):
        path = os.path.join(RESULTS, f"{prefix}gsm8k_{mode}_{arm}.jsonl")
        s = summarize_cell(path)
        print(f"{mode:>7}/{arm:<8} acc={s['accuracy']:.3f} "
              f"trunc={s['truncation_rate']:.2f} "
              f"fmt_break={s['format_break_rate']:.2f} "
              f"len={s['mean_new_tokens']:.0f}")

    def acc_map(rows):
        return {(r["problem_id"], r["sample_idx"]): r["correct"] for r in rows}

    problems = sorted({r["problem_id"] for r in cells[("direct", "clean")]})
    maps = {key: acc_map(rows) for key, rows in cells.items()}

    def rel_drop(mode, arm, prob_subset) -> float | None:
        clean, abl = maps[(mode, "clean")], maps[(mode, arm)]
        keys = [k for k in clean if k in abl and k[0] in prob_subset]
        if not keys:
            return None
        a = np.mean([clean[k] for k in keys])
        b = np.mean([abl[k] for k in keys])
        if a == 0:
            return None  # undefined: no clean successes to lose
        return (a - b) / a

    rng = np.random.default_rng(0)
    print()
    for arm in ("jspace", "randtok"):
        point = {m: rel_drop(m, arm, set(problems)) for m in ("direct", "cot")}
        if None in point.values():
            print(f"{arm}: rel_drop undefined (clean accuracy 0 in a mode)")
            continue
        asyms = []
        for _ in range(2000):
            sub = set(rng.choice(problems, len(problems)))
            d = rel_drop("direct", arm, sub)
            c = rel_drop("cot", arm, sub)
            if d is not None and c is not None:
                asyms.append(d - 1.5 * c)
        lo, hi = np.percentile(asyms, [2.5, 97.5])
        verdict = ("H4-SUPPORTED" if lo > 0 else
                   "H4-REJECTED (CI below 0)" if hi < 0 else "inconclusive")
        print(f"{arm}: rel_drop direct={point['direct']:+.3f} "
              f"cot={point['cot']:+.3f}  "
              f"asym(d-1.5c)={point['direct'] - 1.5 * point['cot']:+.3f} "
              f"CI95=[{lo:+.3f},{hi:+.3f}] -> {verdict}"
              + ("" if arm == "jspace" else "  (expect ~0 for fair control)"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["run", "analyze"])
    parser.add_argument("--lens")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--smoke", action="store_true",
                        help="6 problems only, smoke_ file prefix")
    args = parser.parse_args()
    if args.cmd == "run" and not args.lens:
        parser.error("run requires --lens")
    {"run": run, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
