"""Gate-D EXTENSION (documented exploratory pass; DECISION.md 2026-08-03).

Staged re-approach after the frozen sweep failed on coherence everywhere:

STAGE 1 — semantics diagnostic (coherence-only, cheap). The frozen sweep ran
with two implementation choices now under suspicion: abs-selection (zeroing
NEGATIVE projections injects content — likely bug vs the paper's "most
strongly activated") and no attention-sink skipping (lens vectors were never
fitted at positions < 16). Measure coherence at the full-band k=10 reference
under {select: abs|positive} x {skip_first_positions: 0|16}.
(The 'pervector' removal variant was dropped pre-GPU: unit tests proved it
OVERSHOOTS the exact span projection under correlated vectors.)

STAGE 2 — gentle-dose sweep with corrected semantics (open question #2):
k in {1,2,3} on the full band; k in {3,10} on thin 4-layer windows.
C1 unlocks ONLY at the ORIGINAL frozen thresholds (top1>=0.85, wf>=0.90,
two-hop rel drop>=0.50).

STAGE 3 — J-vs-random matched-norm contrast (open question #1) at the
reference point and at the best stage-2 point: is the damage J-specific?

    python scripts/gate_d_ext.py --lens results/lens_qwen3-4b_n148_repaired.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "phase0")
FULL = list(range(12, 29))
WINDOWS = {
    "win12-15": list(range(12, 16)), "win16-19": list(range(16, 20)),
    "win20-23": list(range(20, 24)), "win24-27": list(range(24, 28)),
}
CORRECTED = dict(select="positive", skip_first_positions=16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    from scripts.phase0 import (held_out_passages, load_everything,
                                twohop_generation)
    from src.phase0 import fetch_official_eval, pretraining_top1_match

    model, tok, lens = load_everything(args.lens, args.model)
    passages = held_out_passages(20)
    items = fetch_official_eval("multihop")["items"][:30]
    out = {"stage1": [], "stage2": [], "stage3": []}

    print("== STAGE 1: semantics diagnostic (full band, k=10, coherence) ==")
    for select in ("abs", "positive"):
        for skip in (0, 16):
            top1 = pretraining_top1_match(
                model, tok, lens, FULL, passages, k=10,
                select=select, skip_first_positions=skip)
            row = {"select": select, "skip_first": skip, "top1": top1}
            out["stage1"].append(row)
            print(f"select={select:>8} skip={skip:>2}: top1={top1:.3f}")

    print("\n== STAGE 2: gentle doses, corrected semantics ==")
    configs = ([(f"full-k{k}", FULL, k) for k in (1, 2, 3)] +
               [(f"{n}-k{k}", b, k) for n, b in WINDOWS.items()
                for k in (3, 10)])
    best = None
    for name, band, k in configs:
        top1 = pretraining_top1_match(model, tok, lens, band, passages, k=k,
                                      **CORRECTED)
        clean, abl, wf = twohop_generation(model, tok, lens, band, k, items,
                                           **CORRECTED)
        rel = 1 - abl / max(1e-9, clean)
        viable = top1 >= 0.85 and wf >= 0.90 and rel >= 0.50
        row = {"config": name, "band": band, "k": k, "top1": top1,
               "twohop_clean": clean, "twohop_ablated": abl,
               "twohop_rel_drop": rel, "wellformed": wf,
               "viable_at_frozen_thresholds": viable}
        out["stage2"].append(row)
        if viable and (best is None or rel > best["twohop_rel_drop"]):
            best = row
        print(f"{name:>12}: top1={top1:.3f} 2hop {clean:.2f}->{abl:.2f} "
              f"(rel {rel:.2f}) wf={wf:.2f} "
              f"{'** VIABLE **' if viable else ''}")

    print("\n== STAGE 3: J-vs-random matched-norm contrast ==")
    points = [("reference full-k10", FULL, 10)]
    if best:
        points.append((f"best viable {best['config']}", best["band"], best["k"]))
    for label, band, k in points:
        row = {"point": label}
        for mode in ("jspace", "random"):
            row[f"top1_{mode}"] = pretraining_top1_match(
                model, tok, lens, band, passages, k=k, mode=mode, **CORRECTED)
        gen = twohop_generation(model, tok, lens, band, k, items,
                                arms=("none", "jspace", "random"), **CORRECTED)
        row["twohop"] = {m: {"acc": a, "wf": w} for m, (a, w) in gen.items()}
        out["stage3"].append(row)
        print(f"{label}: top1 J={row['top1_jspace']:.3f} "
              f"rand={row['top1_random']:.3f} | 2hop "
              + " ".join(f"{m}={gen[m][0]:.2f}" for m in gen))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "gate_d_ext.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote gate_d_ext.json")


if __name__ == "__main__":
    main()
