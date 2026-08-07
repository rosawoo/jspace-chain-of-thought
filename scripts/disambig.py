"""Disambiguation run (DECISION.md 2026-08-04): does the two-hop reasoning
effect survive HONEST ablation semantics?

The original Gate-D collapse (0.50 -> 0.03 at full-k10) ran with the
abs-selection bug (negative-projection injection) and no sink handling. The
extension's corrected runs added skip_first_positions=16, which protected
nearly all of each ~20-token two-hop prompt — confounding the "effect
vanished" observation. This run isolates the semantics factor:
select=positive (bug fixed), skip_first_positions=0 (prompts fully ablated).

If two-hop collapses again -> original effect stands under honest semantics;
fragility headline: effect exists only at incoherent doses.
If two-hop survives -> the original "reasoning destruction" was largely the
injection artifact.

    python -u scripts/disambig.py --lens results/lens_qwen3-4b_n148_repaired.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "gates")
FULL = list(range(12, 29))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    from scripts.gates import load_everything, twohop_generation
    from src.gates import fetch_official_eval

    model, tok, lens = load_everything(args.lens, args.model)
    items = fetch_official_eval("multihop")["items"][:30]
    out = []
    for name, band, k in [("full-k10", FULL, 10),
                          ("win12-15-k10", list(range(12, 16)), 10)]:
        clean, abl, wf = twohop_generation(
            model, tok, lens, band, k, items,
            select="positive", skip_first_positions=0)
        rel = 1 - abl / max(1e-9, clean)
        row = {"config": name, "semantics": "positive+skip0",
               "twohop_clean": clean, "twohop_ablated": abl,
               "twohop_rel_drop": rel, "wellformed": wf}
        out.append(row)
        print(f"{name}: 2hop {clean:.2f}->{abl:.2f} (rel {rel:.2f}) wf={wf:.2f}",
              flush=True)

    with open(os.path.join(RESULTS, "disambig.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote disambig.json", flush=True)


if __name__ == "__main__":
    main()
