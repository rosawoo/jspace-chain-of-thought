"""Surgical outlier removal from a fitted lens checkpoint.

The fit checkpoint stores the running SUM of per-prompt Jacobians, so an
outlier prompt can be removed exactly: recompute its Jacobian (~40 s each on
A100), subtract from the sum, divide by (n - n_removed).

Safety: the recomputed per-prompt norm must match the value recorded in
fit.log to 1% — this simultaneously verifies that prompt regeneration
(same seed, same corpus stream) is deterministic. Aborts otherwise.

    python scripts/repair_lens.py --checkpoint results/fit_ckpt.pt \
        --outliers 24 139 --expected-norms 141.576 275.659 \
        --out results/lens_qwen3-4b_n148_repaired.pt

--outliers are ZERO-based prompt indices (fit.log prints 1-based: "prompt 25"
-> index 24).
"""

from __future__ import annotations

import argparse
import math
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outliers", type=int, nargs="+", required=True)
    parser.add_argument("--expected-norms", type=float, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n-prompts", type=int, default=150)
    parser.add_argument("--prompt-seed", type=int, default=0)
    parser.add_argument("--target-layer", type=int, default=-2)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    args = parser.parse_args()
    assert len(args.outliers) == len(args.expected_norms)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import jlens
    from jlens.fitting import jacobian_for_prompt
    from jlens.lens import JacobianLens

    sys.path.insert(0, ".")
    from src.jlens_fit import load_pretraining_prompts

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    jacobian_sum, n_done = state["jacobian_sum"], state["n_done"]
    source_layers = state["source_layers"]
    d_model = next(iter(jacobian_sum.values())).shape[0]
    sqrt_d = math.sqrt(d_model)
    print(f"checkpoint: n_done={n_done}, {len(source_layers)} layers, d={d_model}")

    prompts = load_pretraining_prompts(args.n_prompts, seed=args.prompt_seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lens_model = jlens.from_hf(model, tokenizer)

    to_subtract = []
    for idx, expected in zip(args.outliers, args.expected_norms):
        per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
            lens_model, prompts[idx], source_layers,
            target_layer=args.target_layer, dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len)
        norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        print(f"prompt idx {idx}: recomputed norm {norm:.3f} "
              f"(fit.log said {expected:.3f})")
        if abs(norm - expected) / expected > 0.01:
            raise SystemExit(
                "ABORT: recomputed norm does not match fit.log — prompt "
                "regeneration is not reproducing the fitted corpus; do NOT "
                "subtract. Fall back to a filtered refit.")
        to_subtract.append(per_prompt_J)

    n_new = n_done - len(to_subtract)
    repaired = {}
    for l in source_layers:
        total = jacobian_sum[l].clone()
        for J in to_subtract:
            total -= J[l]
        repaired[l] = total / n_new
    JacobianLens(jacobians=repaired, n_prompts=n_new, d_model=d_model).save(args.out)
    print(f"saved repaired lens (n={n_new}) to {args.out}")


if __name__ == "__main__":
    main()
