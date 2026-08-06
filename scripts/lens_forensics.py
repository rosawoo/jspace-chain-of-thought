"""Pre-Gate-A lens forensics — runs on CPU (laptop), no model needed.

    python scripts/lens_forensics.py --lens results/lens_qwen3-4b_n150.pt \
        --log results/fit.log

Checks:
1. fit.log prompt-norm ledger: outlier count and share of total magnitude.
2. Per-layer J structure: ||J||_F, effective rank, distance from identity.
   Healthy profile: late layers near-identity (logit-lens regime), early
   layers far from it, smooth progression, no rank collapse.
"""

from __future__ import annotations

import argparse
import math
import re

import torch


def parse_norm_ledger(log_path: str) -> list[tuple[int, float]]:
    """(prompt_idx, max||J||/sqrt(d)) per fitted prompt, from fit.log."""
    ledger = []
    pattern = re.compile(
        r"prompt (\d+)/\d+ .*max\|\|J\|\|/sqrt\(d\)=([0-9.]+)")
    for line in open(log_path):
        m = pattern.search(line)
        if m:
            ledger.append((int(m.group(1)), float(m.group(2))))
    return ledger


def ledger_report(ledger: list[tuple[int, float]]) -> None:
    norms = torch.tensor([n for _, n in ledger])
    median = norms.median().item()
    print(f"prompts fitted: {len(norms)}  median norm: {median:.2f}  "
          f"max: {norms.max():.2f}")
    outliers = [(i, n) for i, n in ledger if n > 10 * median]
    share = sum(n for _, n in outliers) / norms.sum().item()
    print(f"outliers (>10x median): {len(outliers)} -> {outliers}")
    print(f"outlier share of summed per-prompt norm: {share:.1%}")
    if share > 0.25:
        print("!! outliers dominate the mean — consider the checkpoint "
              "subtraction repair (recompute outlier prompts' J, subtract "
              "from jacobian_sum in fit_ckpt.pt, divide by n - n_outliers)")


def lens_report(lens_path: str, sample_layers: int = 12) -> None:
    ckpt = torch.load(lens_path, map_location="cpu", weights_only=True)
    jac = ckpt["J"] if "J" in ckpt else ckpt
    layers = sorted(jac)
    step = max(1, len(layers) // sample_layers)
    print(f"\nlayer  ||J||_F/sqrt(d)  eff_rank  ||J-I||/||I||  top_sv")
    for l in layers[::step]:
        J = jac[l].float()
        d = J.shape[0]
        eye = torch.eye(d)
        sv = torch.linalg.svdvals(J)
        eff_rank = (sv.sum() ** 2 / (sv ** 2).sum()).item()  # participation ratio
        print(f"L{l:>3}  {J.norm().item() / math.sqrt(d):>13.3f}  "
              f"{eff_rank:>8.0f}  {(J - eye).norm().item() / eye.norm().item():>12.3f}"
              f"  {sv[0].item():>7.2f}")
    print("\nhealthy: ||J-I||/||I|| large early, shrinking toward late layers;"
          "\neff_rank smooth across depth (no isolated collapses/spikes).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    if args.log:
        ledger_report(parse_norm_ledger(args.log))
    lens_report(args.lens)


if __name__ == "__main__":
    main()
