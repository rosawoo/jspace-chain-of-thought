"""Fit the J-lens for Qwen3-4B using the official anthropics/jacobian-lens
package (pip install "git+https://github.com/anthropics/jacobian-lens").

Defaults follow the paper's appendix recipe as documented in NOTES.md:
penultimate target layer, mean aggregation over source positions and later
targets, 128-token prompts, SKIP_FIRST=16, fp32 accumulation, checkpointing.

Usage (on the GPU pod):
    python -m src.jlens_fit --n-prompts 150 \
        --out results/lens_qwen3-4b_n150.pt --checkpoint results/fit_ckpt.pt
"""

from __future__ import annotations

import argparse
import logging


def load_pretraining_prompts(n: int, min_chars: int = 600, seed: int = 0) -> list[str]:
    """Sample pretraining-like prompts (fineweb sample-10BT, streamed)."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    prompts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_chars:
            prompts.append(text[:4000])
        if len(prompts) >= n:
            break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n-prompts", type=int, default=150)
    parser.add_argument("--target-layer", type=int, default=-2,
                        help="-2 = penultimate (paper appendix default recipe)")
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt-seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import jlens  # official package
    from jlens.fitting import fit

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lens_model = jlens.from_hf(model, tokenizer)

    prompts = load_pretraining_prompts(args.n_prompts, seed=args.prompt_seed)
    lens = fit(
        lens_model,
        prompts,
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
        checkpoint_every=1,
    )
    lens.save(args.out)
    print(f"saved lens to {args.out}: {lens!r}")


if __name__ == "__main__":
    main()
