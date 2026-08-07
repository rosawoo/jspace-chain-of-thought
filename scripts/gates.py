"""Phase-0 gate driver. Run on the GPU pod after lens fitting.

Usage:
    python scripts/gates.py gate-a --lens results/lens_qwen3-4b_n150.pt
    python scripts/gates.py gate-b --lens ... --band 12 13 ... 28
    python scripts/gates.py gate-c --lens ... --band ...
    python scripts/gates.py gate-d --lens ... --band ...   # operating-point sweep

Every subcommand writes JSON to results/gates/<gate>.json. Gate thresholds
live in README.md and are FROZEN; this script only reports numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "gates")


def load_everything(lens_path: str, model_name: str = "Qwen/Qwen3-4B"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.jlens_core import LensSpace, ModelAdapter, load_lens_matrices

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    adapter = ModelAdapter.from_hf(model)
    jac = load_lens_matrices(lens_path)
    lens = LensSpace(adapter, jac,
                     row_norm_cache=lens_path.replace(".pt", "_rownorms.pt"))
    return model, tok, lens


def held_out_passages(n: int, seed: int = 1) -> list[str]:
    from src.jlens_fit import load_pretraining_prompts
    return load_pretraining_prompts(n, seed=seed)  # different seed than fitting


def save(gate: str, payload: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{gate}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"wrote {path}")


def gate_a(args) -> None:
    from src.gates import band_metrics, occupancy_curve

    model, tok, lens = load_everything(args.lens, args.model)
    prompts = held_out_passages(args.n_prompts)
    layers = lens.layers
    metrics = band_metrics(model, tok, lens, prompts, layers)
    occ_layers = layers[:: max(1, len(layers) // 9)]
    occupancy = occupancy_curve(model, tok, lens, prompts[:args.n_occupancy],
                                occ_layers, k_max=40)
    save("gate_a", {"band_metrics": {str(l): m for l, m in metrics.items()},
                    "occupancy": {str(l): o for l, o in occupancy.items()}})
    print("\nLayer  kurtosis  agree@10  autocorr_excess")
    for l in layers:
        m = metrics[l]
        print(f"L{l:>3}  {m['kurtosis']:>8.2f}  {m['next_token_agree']:>8.3f}"
              f"  {m['autocorr_excess']:>8.3f}")


def gate_b(args) -> None:
    from src.gates import two_hop_readout

    model, tok, lens = load_everything(args.lens, args.model)
    out = two_hop_readout(model, tok, lens, args.band, n_items=20)
    out["pass"] = out["hits"] >= 12
    save("gate_b", out)
    print(f"Gate B: {out['hits']}/20 bridge entities in top-10 "
          f"({'PASS' if out['pass'] else 'FAIL'} at frozen threshold 12/20)")


def gate_c(args) -> None:
    from datasets import load_dataset

    from src.gates import numeric_loading_audit

    model, tok, lens = load_everything(args.lens, args.model)
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = ([direct_prompt(tok, r["question"]) for r in gsm.select(range(50))]
               + [direct_prompt(tok, r["problem"]) for r in math.select(range(25))])
    out = numeric_loading_audit(model, tok, lens, args.band, prompts, k=args.k)
    out["pass"] = out["numeric_fraction"] >= 0.10
    save("gate_c", out)
    print(f"Gate C: numeric fraction {out['numeric_fraction']:.3f} "
          f"({'PASS' if out['pass'] else 'FAIL — promote S4, pre-scope math nulls'})")


def direct_prompt(tok, question: str) -> str:
    messages = [{"role": "user", "content":
                 question + "\nAnswer with only the final answer."}]
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)


def gate_d(args) -> None:
    """Operating-point sweep on the VERBAL two-hop control only (never math)."""
    from src.ablate import AblationConfig, JSpaceAblator, generate_with_ablation
    from src.gates import fetch_official_eval, pretraining_top1_match

    model, tok, lens = load_everything(args.lens, args.model)
    passages = held_out_passages(50)
    items = fetch_official_eval("multihop")["items"][:30]
    bands = {
        "full": args.band,
        "early-half": args.band[: len(args.band) // 2],
        "late-half": args.band[len(args.band) // 2:],
    }
    sweep = []
    for band_name, band in bands.items():
        for k in (3, 5, 10):
            top1 = pretraining_top1_match(model, tok, lens, band,
                                          passages[:20], k=k)
            twohop_clean, twohop_abl, wellformed = twohop_generation(
                model, tok, lens, band, k, items)
            rel_drop = 1 - twohop_abl / max(1e-9, twohop_clean)
            entry = {"band": band_name, "layers": band, "k": k,
                     "pretraining_top1_match": top1,
                     "twohop_clean": twohop_clean, "twohop_ablated": twohop_abl,
                     "twohop_rel_drop": rel_drop, "wellformed": wellformed,
                     "coherent": top1 >= 0.85 and wellformed >= 0.90,
                     "effective": rel_drop >= 0.50}
            entry["viable"] = entry["coherent"] and entry["effective"]
            sweep.append(entry)
            print(f"{band_name:>10} k={k:>2}: top1={top1:.3f} "
                  f"2hop {twohop_clean:.2f}->{twohop_abl:.2f} "
                  f"wf={wellformed:.2f} "
                  f"{'VIABLE' if entry['viable'] else ''}")
    save("gate_d", {"sweep": sweep,
                    "viable": [e for e in sweep if e["viable"]]})


def twohop_generation(model, tok, lens, band, k, items,
                      max_new_tokens: int = 12, arms=("none", "jspace"),
                      **ablation_kwargs):
    """Two-hop answer accuracy, clean vs ablated, + well-formedness."""
    from src.ablate import AblationConfig, JSpaceAblator, generate_with_ablation

    def run(mode):
        correct, wellformed = 0, 0
        for i, item in enumerate(items):
            ids = tok(item["prompt"].rstrip(), return_tensors="pt"
                      ).input_ids.to(model.device)
            ablator = JSpaceAblator(lens, AblationConfig(
                band_layers=band, k=k, mode=mode, spare=True,
                **ablation_kwargs))
            out = generate_with_ablation(
                model, ids, ablator, max_new_tokens=max_new_tokens,
                temperature=0.7, top_p=0.8, seed=1000 + i,
                eos_token_ids=[tok.eos_token_id])
            text = tok.decode(out["sequence"])
            correct += item["target"].lower() in text.lower()
            # printable-or-whitespace: isprintable() alone rejects newlines
            wellformed += (len(text.strip()) > 0
                           and all(c.isprintable() or c.isspace() for c in text))
        return correct / len(items), wellformed / len(items)

    results = [run(m) for m in arms]
    if arms == ("none", "jspace"):
        (clean_acc, _), (abl_acc, wf) = results
        return clean_acc, abl_acc, wf
    return {m: r for m, r in zip(arms, results)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("gate-a", "gate-b", "gate-c", "gate-d"):
        s = sub.add_parser(name)
        s.add_argument("--lens", required=True)
        s.add_argument("--model", default="Qwen/Qwen3-4B")
        s.add_argument("--band", type=int, nargs="+", default=None,
                       help="band layer indices (from Gate A) for gates b/c/d")
        s.add_argument("--n-prompts", type=int, default=50)
        s.add_argument("--n-occupancy", type=int, default=10)
        s.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.cmd != "gate-a" and not args.band:
        parser.error(f"{args.cmd} requires --band (identified by gate-a)")
    {"gate-a": gate_a, "gate-b": gate_b,
     "gate-c": gate_c, "gate-d": gate_d}[args.cmd](args)


if __name__ == "__main__":
    main()
