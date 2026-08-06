"""Pre-ship experiment package, ranks 2-4 (DECISION.md 2026-08-04).

Rank 2 — paper's-own-criterion battery at the strong point: MMLU-100 +
         copy-20, arms clean / jspace / random_tokens (fair control).
Rank 3 — teacher-forced two-hop on all 93 items + 30-item one-hop control:
         continuous log p(target) replaces the noisy substring-scored n=30
         sampled generation; first-answer-token metrics depend only on
         (ablated) prompt processing.
Rank 4 — renorm arm + operator-matched random-token control.

All points at honest semantics (positive selection, skip=0). Persists dose
stats and every transcript. Run with `python -u`.

    python -u scripts/preship.py --lens results/lens_qwen3-4b_n148_repaired.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "preship")
FULL = list(range(12, 29))
WIN = list(range(12, 16))

# rank 3+4 measurement points, all honest semantics
POINTS = [
    ("full-k10",          dict(band_layers=FULL, k=10, mode="jspace")),
    ("full-k10-nospare",  dict(band_layers=FULL, k=10, mode="jspace", spare=False)),
    ("win12-15-k10",      dict(band_layers=WIN, k=10, mode="jspace")),
    ("win12-15-k3",       dict(band_layers=WIN, k=3, mode="jspace")),
    ("full-k10-renorm",   dict(band_layers=FULL, k=10, mode="jspace", renorm=True)),
    ("full-k10-randtok0", dict(band_layers=FULL, k=10, mode="random_tokens", seed=0)),
    ("full-k10-randtok1", dict(band_layers=FULL, k=10, mode="random_tokens", seed=1)),
    ("full-k10-randtok2", dict(band_layers=FULL, k=10, mode="random_tokens", seed=2)),
    ("full-k10-rank1rand", dict(band_layers=FULL, k=10, mode="random")),
]

ONE_HOP = [  # domain-matched to the multihop items' second hops
    ("Fact: The chemical symbol for gold is", "Au"),
    ("Fact: The chemical symbol for iron is", "Fe"),
    ("Fact: The chemical symbol for copper is", "Cu"),
    ("Fact: The capital of France is", "Paris"),
    ("Fact: The capital of Japan is", "Tokyo"),
    ("Fact: The capital of Kansas is", "Topeka"),
    ("Fact: The ocean on the east coast of Brazil is the", "Atlantic"),
    ("Fact: The language most spoken in Brazil is", "Portuguese"),
    ("Fact: The color of the planet Mars is", "red"),
    ("Fact: The number of legs on a spider is", "8"),
    ("Fact: The number of players per side in basketball is", "5"),
    ("Fact: The continent where China is located is", "Asia"),
    ("Fact: The season when Christmas occurs in the north is", "winter"),
    ("Fact: The state west of Kansas is", "Colorado"),
    ("Fact: The number of natural moons orbiting Earth is", "1"),
    ("Fact: The Roman god of war is named", "Mars"),
    ("Fact: The US state where Seattle is located is", "Washington"),
    ("Fact: The European country whose capital is Madrid is", "Spain"),
    ("Fact: The number that doubled gives eight is", "4"),
    ("Fact: The large mammal that rhymes with 'chair' is the", "bear"),
    ("Fact: The state of matter of water at room temperature is", "liquid"),
    ("Fact: The flame color of burning sodium is", "yellow"),
    ("Fact: The planet third from the Sun is", "Earth"),
    ("Fact: The month named after Julius Caesar is", "July"),
    ("Fact: The sport invented in Springfield, Massachusetts is", "basketball"),
    ("Fact: The country where Carnival in Rio takes place is", "Brazil"),
    ("Fact: The river that ends in Brazil is the", "Amazon"),
    ("Fact: The college football rival of Ohio State is", "Michigan"),
    ("Fact: The element with atomic number 79 is", "gold"),
    ("Fact: The holiday with a decorated tree is", "Christmas"),
]


def make_ablator(lens, point_cfg):
    from src.ablate import AblationConfig, JSpaceAblator
    return JSpaceAblator(lens, AblationConfig(
        select="positive", skip_first_positions=0, **point_cfg))


@torch.no_grad()
def teacher_forced_items(model, tok, lens, items, point_cfg, clean_cache):
    """Per item: first-answer-token logprob & rank, full-target mean logprob,
    under teacher forcing with (or without) ablation of every position."""
    rows = []
    for idx, (prompt, target) in enumerate(items):
        ids_p = tok(prompt.rstrip(), return_tensors="pt").input_ids
        ids_t = tok(" " + target, add_special_tokens=False,
                    return_tensors="pt").input_ids
        seq = torch.cat([ids_p, ids_t], dim=1).to(model.device)
        n_p = ids_p.shape[1]

        if point_cfg is None:
            logits = model(input_ids=seq).logits[0].float()
        else:
            ablator = make_ablator(lens, point_cfg)
            spare = None
            if ablator.cfg.spare and ablator.cfg.mode != "none":
                key = ("clean_logits", prompt, target)
                if key not in clean_cache:
                    clean_cache[key] = model(input_ids=seq).logits[0]
                spare = clean_cache[key].topk(10, dim=-1).indices.unsqueeze(0)
            ablator.set_spare_ids(spare)
            ablator.set_position_offset(0)
            with ablator.hooks():
                logits = model(input_ids=seq).logits[0].float()

        logp = logits.log_softmax(-1)
        first_id = int(seq[0, n_p])
        first_lp = float(logp[n_p - 1, first_id])
        first_rank = int((logits[n_p - 1] > logits[n_p - 1, first_id]).sum()) + 1
        t_ids = seq[0, n_p:]
        t_lps = [float(logp[n_p - 1 + i, int(t)]) for i, t in enumerate(t_ids)]
        rows.append({"idx": idx, "target": target, "first_logprob": first_lp,
                     "first_rank": first_rank,
                     "target_mean_logprob": sum(t_lps) / len(t_lps)})
    return rows


@torch.no_grad()
def coherence(model, tok, lens, passages, point_cfg):
    """Top-1 match vs clean on passages; overall + in-fit/out-fit strata +
    removed-norm dose."""
    from src.ablate import AblationConfig, JSpaceAblator
    stats = {"in": [0, 0], "out": [0, 0]}
    ablator = make_ablator(lens, point_cfg)
    for p in passages:
        ids = tok(p.rstrip(), return_tensors="pt", truncation=True,
                  max_length=384).input_ids.to(model.device)
        clean_logits = model(input_ids=ids).logits[0]
        spare = (clean_logits.topk(10, dim=-1).indices.unsqueeze(0)
                 if ablator.cfg.spare else None)
        ablator.set_spare_ids(spare)
        ablator.set_position_offset(0)
        with ablator.hooks():
            abl_logits = model(input_ids=ids).logits[0]
        match = clean_logits.argmax(-1) == abl_logits.argmax(-1)
        T = ids.shape[1]
        for t in range(T):
            key = "in" if 16 <= t < 128 else "out"
            stats[key][0] += int(match[t])
            stats[key][1] += 1
    s = ablator.stats
    total = stats["in"][0] + stats["out"][0]
    n = stats["in"][1] + stats["out"][1]
    return {"top1": total / n,
            "top1_infit": stats["in"][0] / max(1, stats["in"][1]),
            "top1_outfit": stats["out"][0] / max(1, stats["out"][1]),
            "removed_fraction": s.mean_removed_fraction,
            "mean_removed_norm": s.mean_removed_norm}


@torch.no_grad()
def battery(model, tok, lens, arms):
    """Rank 2: MMLU-100 + copy-20 per arm, greedy decoding, transcripts kept."""
    from datasets import load_dataset

    from src.ablate import generate_with_ablation

    mmlu = load_dataset("cais/mmlu", "all", split="validation").select(range(100))
    copy_text = ("The quick brown fox jumps over the lazy dog near the "
                 "riverbank at dawn.")
    out = {}
    transcripts = []
    for arm_name, point_cfg in arms.items():
        correct = 0
        for i, row in enumerate(mmlu):
            letters = ["A", "B", "C", "D"]
            q = row["question"] + "\n" + "\n".join(
                f"{l}. {c}" for l, c in zip(letters, row["choices"]))
            text = tok.apply_chat_template(
                [{"role": "user", "content": q + "\n\nAnswer with just the letter."}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            ablator = make_ablator(lens, point_cfg or dict(
                band_layers=FULL, k=10, mode="none"))
            res = generate_with_ablation(
                model, ids, ablator, max_new_tokens=4, temperature=0.0,
                top_p=1.0, seed=i, eos_token_ids=[tok.eos_token_id])
            answer = tok.decode(res["sequence"]).strip()[:1].upper()
            ok = answer == letters[row["answer"]]
            correct += ok
            transcripts.append({"task": "mmlu", "arm": arm_name, "i": i,
                                "text": tok.decode(res["sequence"]), "ok": ok})
        copied = 0
        for i in range(20):
            text = tok.apply_chat_template(
                [{"role": "user",
                  "content": f"Copy this sentence exactly: {copy_text}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            ablator = make_ablator(lens, point_cfg or dict(
                band_layers=FULL, k=10, mode="none"))
            res = generate_with_ablation(
                model, ids, ablator, max_new_tokens=40, temperature=0.0,
                top_p=1.0, seed=i, eos_token_ids=[tok.eos_token_id])
            gen = tok.decode(res["sequence"])
            copied += copy_text in gen
            transcripts.append({"task": "copy", "arm": arm_name, "i": i,
                                "text": gen})
        out[arm_name] = {"mmlu100": correct / 100, "copy20": copied / 20}
        print(f"battery {arm_name}: mmlu={correct/100:.2f} copy={copied/20:.2f}",
              flush=True)
    return out, transcripts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    from scripts.phase0 import held_out_passages, load_everything
    from src.phase0 import fetch_official_eval

    model, tok, lens = load_everything(args.lens, args.model)
    os.makedirs(RESULTS, exist_ok=True)

    two_hop = [(it["prompt"], it["target"])
               for it in fetch_official_eval("multihop")["items"]]  # all 93
    passages = held_out_passages(10)
    clean_cache: dict = {}

    print("== clean reference (teacher-forced) ==", flush=True)
    ref = {"two_hop": teacher_forced_items(model, tok, lens, two_hop, None, clean_cache),
           "one_hop": teacher_forced_items(model, tok, lens, ONE_HOP, None, clean_cache)}
    results = {"clean": ref, "points": {}}

    for name, cfg in POINTS:
        point = {
            "two_hop": teacher_forced_items(model, tok, lens, two_hop, cfg, clean_cache),
            "one_hop": teacher_forced_items(model, tok, lens, ONE_HOP, cfg, clean_cache),
            "coherence": coherence(model, tok, lens, passages, cfg),
        }
        results["points"][name] = point
        d2 = (sum(r["first_logprob"] for r in ref["two_hop"])
              - sum(r["first_logprob"] for r in point["two_hop"])) / len(two_hop)
        d1 = (sum(r["first_logprob"] for r in ref["one_hop"])
              - sum(r["first_logprob"] for r in point["one_hop"])) / len(ONE_HOP)
        c = point["coherence"]
        print(f"{name:>20}: d_logp 2hop={d2:+.2f} 1hop={d1:+.2f} "
              f"top1={c['top1']:.3f} (in {c['top1_infit']:.3f}/out "
              f"{c['top1_outfit']:.3f}) dose={c['removed_fraction']:.3f}",
              flush=True)
        with open(os.path.join(RESULTS, "teacher_forced.json"), "w") as f:
            json.dump(results, f, indent=1)

    print("== rank 2: battery ==", flush=True)
    arms = {"clean": None,
            "jspace": dict(band_layers=FULL, k=10, mode="jspace"),
            "randtok": dict(band_layers=FULL, k=10, mode="random_tokens", seed=0)}
    batt, transcripts = battery(model, tok, lens, arms)
    with open(os.path.join(RESULTS, "battery.json"), "w") as f:
        json.dump(batt, f, indent=1)
    with open(os.path.join(RESULTS, "transcripts.jsonl"), "w") as f:
        for t in transcripts:
            f.write(json.dumps(t) + "\n")
    print("preship complete", flush=True)


if __name__ == "__main__":
    main()
