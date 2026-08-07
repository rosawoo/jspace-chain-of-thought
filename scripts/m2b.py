"""M2b — lesion-free workspace telemetry (H5), redesigned after review.

REVIEW FINDING (blocking, 2026-08-04): Qwen3 tokenizes numbers digit-by-digit,
so single-token numeric probes are impossible for multi-digit values. H5 is
therefore split:

PART A (primary, validated probes): H5(a) demand-contrast on the 93 multihop
items — J-lens loading of the BRIDGE ENTITY (single-token, Gate-B-validated)
at the last 5 stimulus positions, under a direct-answer prefix vs a
think-step-by-step prefix. Identical stimulus tokens; only the instruction
prefix differs. Prediction: bridge loading higher under direct (the workspace
must hold hop-1 internally when externalization is disallowed).

PART B/C (exploratory, digit probes): math difficulty axis + CoT lookahead
using the FIRST-DIGIT lens log-probability (not rank) with within-problem
paired contrasts only (token-frequency confounds cancel in the contrast, not
in absolute values). GSM8K intermediates = RHS of gold <<...=X>> calculator
annotations; MATH values filtered (no 1-digit, no ^_{ contexts); lookahead
anchored by boundary regex and id-based token mapping.

    python -u scripts/m2b.py --lens LENS [--smoke]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "m2b")
PROBE_LAYERS = [12, 16, 20, 24, 28]
EOS_IDS = [151643, 151645]  # <|endoftext|>, <|im_end|>

DIRECT_PREFIX = "Answer immediately with only the final word or number.\n"
COT_PREFIX = ("Think step by step, writing out every intermediate fact, "
              "before giving the answer.\n")


# ---------- shared measurement core ----------

@torch.no_grad()
def lens_readout_at(model, tok, lens, input_ids: torch.Tensor,
                    positions: list[int]) -> dict[int, torch.Tensor]:
    """{layer: [P, vocab] lens log-softmax} at the given positions."""
    from src.jlens_core import record_residuals

    T = input_ids.shape[1]
    positions = [p for p in positions if 0 <= p < T]
    if not positions:
        return {}
    with record_residuals(lens.adapter.layers, at=PROBE_LAYERS) as rec:
        model(input_ids=input_ids)
    out = {}
    for layer in PROBE_LAYERS:
        h = rec.acts[layer][0][positions].float()
        out[layer] = lens.strengths(h, layer).log_softmax(-1).cpu()
    return out


def best_rank(readout: dict[int, torch.Tensor], token_ids: list[int]) -> int | None:
    if not readout or not token_ids:
        return None
    best = None
    for lg in readout.values():
        vals = lg[:, token_ids].max(dim=-1).values  # [P]
        ranks = (lg > vals.unsqueeze(-1)).sum(-1) + 1  # [P]
        r = int(ranks.min())
        best = r if best is None else min(best, r)
    return best


def best_logprob(readout: dict[int, torch.Tensor], token_ids: list[int]) -> float | None:
    if not readout or not token_ids:
        return None
    return max(float(lg[:, token_ids].max()) for lg in readout.values())


# ---------- PART A: verbal demand-contrast ----------

def stimulus_window(tok, prefix: str, stimulus: str, n: int = 5) -> tuple[torch.Tensor, list[int]]:
    """Tokenize prefix+stimulus; return ids and the last n token positions
    overlapping the stimulus span (offset-mapping based, junction-safe)."""
    text = prefix + stimulus
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    start = len(prefix)
    stim_positions = [i for i, (a, b) in enumerate(enc.offset_mapping[0].tolist())
                      if b > start]
    return enc.input_ids, stim_positions[-n:]


def part_a(model, tok, lens, smoke: bool) -> list[dict]:
    from src.gates import _single_token_ids, fetch_official_eval

    items = fetch_official_eval("multihop")["items"]
    if smoke:
        items = items[:5]
    rows = []
    for i, item in enumerate(items):
        bridge_ids = []
        for word in item["intermediates"]:
            bridge_ids += _single_token_ids(tok, word)
        if not bridge_ids:
            continue
        row = {"idx": i, "bridge": item["intermediates"]}
        for mode, prefix in (("direct", DIRECT_PREFIX), ("cot", COT_PREFIX)):
            ids, window = stimulus_window(tok, prefix, item["prompt"].rstrip())
            readout = lens_readout_at(model, tok, lens, ids.to(model.device),
                                      window)
            row[f"rank_{mode}"] = best_rank(readout, bridge_ids)
            row[f"logprob_{mode}"] = best_logprob(readout, bridge_ids)
        rows.append(row)
        if i % 20 == 0 or smoke:
            print(f"A[{i}] bridge={item['intermediates']} "
                  f"rank direct={row['rank_direct']} cot={row['rank_cot']}",
                  flush=True)
    return rows


# ---------- PART B/C: exploratory digit probes on math ----------

def clean_number(n: str) -> str:
    n = n.strip(",.").replace(",", "")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def extract_intermediates(question: str, rationale: str, final: str,
                          source: str, max_n: int = 6) -> list[str]:
    q_nums = {clean_number(n) for n in re.findall(r"\d[\d,]*\.?\d*", question)}
    final = clean_number(final)
    if source == "gsm8k":  # gold calculator annotations: RHS of <<...=X>>
        cands = re.findall(r"=\s*([\d.,]+)\s*>>", rationale)
    else:  # MATH LaTeX: exclude ^{...}, _{...}, and enumeration contexts
        cands = [m.group(1) for m in
                 re.finditer(r"(?<![\^_{(\d])(\d[\d,]*\.?\d*)", rationale)]
    seen, out = set(), []
    for n in cands:
        c = clean_number(n)
        if (not c or len(c) < 2 or c in q_nums or c == final or c in seen):
            continue
        seen.add(c)
        out.append(c)
    return out[:max_n]


def first_digit_id(tok, value: str) -> list[int]:
    """Token id(s) of the value's first token in ' <value>' context."""
    enc = tok.encode(" " + value, add_special_tokens=False)
    enc2 = tok.encode(value, add_special_tokens=False)
    return sorted({enc[0], enc2[0]} if enc and enc2 else set())


def boundary_find(text: str, value: str) -> int:
    """First standalone occurrence of value; tolerates thousands separators
    (the model writes 130,000 where the cleaned gold value is 130000)."""
    if "." in value:
        core = re.escape(value)
    else:
        core = r",?".join(re.escape(d) for d in value)
    m = re.search(rf"(?<![\d.])(?:{core})(?!\.?\d)", text)
    return m.start() if m else -1


def gen_token_index(tok, gen_ids: list[int], char_pos: int) -> int | None:
    """First generated-token index whose cumulative decode reaches char_pos."""
    text = ""
    for i, t in enumerate(gen_ids):
        text = tok.decode(gen_ids[:i + 1])
        if len(text) > char_pos:
            return i
    return None


def load_math_problems(smoke: bool) -> list[dict]:
    from datasets import load_dataset

    out = []
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    n_gsm = 3 if smoke else 25
    for row in gsm.select(range(300)):
        if sum(p["source"] == "gsm8k" for p in out) >= n_gsm:
            break
        final = row["answer"].split("####")[-1].strip().replace(",", "")
        inter = extract_intermediates(row["question"], row["answer"], final,
                                      "gsm8k")
        if len(inter) >= 2:
            out.append({"source": "gsm8k", "level": 0,
                        "question": row["question"], "final": final,
                        "intermediates": inter})
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    per_level = 1 if smoke else 15
    for level in (1, 3, 5):
        picked = 0
        for row in math:
            if int(row["level"]) != level or picked >= per_level:
                continue
            inter = extract_intermediates(row["problem"], row["solution"],
                                          str(row["answer"]), "math500")
            if len(inter) >= 2:
                out.append({"source": "math500", "level": level,
                            "question": row["problem"],
                            "final": str(row["answer"]),
                            "intermediates": inter})
                picked += 1
    return out


def part_bc(model, tok, lens, smoke: bool) -> list[dict]:
    from src.ablate import AblationConfig, JSpaceAblator, generate_with_ablation

    problems = load_math_problems(smoke)
    print(f"B/C: {len(problems)} problems with >=2 gold intermediates",
          flush=True)
    rows = []
    for pi, prob in enumerate(problems):
        probes = [first_digit_id(tok, v) for v in prob["intermediates"]]
        row = {"idx": pi, "source": prob["source"], "level": prob["level"],
               "intermediates": prob["intermediates"]}

        # (B) end-of-question first-digit logprob, direct vs cot prefix
        for mode, prefix in (("direct", DIRECT_PREFIX), ("cot", COT_PREFIX)):
            ids, window = stimulus_window(tok, prefix, prob["question"])
            readout = lens_readout_at(model, tok, lens, ids.to(model.device),
                                      window)
            row[f"eoq_logprob_{mode}"] = [best_logprob(readout, p)
                                          for p in probes]

        # (C) CoT lookahead: pre-write window vs post-write control window.
        # Chat-template CoT (like M2a) — the raw-completion prefix triggers
        # unbounded thinking-style rambling where intermediates only appear
        # after token ~500 (diagnosed 2026-08-05); template CoT is ~230 tokens.
        text = tok.apply_chat_template(
            [{"role": "user", "content": prob["question"] +
              "\n\nThink step by step, then give the final answer inside "
              "\\boxed{}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        ablator = JSpaceAblator(lens, AblationConfig(band_layers=[12],
                                                     mode="none"))
        gen = generate_with_ablation(
            model, ids, ablator, max_new_tokens=256 if smoke else 640,
            temperature=0.7, top_p=0.8, seed=pi, eos_token_ids=EOS_IDS)
        gen_ids = gen["sequence"]
        gen_text = tok.decode(gen_ids)
        full_ids = torch.cat([ids, torch.tensor([gen_ids],
                                                device=ids.device)], dim=1)
        n_prompt = ids.shape[1]
        # one batched readout over the union of all windows for this problem
        windows, spans = [], []
        for vi, value in enumerate(prob["intermediates"]):
            cpos = boundary_find(gen_text, value)
            gtok = gen_token_index(tok, gen_ids, cpos) if cpos >= 0 else None
            if gtok is None:
                spans.append(None)
                continue
            pre = list(range(max(n_prompt, n_prompt + gtok - 10),
                             n_prompt + gtok))
            post = list(range(n_prompt + gtok + 1,
                              min(full_ids.shape[1], n_prompt + gtok + 11)))
            spans.append((len(windows), len(pre), len(post)))
            windows += pre + post
        readout = lens_readout_at(model, tok, lens, full_ids, windows)
        look = []
        for vi, span in enumerate(spans):
            if span is None or not readout:
                look.append(None)
                continue
            off, n_pre, n_post = span
            sub_pre = {l: lg[off:off + n_pre] for l, lg in readout.items()}
            sub_post = {l: lg[off + n_pre:off + n_pre + n_post]
                        for l, lg in readout.items()}
            look.append({"pre": best_logprob(sub_pre, probes[vi]),
                         "post": best_logprob(sub_post, probes[vi])})
        row["lookahead"] = look
        row["cot_text"] = gen_text
        rows.append(row)
        if pi % 10 == 0 or smoke:
            print(f"BC[{pi}] {prob['source']} L{prob['level']} "
                  f"lookahead={look[:2]}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from scripts.gates import load_everything

    model, tok, lens = load_everything(args.lens, args.model)
    os.makedirs(RESULTS, exist_ok=True)
    tag = "smoke_" if args.smoke else ""

    rows_a = part_a(model, tok, lens, args.smoke)
    with open(os.path.join(RESULTS, f"{tag}part_a.jsonl"), "w") as f:
        for r in rows_a:
            f.write(json.dumps(r) + "\n")
    d = [r for r in rows_a if r["rank_direct"] and r["rank_cot"]]
    if d:
        import statistics
        med = lambda k: statistics.median(r[k] for r in d)
        print(f"PART A summary (n={len(d)}): median bridge rank "
              f"direct={med('rank_direct')} cot={med('rank_cot')} "
              f"(H5a predicts direct < cot)", flush=True)

    rows_bc = part_bc(model, tok, lens, args.smoke)
    with open(os.path.join(RESULTS, f"{tag}part_bc.jsonl"), "w") as f:
        for r in rows_bc:
            f.write(json.dumps(r) + "\n")
    print("m2b complete", flush=True)


if __name__ == "__main__":
    main()
