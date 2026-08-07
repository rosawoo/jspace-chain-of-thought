"""Recompute every number in the report's tables from the raw results.

    .venv/bin/python scripts/make_tables.py

Prints, in order:
  Table 1  (report tab:h4)   GSM8K accuracy by mode x intervention,
                             matched 75 problems, Wilson CIs, McNemar,
                             pre-registered H4 statistic (paired bootstrap,
                             seed 0), plus the full-150 direct-mode check.
  Table 2  (report tab:h5a)  H5(a) bridge-loading stats and sign test.
  Table 3  (report tab:spec) damage by condition at the strong operating
                             point (teacher-forced, coherence, MMLU).
  Fig. 2 stats (H5c)         pre- vs post-write loading and sign test.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test, ties excluded."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_cell(mode: str, arm: str) -> dict:
    path = os.path.join(R, "m2a", f"gsm8k_{mode}_{arm}.jsonl")
    return {(r["problem_id"], r["sample_idx"]): bool(r["correct"])
            for r in map(json.loads, open(path)) if r.get("ok")}


def table1() -> None:
    maps = {(m, a): load_cell(m, a)
            for m in ("direct", "cot") for a in ("clean", "jspace", "randtok")}
    shared = sorted(set.intersection(
        *[{k[0] for k in maps[("cot", a)]} for a in ("clean", "jspace", "randtok")]))

    def acc(mode, arm, probs):
        keys = [k for k in maps[(mode, "clean")]
                if k in maps[(mode, arm)] and k[0] in probs]
        wins = sum(maps[(mode, arm)][k] for k in keys)
        return wins, len(keys)

    print(f"== Table 1 (tab:h4): matched {len(shared)} problems ==")
    for mode in ("direct", "cot"):
        k_clean, n = acc(mode, "clean", set(shared))
        for arm in ("clean", "jspace", "randtok"):
            k, n = acc(mode, arm, set(shared))
            lo, hi = wilson(k, n)
            rel = "" if arm == "clean" else \
                f"  rel change {(k / n - k_clean / n) / (k_clean / n):+.0%}"
            print(f"  {mode:>6} {arm:<8} {k}/{n} = {k/n:.3f} "
                  f"[{lo:.2f}, {hi:.2f}]{rel}")
        # McNemar clean vs randtok
        c, r = maps[(mode, "clean")], maps[(mode, "randtok")]
        keys = [k for k in c if k in r and k[0] in set(shared)]
        b = sum(1 for k in keys if c[k] and not r[k])
        c01 = sum(1 for k in keys if not c[k] and r[k])
        print(f"  {mode:>6} McNemar clean vs control: {b} vs {c01}, "
              f"p = {sign_test(b, c01):.2f}")

    def rel_drop(mode, arm, probs):
        clean, abl = maps[(mode, "clean")], maps[(mode, arm)]
        keys = [k for k in clean if k in abl and k[0] in probs]
        if not keys:
            return None
        a = np.mean([clean[k] for k in keys])
        b = np.mean([abl[k] for k in keys])
        return None if a == 0 else (a - b) / a

    for label, probs in (("matched 75", shared),
                         ("full 150 (direct check)",
                          sorted({k[0] for k in maps[("direct", "clean")]}))):
        rng = np.random.default_rng(0)
        d = rel_drop("direct", "jspace", set(probs))
        c = rel_drop("cot", "jspace", set(probs))
        asyms = []
        for _ in range(2000):
            sub = set(rng.choice(probs, len(probs)))
            dd = rel_drop("direct", "jspace", sub)
            cc = rel_drop("cot", "jspace", sub)
            if dd is not None and cc is not None:
                asyms.append(dd - 1.5 * cc)
        lo, hi = np.percentile(asyms, [2.5, 97.5])
        print(f"  H4 statistic ({label}): drops {d:+.3f}/{c:+.3f} "
              f"ratio {d/c:.2f}  stat {d - 1.5*c:+.3f}  "
              f"CI [{lo:+.3f}, {hi:+.3f}]")


def table2() -> None:
    rows = [json.loads(l) for l in open(os.path.join(R, "m2b", "part_a.jsonl"))]
    rd = np.array([r["rank_direct"] for r in rows])
    rc = np.array([r["rank_cot"] for r in rows])
    ld = np.array([r["logprob_direct"] for r in rows])
    lc = np.array([r["logprob_cot"] for r in rows])
    print(f"\n== Table 2 (tab:h5a): n = {len(rows)} items ==")
    for name, rank, lp in (("immediate answer", rd, ld),
                           ("step-by-step", rc, lc)):
        print(f"  {name:<17} median rank {np.median(rank):.1f}  "
              f"IQR [{np.percentile(rank, 25):.0f}, "
              f"{np.percentile(rank, 75):.0f}]  "
              f"top-10 {np.mean(rank <= 10):.0%}  "
              f"mean logprob {lp.mean():.2f}")
    wins, losses = int((ld > lc).sum()), int((ld < lc).sum())
    print(f"  sign test (logprob): {wins} vs {losses}, "
          f"p = {sign_test(wins, losses):.2f}")


def table3() -> None:
    tf = json.load(open(os.path.join(R, "preship", "teacher_forced.json")))
    batt = json.load(open(os.path.join(R, "preship", "battery.json")))

    def dlogp(point, task):
        ref, pt = tf["clean"][task], tf["points"][point][task]
        return (sum(r["first_logprob"] for r in ref)
                - sum(r["first_logprob"] for r in pt)) / len(ref)

    print("\n== Table 3 (tab:spec): strong operating point ==")
    print(f"  clean: 0 / 0 / coherence 1.00 / MMLU {batt['clean']['mmlu100']:.2f} "
          f"/ copy {batt['clean']['copy20']:.2f}")
    rows = [("J-space k=10", "full-k10", batt["jspace"]["mmlu100"]),
            ("J-space + renorm", "full-k10-renorm", None),
            ("thin window k=10", "win12-15-k10", None)]
    for label, point, mmlu in rows:
        coh = tf["points"][point]["coherence"]["top1"]
        m = f"{mmlu:.2f}" if mmlu is not None else "--"
        print(f"  {label:<18} two-hop {dlogp(point, 'two_hop'):+.2f}  "
              f"one-hop {dlogp(point, 'one_hop'):+.2f}  "
              f"coherence {coh:.2f}  MMLU {m}")
    seeds = ["full-k10-randtok0", "full-k10-randtok1", "full-k10-randtok2"]
    d2 = np.mean([dlogp(s, "two_hop") for s in seeds])
    d1 = np.mean([dlogp(s, "one_hop") for s in seeds])
    coh = np.mean([tf["points"][s]["coherence"]["top1"] for s in seeds])
    print(f"  {'random tokens (3 seeds)':<18} two-hop {d2:+.2f}  "
          f"one-hop {d1:+.2f}  coherence {coh:.2f}  "
          f"MMLU {batt['randtok']['mmlu100']:.2f}")


def fig2_stats() -> None:
    rows = [json.loads(l) for l in open(os.path.join(R, "m2b", "part_bc.jsonl"))]
    pre, post = [], []
    for r in rows:
        for l in r.get("lookahead") or []:
            if l and l.get("pre") is not None and l.get("post") is not None:
                pre.append(l["pre"])
                post.append(l["post"])
    diffs = np.array(pre) - np.array(post)
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    print(f"\n== Figure 2 stats (H5c): n = {len(diffs)} intermediates ==")
    print(f"  pre-write mean {np.mean(pre):.1f}  post-write mean "
          f"{np.mean(post):.1f}  higher-before {wins}/{len(diffs)}  "
          f"sign test p = {sign_test(wins, losses):.1e}")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    fig2_stats()
