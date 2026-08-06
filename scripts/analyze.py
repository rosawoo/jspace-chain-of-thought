"""Pre-registered analysis for H1-H3. Committed BEFORE any evaluation run
(see git history); numbers change, the tests do not.

    python scripts/analyze.py h1i     # C1: replication + random control + H2 sparing
    python scripts/analyze.py h1ii    # C2: difficulty-slope difference J vs random
    python scripts/analyze.py c3      # AIME anchor survival ratios
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def rows(path: str) -> list[dict]:
    full = os.path.join(RESULTS, path)
    if not os.path.exists(full):
        print(f"  [missing: {path}]")
        return []
    return [r for r in map(json.loads, open(full)) if r.get("ok")]


def acc(rs: list[dict]) -> float:
    return sum(r["correct"] for r in rs) / max(1, len(rs))


def h1i(args) -> None:
    """H1(i): rel_drop(direct)/rel_drop(cot) >= 2 in J arm, absent in random.
    H2: differential survives >=50% with sparing OFF."""
    from src.eval_harness import paired_bootstrap

    out = {}
    for mode in ("direct", "cot"):
        clean = rows(f"c1/gsm8k_{mode}_clean.jsonl")
        for arm in ("jspace", "random", "jspace-nospare"):
            abl = rows(f"c1/gsm8k_{mode}_{arm}.jsonl")
            if not (clean and abl):
                continue
            boot = paired_bootstrap(clean, abl)
            rel_drop = (boot["acc_a"] - boot["acc_b"]) / max(1e-9, boot["acc_a"])
            out[(mode, arm)] = {"clean": boot["acc_a"], "ablated": boot["acc_b"],
                                "rel_drop": rel_drop, "diff_ci95": boot["ci95"]}
            print(f"{mode:>7}/{arm:<15} clean={boot['acc_a']:.3f} "
                  f"abl={boot['acc_b']:.3f} rel_drop={rel_drop:.3f} "
                  f"CI95={boot['ci95']}")
    for arm, label in (("jspace", "H1(i) J-arm"), ("random", "random control"),
                       ("jspace-nospare", "H2 sparing-OFF")):
        d, c = out.get(("direct", arm)), out.get(("cot", arm))
        if d and c:
            ratio = d["rel_drop"] / max(1e-9, c["rel_drop"])
            print(f"{label}: drop-ratio direct/cot = {ratio:.2f} "
                  f"(H1(i) predicts >=2 for J, ~1 for random)")


def h1ii(args) -> None:
    """H1(ii): survival-ratio slope over MATH levels, J vs random.
    Pre-registered statistic: bootstrap (clustered by problem) of
    slope_J - slope_random, where slope = OLS of per-level survival ratio
    on level. CONFIRM-breakdown if slope_J significantly more negative;
    CONFIRM-persistence if difference ~ 0 with r_J flat."""
    clean = rows("c2/math500_cot_clean.jsonl")
    arms = {a: rows(f"c2/math500_cot_{a}.jsonl") for a in ("jspace", "random")}
    if not clean or not all(arms.values()):
        print("missing cells; run C2 first")
        return

    def per_level_acc(rs, level):
        sub = [r for r in rs if r["level"] == level]
        return acc(sub) if sub else np.nan

    levels = [1, 2, 3, 4, 5]
    for arm, rs in arms.items():
        ratios = [per_level_acc(rs, l) / max(1e-9, per_level_acc(clean, l))
                  for l in levels]
        print(f"r_{arm} by level: " +
              " ".join(f"L{l}={r:.2f}" for l, r in zip(levels, ratios)))

    clean_by = {(r["problem_id"]): r["correct"] for r in clean}
    level_by = {r["problem_id"]: r["level"] for r in clean}
    problems = sorted(clean_by)
    rng = np.random.default_rng(0)
    diffs = []
    arm_correct = {a: {r["problem_id"]: r["correct"] for r in rs}
                   for a, rs in arms.items()}
    for _ in range(2000):
        sample = rng.choice(problems, len(problems))
        slopes = {}
        for a in arms:
            xs, ys = [], []
            for l in levels:
                pl = [p for p in sample if level_by.get(p) == l]
                cl = np.mean([clean_by[p] for p in pl]) if pl else np.nan
                ab = np.mean([arm_correct[a].get(p, 0) for p in pl]) if pl else np.nan
                if pl and cl > 0:
                    xs.append(l)
                    ys.append(ab / cl)
            slopes[a] = np.polyfit(xs, ys, 1)[0] if len(xs) >= 3 else np.nan
        diffs.append(slopes["jspace"] - slopes["random"])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    med = np.nanmedian(diffs)
    print(f"\nslope_J - slope_random: median={med:.4f}, CI95=[{lo:.4f}, {hi:.4f}]")
    if hi < 0:
        print("=> CONFIRM-breakdown: J-survival declines faster with difficulty")
    elif lo > 0:
        print("=> reverse: J-survival declines SLOWER than random (unexpected)")
    else:
        print("=> no significant slope difference "
              "(CONFIRM-persistence if r_J also flat/high; check table above)")


def c3(args) -> None:
    clean = rows("c3/aime_thinking_clean.jsonl")
    if clean:
        gate = acc(clean)
        print(f"AIME clean (hooked, capped) accuracy: {gate:.3f} "
              f"({'analyzable' if gate >= 0.20 else 'CENSORED ANCHOR (<0.20)'})")
    for arm in ("jspace", "random"):
        rs = rows(f"c3/aime_thinking_{arm}.jsonl")
        if rs and clean:
            print(f"  r_{arm} = {acc(rs) / max(1e-9, acc(clean)):.3f} "
                  f"(acc={acc(rs):.3f}, trunc="
                  f"{sum(r['truncated'] for r in rs) / len(rs):.2f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["h1i", "h1ii", "c3"])
    args = parser.parse_args()
    {"h1i": h1i, "h1ii": h1ii, "c3": c3}[args.cmd](args)


if __name__ == "__main__":
    main()
