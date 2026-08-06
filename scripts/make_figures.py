"""Generate report figures from results/ JSONs. Skips figures whose data is
not yet present (rerun after M2 completes).

    .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#b0afa9", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": GRAY, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def load(path):
    full = os.path.join(R, path)
    return json.load(open(full)) if os.path.exists(full) else None


def fig1_frontier():
    """Selectivity frontier: two-hop effect vs coherence, all operating
    points, with the pre-registered viable quadrant."""
    gate_d = load("phase0/gate_d.json")
    ext = load("phase0/gate_d_ext.json")
    disambig = load("phase0/disambig.json")
    preship = load("preship/teacher_forced.json")
    if not (gate_d and ext and disambig):
        return print("fig1: missing data, skipped")

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.axvspan(0.85, 1.0, ymin=0.5, color="#1baf7a", alpha=0.10, lw=0)
    ax.text(0.925, 0.95, "pre-registered\nviable region", ha="center",
            va="top", fontsize=7, color="#0e7a52")

    for e in gate_d["sweep"]:
        ax.scatter(e["pretraining_top1_match"], e["twohop_rel_drop"],
                   marker="o", s=18, color=GRAY, zorder=3,
                   label="frozen sweep (abs semantics)"
                   if e is gate_d["sweep"][0] else None)
    for e in ext["stage2"]:
        ax.scatter(e["top1"], e["twohop_rel_drop"], marker="s", s=18,
                   color=BLUE, zorder=3,
                   label="extension (corrected, skip16)"
                   if e is ext["stage2"][0] else None)
    # disambiguation points (honest semantics) with preship coherence
    coh = {"full-k10": 0.447, "win12-15-k10": 0.727}
    for e in disambig:
        ax.scatter(coh[e["config"]], e["twohop_rel_drop"], marker="D", s=30,
                   color=ORANGE, zorder=4,
                   label="honest semantics" if e is disambig[0] else None)
        ax.annotate(e["config"], (coh[e["config"]], e["twohop_rel_drop"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.5,
                    color=MUTED)
    ax.set_xlabel("general coherence (pretraining top-1 match vs clean)")
    ax.set_ylabel("two-hop effect (relative drop)")
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(-0.15, 1.0)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left",
              bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "frontier.pdf"), bbox_inches="tight")
    print("fig1 frontier.pdf")


def fig2_specificity():
    """Pre-ship: (a) teacher-forced damage 1-hop vs 2-hop; (b) capability
    battery under the paper's own criterion."""
    tf = load("preship/teacher_forced.json")
    batt = load("preship/battery.json")
    if not (tf and batt):
        return print("fig2: missing data, skipped")

    def mean_dlogp(point, task):
        ref = tf["clean"][task]
        pt = tf["points"][point][task]
        return (sum(r["first_logprob"] for r in ref)
                - sum(r["first_logprob"] for r in pt)) / len(ref)

    points = ["full-k10", "full-k10-renorm", "win12-15-k10",
              "full-k10-randtok0", "full-k10-randtok1", "full-k10-randtok2"]
    labels = ["J-space k=10", "J-space + renorm", "thin window k=10",
              "random tokens s0", "random tokens s1", "random tokens s2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ys = range(len(points))
    for y, p in zip(ys, points):
        d2, d1 = mean_dlogp(p, "two_hop"), mean_dlogp(p, "one_hop")
        ax1.plot([d2, d1], [y, y], color=GRAY, lw=1, zorder=2)
        ax1.scatter([d2], [y], color=BLUE, s=22, zorder=3)
        ax1.scatter([d1], [y], color=ORANGE, s=22, zorder=3)
    ax1.set_yticks(list(ys), labels)
    ax1.invert_yaxis()
    ax1.axvline(0, color=GRAY, lw=0.6)
    ax1.set_xlabel(r"teacher-forced damage $\Delta$ log p(answer), nats")
    ax1.scatter([], [], color=BLUE, label="two-hop (93 items)")
    ax1.scatter([], [], color=ORANGE, label="one-hop control (30 items)")
    ax1.legend(frameon=False, fontsize=6.5, loc="lower right")

    arms = ["clean", "jspace", "randtok"]
    colors = [GRAY, BLUE, AQUA]
    vals = [batt[a]["mmlu100"] for a in arms]
    bars = ax2.bar(range(3), vals, color=colors, width=0.62)
    for b, v, a in zip(bars, vals, arms):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                 ha="center", fontsize=7, color=INK)
    ax2.set_xticks(range(3), ["clean", "J-space", "random\ntokens"])
    ax2.set_ylabel("MMLU-100 accuracy")
    ax2.set_ylim(0, 0.9)
    ax2.axhline(0.25, color=GRAY, lw=0.6, ls=":")
    ax2.text(2.45, 0.26, "chance", fontsize=6, color=MUTED, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "specificity.pdf"), bbox_inches="tight")
    print("fig2 specificity.pdf")


def fig3_m2a():
    """M2a H4: accuracy by mode x arm (full cells only)."""
    from src.eval_harness import summarize_cell

    cells = {}
    for mode in ("direct", "cot"):
        for arm in ("clean", "jspace", "randtok"):
            p = os.path.join(R, "m2a", f"gsm8k_{mode}_{arm}.jsonl")
            if os.path.exists(p):
                cells[(mode, arm)] = summarize_cell(p)
    if len(cells) < 6:
        return print(f"fig3: {len(cells)}/6 cells present, skipped")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    arms = ["clean", "jspace", "randtok"]
    colors = [GRAY, BLUE, AQUA]
    width = 0.26
    for i, (arm, c) in enumerate(zip(arms, colors)):
        xs = [0 + (i - 1) * width, 1 + (i - 1) * width]
        vals = [cells[("direct", arm)]["accuracy"],
                cells[("cot", arm)]["accuracy"]]
        ax.bar(xs, vals, width=width * 0.92, color=c,
               label={"clean": "clean", "jspace": "J-space ablated",
                      "randtok": "random-token control"}[arm])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.5,
                    color=INK)
    ax.set_xticks([0, 1], ["direct answer", "chain of thought"])
    ax.set_ylabel("GSM8K accuracy")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "m2a.pdf"), bbox_inches="tight")
    print("fig3 m2a.pdf")


def fig4_m2b():
    """M2b: (a) bridge loading direct vs cot; (b) lookahead pre/post."""
    pa = os.path.join(R, "m2b", "part_a.jsonl")
    pbc = os.path.join(R, "m2b", "part_bc.jsonl")
    if not (os.path.exists(pa) and os.path.exists(pbc)):
        return print("fig4: m2b data not present, skipped")
    rows_a = [json.loads(l) for l in open(pa)]
    rows_bc = [json.loads(l) for l in open(pbc)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    d = [(r["rank_direct"], r["rank_cot"]) for r in rows_a
         if r.get("rank_direct") and r.get("rank_cot")]
    ax1.scatter([x[1] for x in d], [x[0] for x in d], s=14, color=BLUE,
                alpha=0.75)
    lim = max(max(x) for x in d) * 1.5
    ax1.plot([1, lim], [1, lim], color=GRAY, lw=0.7, ls=":")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("bridge rank, CoT-invited prompt")
    ax1.set_ylabel("bridge rank, direct-demand prompt")
    ax1.text(0.05, 0.92, f"below diagonal = loaded harder\nunder direct demand"
             f"  (n={len(d)})", transform=ax1.transAxes, fontsize=6.5,
             color=MUTED)

    pre, post = [], []
    for r in rows_bc:
        for l in r.get("lookahead") or []:
            if l and l.get("pre") is not None and l.get("post") is not None:
                pre.append(l["pre"]); post.append(l["post"])
    if pre:
        ax2.scatter(post, pre, s=14, color=ORANGE, alpha=0.75)
        lo = min(min(pre), min(post)) - 1
        ax2.plot([lo, 1], [lo, 1], color=GRAY, lw=0.7, ls=":")
        ax2.set_xlabel("lens log p(value), 10 tokens after written")
        ax2.set_ylabel("lens log p(value), 10 tokens before")
        ax2.text(0.05, 0.92, f"above diagonal = loaded before\nexternalized"
                 f"  (n={len(pre)})", transform=ax2.transAxes, fontsize=6.5,
                 color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "m2b.pdf"), bbox_inches="tight")
    print("fig4 m2b.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1_frontier()
    fig2_specificity()
    fig3_m2a()
    fig4_m2b()
