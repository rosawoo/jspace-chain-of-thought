"""Generate report figures from results/ JSONs, designed at their exact
print sizes for the ICLR layout (text width ~5.5in) so fonts stay readable.

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
GRAY, INK, MUTED = "#9c9b95", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": GRAY, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def load(path):
    full = os.path.join(R, path)
    return json.load(open(full)) if os.path.exists(full) else None


def fig1_frontier():
    """Effect vs coherence, all operating points. Print width 2.6in."""
    gate_d = load("gates/gate_d.json")
    ext = load("gates/gate_d_ext.json")
    disambig = load("gates/disambig.json")
    if not (gate_d and ext and disambig):
        return print("fig1: missing data, skipped")

    fig, ax = plt.subplots(figsize=(2.6, 2.9))
    # viable region with explicit dashed boundaries
    ax.axvline(0.85, color="#0e7a52", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0.50, color="#0e7a52", lw=0.8, ls="--", alpha=0.7,
               xmin=(0.85 - 0.35) / (1.03 - 0.35))
    ax.fill_between([0.85, 1.03], 0.50, 1.02, color=AQUA, alpha=0.12, lw=0)
    ax.text(0.94, 0.97, "viable\nregion", ha="center", va="top", fontsize=8,
            color="#0e7a52", fontweight="bold")

    coh = {"full-k10": 0.447, "win12-15-k10": 0.727}
    pts = ([(e["pretraining_top1_match"], e["twohop_rel_drop"])
            for e in gate_d["sweep"]]
           + [(e["top1"], e["twohop_rel_drop"]) for e in ext["stage2"]]
           + [(coh[e["config"]], e["twohop_rel_drop"]) for e in disambig])
    ax.scatter([x for x, _ in pts], [y for _, y in pts], marker="o", s=28,
               color=BLUE, alpha=0.85, zorder=3, edgecolors="none")

    # the dots are one experiment at different strengths: show the dose axis
    ax.annotate("heavier\nablation", xy=(0.47, 0.80), xytext=(0.68, 0.55),
                fontsize=7.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.text(0.365, 0.40, "strong damage,\nbroken coherence", fontsize=7.5,
            color=MUTED, style="italic", ha="left", va="top")
    ax.text(0.395, -0.125, "lighter: coherent, no effect", fontsize=7.5,
            color=MUTED, style="italic", ha="left")

    ax.set_xlabel("general coherence\n(clean-model top-1 match)")
    ax.set_ylabel("reasoning damage (two-hop rel. drop)")
    ax.set_xlim(0.35, 1.03)
    ax.set_ylim(-0.14, 1.02)
    ax.axhline(0, color=GRAY, lw=0.6)
    # single series: no legend needed; provenance lives in the caption
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "frontier.pdf"), bbox_inches="tight")
    print("fig1 frontier.pdf")


def fig4_m2b():
    """M2b telemetry, two panels. Print width 5.5in."""
    pa = os.path.join(R, "m2b", "part_a.jsonl")
    pbc = os.path.join(R, "m2b", "part_bc.jsonl")
    if not (os.path.exists(pa) and os.path.exists(pbc)):
        return print("fig4: m2b data not present, skipped")
    rows_a = [json.loads(l) for l in open(pa)]
    rows_bc = [json.loads(l) for l in open(pbc)]

    fig, ax2 = plt.subplots(figsize=(3.4, 2.5))

    pre, post = [], []
    for r in rows_bc:
        for l in r.get("lookahead") or []:
            if l and l.get("pre") is not None and l.get("post") is not None:
                pre.append(l["pre"])
                post.append(l["post"])
    diffs = [p_ - q for p_, q in zip(pre, post)]
    above = sum(1 for d in diffs if d > 0)
    ax2.hist(diffs, bins=18, color=ORANGE, alpha=0.85, edgecolor="white",
             linewidth=0.4)
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_xlabel("how much higher the value loads\nbefore writing than after (nats)")
    ax2.set_ylabel("number of intermediates")
    ax2.set_title("H5c: values load into the workspace\n"
                  "before they are written (n=116)", fontsize=8.5, loc="left")
    # interpretation lives in the report text, not on the plot
    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIG, "m2b.pdf"), bbox_inches="tight")
    print("fig4 m2b.pdf")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1_frontier()
    fig4_m2b()
