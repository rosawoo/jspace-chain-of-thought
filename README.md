# J-space and chain of thought

Does written chain of thought (CoT) substitute for a language model's internal
verbalizable workspace, and does that substitution protect reasoning when the
workspace is damaged? This repository tests whether the workspace-ablation
results of [Verbalizable Representations Form a Global Workspace in Language
Models](https://transformer-circuits.pub/2026/workspace/index.html) (Gurnee,
Sofroniew, et al., 2026) transfer to Qwen3-4B, and how the internal-external
trade-off behaves as problems get harder.

Headline result: a dissociation. The externalization mechanism is real and
directly observable (values load into the workspace before the model writes
them and release after, 93/116 intermediates, p < 1e-10), but the protective
consequence does not transfer (no ablation strength damages reasoning while
sparing general function, and CoT confers no J-specific protection over
direct answering). The full argument is in the report.

- Report: `report.pdf` (repo root)
- Frozen pre-registration: `HYPOTHESES.md`

## Repository structure

```
HYPOTHESES.md          pre-registered hypotheses, frozen before any run
README.md              this file
requirements.txt       python dependencies (plus jlens, installed from github)
src/
  jlens_fit.py         fit the Jacobian lens (paper appendix A.7 estimator)
  jlens_core.py        lens loading, transport, readout helpers
  ablate.py            the ablation engine: adaptive top-k J-direction removal
                       with sparing, span projection, random-token control
  eval_harness.py      resumable generation cells, paired seeds, scoring,
                       paired bootstrap
  gates.py             gate measurements (band metrics, readout control)
scripts/
  pod_setup.sh         one-time GPU pod environment setup
  runbook.md           step-by-step cloud execution log/playbook
  gates.py             run gates A/B and the operating-point sweep
  gate_d_ext.py        operating-point sweep extension (gentler strengths,
                       implementation checks, random-direction contrast)
  disambig.py          disambiguation rerun that resolved the skip-16 confound
  preship.py           teacher-forced damage table, capability battery,
                       renorm and random-token controls (report Table 3)
  m2a.py               Experiment 1: GSM8K direct vs CoT under ablation
  m2b.py               Experiment 2: readout-only telemetry (H5a, H5c)
  make_tables.py       recompute every number in the report's tables
  make_figures.py      regenerate the report's two figures
  analyze.py           survival-ratio summaries of generation cells
  lens_forensics.py    per-prompt lens diagnostics (found the two outliers)
  repair_lens.py       exact checkpoint subtraction of outlier prompts
tests/                 24 unit tests for the ablation engine and harness
results/
  gates/              gate outputs (gate_a/b/d.json, gate_d_ext.json,
                       disambig.json)
  preship/             teacher_forced.json, battery.json, transcripts
  m2a/                 per-problem JSONL for all six GSM8K cells
  m2b/                 part_a.jsonl (H5a), part_bc.jsonl (H5c)
report.pdf             the report
report/
  figures/             frontier.pdf, m2b.pdf (built by make_figures.py)
```

Intentionally not committed: `reference/` (read-only clones of the two
external repos), `*.pt` lens checkpoints (large binaries, refit exactly
with the commands below), run logs, and the report's LaTeX source and
style files (the compiled `report.pdf` is the deliverable).

## External resources

- Paper under investigation: https://transformer-circuits.pub/2026/workspace/index.html
- Official lens estimator (fitting code and the paper's eval prompt sets,
  which `src/gates.py` auto-downloads into `data/`):
  https://github.com/anthropics/jacobian-lens
- Third-party Qwen-scale precedent consulted during design:
  https://github.com/idhantgulati/j-lens
- Model: https://huggingface.co/Qwen/Qwen3-4B
- Datasets: [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
  [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500),
  [AIME 2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024),
  [AIME 2025](https://huggingface.co/datasets/yentinglin/aime_2025),
  lens fit prompts from
  [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- Compute: [RunPod](https://www.runpod.io) A100-80GB (~34 GPU-hours total)
- LaTeX compiler: [tectonic](https://tectonic-typesetting.github.io)

## Setup

Analysis and report regeneration need no GPU:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q          # 24 tests, all CPU
```

Full reproduction needs one A100-80GB (40GB works with a smaller
`--dim-batch`). On a fresh pod:

```bash
bash scripts/pod_setup.sh           # or manually:
pip install -r requirements.txt
pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4   # 12x speedup, see runbook.md
export HF_HOME=/workspace/hf_cache
```

## Experimental settings

All settings are pre-registered in `HYPOTHESES.md`, and every deviation
from them is reported where it matters in the report itself. The
load-bearing settings:

- Model `Qwen/Qwen3-4B`, bf16, sampled per the model card, never greedy
  (thinking mode temperature 0.6 and top-p 0.95, non-thinking 0.7 and 0.8).
  Direct-answer cells prefill the assistant turn with `\boxed{`.
- Lens: fit on 150 FineWeb prompts, penultimate-layer target; two heavy-tail
  outlier prompts removed by exact checkpoint subtraction (final n = 148).
- Workspace band: layers 12-28 of 36 (Gate A).
- Ablation: at every position and band layer, remove the k = 10 most
  strongly positively activated J-lens directions by exact span projection,
  sparing tokens in the clean model's top-10 next-token predictions,
  computed by a paired clean forward pass on the identical context.
- Control: the identical operator applied to randomly chosen vocabulary
  tokens' directions (operator-matched random-token control).
- Frozen thresholds: coherence gate 0.85 clean top-1 match, effect bar 0.50
  two-hop relative drop, H4 asymmetry 1.5x.
- Seeds are paired: each generation's seed depends on the problem and sample
  index, never on the intervention condition.

Naming note: the code's `gate-d` (operating-point sweep) is the report's
Gate C, and the code's `gate-c` is the H3 numeric-loading audit, not a
report gate.

## Reproduction pipeline (GPU, in order)

Everything writes to `results/`, is resumable, and checks provenance on
resume. `LENS=results/lens_qwen3-4b_n148_repaired.pt` below.

```bash
# 1. Fit the lens (~2-3 h) and repair the two outlier prompts (~1 min)
python -u -m src.jlens_fit --n-prompts 150 \
  --out results/lens_qwen3-4b_n150.pt --checkpoint results/fit_ckpt.pt
python scripts/lens_forensics.py --lens results/lens_qwen3-4b_n150.pt
python scripts/repair_lens.py --checkpoint results/fit_ckpt.pt \
  --outliers 24 139 --expected-norms 141.576 275.659 --out $LENS

# 2. Gates (~1 h). Gate A finds the band; the rest take its layer list.
python -u scripts/gates.py gate-a --lens $LENS
python -u scripts/gates.py gate-b --lens $LENS --band {12..28}
python -u scripts/gates.py gate-c --lens $LENS --band {12..28}  # H3 numeric audit
python -u scripts/gates.py gate-d --lens $LENS --band {12..28}  # report Gate C

# 3. Operating-point extension and the disambiguation rerun (~3 h)
python -u scripts/gate_d_ext.py --lens $LENS
python -u scripts/disambig.py --lens $LENS

# 4. Damage characterization at the strong point (~2 h, report Table 3)
python -u scripts/preship.py --lens $LENS

# 5. Experiment 1 (~12 h): six GSM8K cells, then the H4 analysis
python -u scripts/m2a.py run --lens $LENS
python scripts/m2a.py analyze

# 6. Experiment 2 (~4 h): readout-only telemetry for H5a and H5c
python -u scripts/m2b.py --lens $LENS
```

Every script accepts `--smoke` (or a small `--n`) for a minutes-long dry
run; we always smoke-tested before full runs.

## Regenerating the report's tables and figures (no GPU)

The committed `results/` files are the outputs of the runs above, so the
report is fully rebuildable without a GPU:

```bash
# Every number in Tables 1-3 plus the Figure 2 statistics, printed with
# provenance (accuracies, Wilson CIs, McNemar and sign tests, the
# pre-registered H4 bootstrap statistic on both problem sets):
.venv/bin/python scripts/make_tables.py

# The two figures (report/figures/frontier.pdf, m2b.pdf):
.venv/bin/python scripts/make_figures.py
```

Mapping to the report: `make_tables.py` prints Table 1 (GSM8K accuracy by
mode and intervention, `results/m2a/`), Table 2 (H5a bridge loading,
`results/m2b/part_a.jsonl`), Table 3 (damage at the strong operating point,
`results/preship/`), and the H5c statistics behind Figure 2
(`results/m2b/part_bc.jsonl`). `make_figures.py` builds Figure 1 from the
24-configuration sweep records (`results/gates/gate_d.json`,
`gate_d_ext.json`, `disambig.json`) and Figure 2 from
`results/m2b/part_bc.jsonl`.
