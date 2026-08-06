# Cloud runbook — RunPod A100 (SXM fine; 80GB preferred, 40GB workable)

Execution follows HYPOTHESES.md **Amendment 1**: one milestone at a time, each
with a single objective and a go/no-go before the next. Everything is
checkpointed/resumable; pod preemption loses at most one prompt or generation.

## Pod setup (once)

1. RunPod → Deploy → **A100** (SXM or PCIe), template `runpod/pytorch`
   (CUDA 12.x, Python 3.11+). Attach a **network volume (≥100 GB)** at
   `/workspace` so lens, caches, and results survive restarts.
2. On the pod:

```bash
cd /workspace
git clone <YOUR_PRIVATE_REPO_URL> jspace-cot && cd jspace-cot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "jlens @ git+https://github.com/anthropics/jacobian-lens"
export HF_HOME=/workspace/hf_cache
python -m pytest tests/ -q        # must pass before continuing
```

---

## MILESTONE 1 — does CoT-protection replicate against a real control? (~6–8 GPU-h)

**Objective:** H1(i) only. GSM8K, {direct, cot} × {clean, jspace, random}.

### M1.1 Lens fit (~2–3 h, unattended)

```bash
nohup python -m src.jlens_fit --n-prompts 150 \
  --out results/lens_qwen3-4b_n150.pt \
  --checkpoint results/fit_ckpt.pt > fit.log 2>&1 &
tail -f fit.log   # max_d_mean falls ~1/n; < 5e-3 after ~20 prompts is healthy
```

Back up the lens off-pod when done.

### M1.2 Band + sanity checks (~1 h)

```bash
LENS=results/lens_qwen3-4b_n150.pt
python scripts/phase0.py gate-a --lens $LENS
# Band = contiguous layers with elevated kurtosis + autocorr_excess but
# next_token_agree not yet ~1 (that tail = motor regime). Ballpark: L12–28/36.
BAND="<layers from gate-a>"
python scripts/phase0.py gate-b --lens $LENS --band $BAND   # ≥12/20 to proceed
```

### M1.3 Operating-point check (single point, not the full sweep) (~0.5 h)

```bash
python scripts/phase0.py gate-d --lens $LENS --band $BAND
# We only need ONE viable row (coherent + two-hop drop ≥50%). k=10 full band
# is the default expectation; the sweep's other rows are just printed context.
```

Record band + k in `results/phase0/DECISION.md`. **Stop rules:** Gate A/B fail
→ logit-lens arbiter check, then non-transfer headline. No viable operating
point → fragility headline. (HYPOTHESES.md; do not loosen thresholds.)

### M1.4 C1-lite (~2–3 h)

```bash
K=10   # from M1.3
python scripts/run_core.py c1 --lens $LENS --band $BAND --k $K
python scripts/run_core.py summarize
python scripts/analyze.py h1i
```

### M1.5 Go/no-go

- Controlled effect (J-drop-ratio ≥ ~2, random ≈ 1) → **M2**.
- No differential, or random shows the same → that's the finding; write it up
  (H1(i) branch), decide with fresh eyes whether M2 still adds value.
- Post-hoc (free, no GPU): numeric fraction of ablated slots from C1 direct
  jspace logs (`ablation_stats.top_selected`) → provisional H3 read.

---

## MILESTONE 2 — does protection erode with difficulty? (~3–4 GPU-h)

Only after M1 go. C2: MATH-500 stratified, CoT, 3 arms (+ direct on L1–2).

```bash
python scripts/run_core.py c2 --lens $LENS --band $BAND --k $K
python scripts/analyze.py h1ii
```

## MILESTONE 3 — the hard-end anchor (~4–6 GPU-h MAX, pre-budgeted)

Only after M2. C3: AIME 2024+2025, thinking mode, 16k cap, 4 samples.

```bash
python scripts/run_core.py c3 --lens $LENS --band $BAND --k $K
python scripts/analyze.py c3   # analyzable only if clean ≥ 0.20
```

## DEFERRED (run only if/when decided)

- H2 sparing-OFF arm:
  `python scripts/run_core.py c1 --lens $LENS --band $BAND --k $K --arms jspace-nospare`
- Standalone Gate C: `python scripts/phase0.py gate-c ...`
- C4 full battery: `python scripts/run_core.py c4 ...`
- Stretch S1–S6 (see judge program in the plan).

## Cost tracker

| Step | Est. GPU-h | Actual |
|---|---|---|
| M1 fit | 2–3 | |
| M1 checks | 1–1.5 | |
| M1 C1-lite | 2–3 | |
| M2 | 3–4 | |
| M3 | 4–6 | |
