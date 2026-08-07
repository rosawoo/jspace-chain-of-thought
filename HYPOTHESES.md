# Pre-registered hypotheses

Every hypothesis in this file was recorded before the runs that test it,
in two stages. H1 through H3 and everything above the "Replacement
hypotheses" heading were frozen on 2026-08-02, before any lens fitting or
evaluation. H4 and H5 were frozen on 2026-08-04, after the operating-point
search showed H1 untestable as stated and before any run that tests them.

## Definitions

- **J-ablation**: at each decode step, for every layer in the workspace
  band and every position, zero the residual stream's projection onto the
  k most strongly activated J-lens vectors, sparing any vector whose token
  appears in the clean top-10 next-token predictions computed on the
  *identical current context* (paired per-step clean forward). k and band
  fixed by the operating-point sweep (Gate C).
- **Random control**: identical procedure, same k / band / positions, but
  ablating random directions scaled so the **removed norm matches** the
  J-ablation condition per (position, layer).
- **Survival ratio** r = (ablated accuracy) / (clean accuracy), computed
  per condition and difficulty stratum.
- **Coherence gate** (frozen thresholds): pretraining top-1 match
  ≥ 85% of clean on 50 held-out passages AND ≥ 90% well-formed outputs AND
  intact 20-prompt copy task.

## H1: Primary

At an ablation operating point that passes the coherence gate while
reducing verbal two-hop accuracy by ≥ 50%:

**(i) Replication.** On GSM8K, J-ablation causes at least 2× the relative
accuracy drop on direct answering compared to instructed CoT, and this
differential does **not** appear under the matched-removed-norm random
control (the random condition's drop ratio within paired-bootstrap noise
of 1).

**(ii) Difficulty interaction.** The CoT survival ratio r_J declines
across MATH-500 levels 1→5 (and to the AIME anchor, analyzed as a separate
regime) **significantly faster than r_random**, by logistic regression
`correct ~ condition × level`, clustered by question. That is, written CoT
partially substitutes for the internal workspace, but the substitution
degrades as per-step internal demand grows.

**Pre-registered disconfirming outcomes (all equally reportable):**

- The direct-vs-CoT differential appears equally under the random control:
  CoT robustness is generic robustness to perturbation, not workspace
  externalization.
- J-vs-random slope difference ≈ 0 with r_J flat and high through AIME:
  full interchangeability at all difficulties, the opposite result,
  reported as strengthening the paper's claim.
- The gates find no workspace band or no viable operating point:
  non-transfer / small-model fragility is the headline finding.

## H2: Sparing artifact (secondary)

The CoT-vs-direct differential of H1(i) survives at ≥ 50% of its magnitude
when the spare-clean-top-10 exemption is disabled. **Falsified if** the
differential exists only with sparing ON, which would reinterpret the
paper's GSM8K-CoT robustness as mechanical protection of about-to-be-emitted
tokens rather than workspace externalization.

## H3: Number-loading validity (secondary)

Numeric tokens (digits and number words) occupy ≥ 10% of in-band ablated
top-10 slots during direct math solving on Qwen3-4B. **Falsified if**
numeric occupancy is near zero (replicating the paper's number-word
caveat), in which case all math results are pre-scoped as tests of
plan/relational workspace content only, and the verbal-arithmetic isomorph
task (S4) is promoted into the core to carry the domain-general claim.

## Pre-registered exclusions (vacuous comparisons we will not run or claim)

1. Any direct-vs-CoT contrast on AIME (clean direct ≈ 0, so the conclusion
   is guaranteed either way).
2. Any accuracy claim at an ablation strength failing the coherence gate.
3. Borrowed-full-CoT "rescue" without the minus-last-step discriminating
   cell.
4. Survival-ratio decline with difficulty without the random-condition
   slope comparison.
5. Any CoT-robustness claim made solely with sparing ON and no
   exemption-OFF condition.

## Pre-registered cut order under time pressure

C2 random condition levels 1–3 → C1 n 200→150 → C3 random condition
samples 4→2. Never cut C3 clean or J conditions; never cut C4.

## Execution order (recorded 2026-08-02, before any run)

No hypothesis, threshold, or exclusion above changes. Execution is
sequenced into milestones with go/no-go gates; later-milestone tests are
deferred, not dropped:

- **M1** (first objective): lens fit → Gate A + Gate B → single
  operating-point check (full band, k=10, coherence + two-hop drop) →
  **C1-lite**: GSM8K {direct, cot} × {clean, jspace, random} → H1(i) test.
- **M2** (if M1 shows a controlled effect): C2 MATH-500 difficulty
  gradient → H1(ii) slope test.
- **M3** (if M2 is clean): C3 AIME anchor.
- **Deferred**: H2's sparing-OFF condition (C1 fourth cell), the
  standalone numeric audit (H3 is computed post hoc from M1 ablation logs,
  which record selected token ids), the full operating-point sweep (other
  bands and k only if the single operating point fails), the C4 battery
  beyond the M1 coherence check, and all stretch experiments (S1–S6).

Interpretation rule: H1(i) conclusions may be drawn from M1 alone; H1(ii)
requires M2 (plus the M3 anchor if analyzable); H2/H3 claims require their
deferred conditions to actually run.

## Replacement hypotheses H4 and H5 (recorded 2026-08-04, before their runs)

M1 concluded that no selective J-space ablation exists on Qwen3-4B within
the tested operator family: the strong-point ablation damages one-hop
recall MORE than two-hop reasoning and collapses MMLU. H1 as originally
operationalized is therefore untestable. The assignment question (does CoT
externalization protect against workspace damage, and does protection
erode with difficulty?) is re-operationalized in two designs that do not
require a selective ablation. Both were frozen before any M2 GPU run. In
the report, M2a is Experiment 1 and M2b is Experiment 2.

**H4 (M2a, ablation differential).** On GSM8K, the relative accuracy drop
under J-ablation (full-band k=10, honest semantics, vs clean) is at least
1.5× larger in DIRECT mode than in CoT mode, and this mode-asymmetry is
absent in the operator-matched random-token condition. Rationale: generic
knowledge damage (quantified: one-hop Δlogp +4.07) hits both modes; the
externalization account predicts additional CoT-specific protection.

- Disconfirming outcome (equally visible): mode-symmetric damage in the J
  condition, meaning written CoT provides no workspace-specific protection
  at 4B.
- Confound controls: paired seeds; the random-token condition; per-condition
  generation length, truncation, and degeneration reported; a secondary
  teacher-forced metric (answer logprob given the model's own generated
  prefix) to separate "cannot compute" from "generation decays over long
  outputs".

**H5 (M2b, ablation-free telemetry).** Measuring J-lens loading of each
problem's gold intermediate values (single-token numbers from the gold
rationale, excluding numbers in the question and the final answer):

- (a) at end-of-question positions, intermediate loading is HIGHER when the
  prompt demands a direct answer than when it invites CoT (workspace
  recruited on demand; the page will do the holding otherwise);
- (b) direct-mode loading grows with difficulty (GSM8K → MATH L1 → L3 → L5);
- (c) in CoT mode, intermediates load BEFORE they are written (lookahead),
  and per-step lookahead loading grows with difficulty, the original
  H1(ii) intuition measured correlationally.

Disconfirming outcomes: no mode difference at matched positions (loading is
stimulus-driven, not demand-driven, against the workspace account); flat
difficulty profiles (per-step internal demand does not grow).

Position-matching note: (a) compares IDENTICAL stimulus positions across
prompts differing only in instruction, avoiding the cross-stream occupancy
confound flagged by the design panel.

Scope: M2a on GSM8K only; M2b difficulty axis = GSM8K +
MATH-500 levels 1/3/5 (AIME excluded: no reliable gold intermediates).
AIME/M3 deferred unless H4/H5 produce a positive trade-off signal.
