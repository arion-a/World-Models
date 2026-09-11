# Task 10 — Baseline framework

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 10 — BASELINE FRAMEWORK" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Determine whether observed representation behavior is meaningful
relative to reasonable alternative representations and trivial
controls, and consolidate this project's baseline logic into one
shared, tested framework.

**Why this task exists.** Tasks 6–9 each called baseline logic individually (persistence, mean,
shuffled-pairing) and DESIGN.md §12 additionally specifies two
baselines — the non-learned pixel-statistics encoder and the
matched-architecture randomly-initialized encoder — that have not yet
been wired into any Task 6–9 experiment. Task 10 exists both to
implement the full DESIGN.md §12 baseline set and to consolidate it
into one reusable, fairly-applied framework so Tasks 11+ invoke one
function instead of re-wiring baseline calls by hand each time, making
"mandatory baselines" (Global Invariant 18–19) enforced in code, not
just convention.

**Relationship to previous tasks.** Formalizes and extends `baselines/identity_baseline.py`,
`baselines/shuffled_pairing_baseline.py`, and Task 6's new
`baselines/mean_baseline.py`. Adds the two DESIGN.md §12 baselines not
yet implemented: the non-learned pixel-statistics encoder
(`encoders/pixel_baseline.py`, already scaffolded per `README.md`'s
"pixel_baseline" mention — inspected and reused, not reimplemented, if
it already exists) and the matched-architecture randomly-initialized
encoder (`VJEPAEncoder(pretrained=False)`, already supported by the
existing encoder interface). Refactors Tasks 6–9's experiment code to
call the new consolidated entry point.

## Scientific question

**Scientific question.** Does the primary representation demonstrate behavior that cannot be
explained by simple or alternative baselines?

**Hypothesis.** None new — this is an infrastructure-consolidation and
baseline-completion task, not a new experiment. Its scientific content
is whatever Tasks 6–9's *re-run* baseline comparisons reveal once the
full DESIGN.md §12 set is applied uniformly.

**Mathematical formulation.** No new formulation; formalizes DESIGN.md §12's four baselines
(identity/persistence, shuffled-pairing/random-pair, pixel-statistics
encoder, randomly-initialized encoder) plus Task 6's mean baseline, as
one uniformly-applied set.

## Implementation requirements

**Inputs.** Whatever `(Z_train, Z'_train, Z_test, Z'_test)` (or `(Z_train, y_train,
Z_test, y_test)` for Task 8-style probes) a calling task already
assembles.

**Outputs.** A single entry point, e.g. `baselines.run_all_baselines(Z_train,
Z_prime_train, Z_test, Z_prime_test, transform_name, alpha, seed) ->
dict[str, BaselineResultLike]`, running the full baseline set and
returning results uniformly; `state/task_10_result.json`.

**Experimental protocol.** 1. Implement (or confirm-and-wire, if partially present) all five
   baselines: persistence, mean, shuffled-pairing, pixel-statistics
   encoder, randomly-initialized encoder.
2. Every baseline must receive the identical scenes, split, labels,
   transformations, metrics, and evaluation protocol as the primary
   representation's result — no baseline may receive privileged
   ground-truth information the primary representation does not also
   have access to (e.g. the pixel-statistics baseline gets rendered
   pixels, not ground-truth `SceneState` coordinates).
3. Refactor Tasks 6–9's experiment code to call `run_all_baselines`
   instead of the three separate calls each currently makes — a real
   refactor, not a parallel implementation left to bit-rot.
4. Verify numerical parity: after refactoring, Tasks 6–9's previously
   recorded persistence/mean/shuffled-pairing numbers must reproduce
   (bit-for-bit or within documented floating-point tolerance) when
   re-run with the same seed/data through the new consolidated call.
5. Re-run Tasks 6–9's flagship comparisons with the two newly-added
   baselines (pixel-statistics, randomly-initialized encoder) and
   report the expanded comparison.

**Dataset requirements.** Reuses Tasks 6–9's existing datasets; no new scene generation required
unless a baseline's fair-comparison requirement cannot otherwise be met
(e.g. the pixel-statistics baseline needs the same rendered frames,
already available).

**Train/test protocol.** Unchanged from whichever task's data is being re-evaluated; this task
must not alter any train/test split, only the baseline-computation
pathway.

**Controls.** The baseline set itself *is* this task's set of controls, per Global
Invariant 18 ("controls must be implemented rather than merely
discussed").

**Baselines.** The full five: persistence, mean, shuffled-pairing/random-pair,
non-learned pixel-statistics encoder, matched-architecture
randomly-initialized encoder. A missing baseline in any downstream
task's use of this framework is an explicit failure, not a silent skip.

**Metrics.** Whatever metric the calling task specifies (R²/cosine/relative-L2 for
equivariance-style tasks; R²/MAE/RMSE for probe-style tasks like Task
8) — `run_all_baselines` must not hardcode one metric family.

## Required artifacts

`baselines/run_all_baselines` (or equivalently named) function,
implemented and used by Tasks 6–9's code; `state/task_10_result.json`.

## Tests required

**Required software tests.** - `run_all_baselines` returns exactly the five baselines with the same
  field names/types as calling them individually.
- A regression test that re-running Task 6 (or a small synthetic
  stand-in with the same seed/inputs) through the new consolidated call
  reproduces its previously recorded metrics.
- Every baseline executes without error on real data.
- No NaN/Inf; reproducibility.

**Required scientific-validity tests.** - Identical scene split, labels, and metrics confirmed across every
  baseline and the primary representation for a given comparison.
- No baseline receives privileged ground-truth information (an explicit
  check, e.g. the pixel-statistics baseline's inputs are traced back to
  rendered pixels only).
- Ask, explicitly, whether a trivial representation (pixel-statistics)
  can obtain performance comparable to the primary encoder — a required
  scientific-QA step, and if so, this must be reported prominently, not
  buried.

**Required research-alignment checks.** Global Invariants 18–19 (controls implemented, baselines evaluated
under the identical protocol) become mechanically checkable once
`run_all_baselines` exists — future orchestrator QA for Tasks 11+
should call it directly rather than re-deriving baseline logic.

## Leakage checks

None new (no new experiments run beyond what Tasks 6–9 already
established) — a leakage regression introduced by this refactor is
itself a failure to catch immediately.

## Acceptance criteria

**Failure conditions.** Any of: a baseline's math silently changed while "consolidating" it
(invalidating Tasks 6–9's already-recorded results); the old per-task
baseline-calling code left in place unused alongside the new function;
a baseline receiving privileged ground-truth information; a missing
baseline silently skipped instead of raising.

**Acceptance criteria.** 1. `run_all_baselines` (or equivalent) exists, tested, and used by
   Tasks 6–9's experiment code.
2. Numerical parity with pre-refactor baseline numbers (documented
   tolerance).
3. The two newly-added DESIGN.md §12 baselines (pixel-statistics,
   randomly-initialized encoder) are wired in and reported for at least
   Task 6/7's flagship comparison.
4. Full existing suite passes, including Tasks 6–9's own tests.

## Prohibited shortcuts

Changing any baseline's math while consolidating it; leaving duplicate
baseline-calling code paths; giving any baseline privileged information;
silently skipping a baseline instead of failing loudly.

## Scientific interpretation limits

**Interpretation rules.** Do not call the primary representation superior to a baseline unless
the comparison is fair (identical protocol) and the margin is reported
with its metric explicitly, not asserted qualitatively.

**What a positive result means.** The primary representation outperforms the full DESIGN.md §12 baseline
set (not just persistence/mean/random-pair) under a fair, identical
protocol — the single most important comparison this project can make,
per DESIGN.md §12 item 4's note on isolating pretraining's contribution.

**What a negative result means.** A baseline (especially the pixel-statistics or randomly-initialized
encoder) performs comparably to the primary representation — direct
evidence that Tasks 6–9's apparent findings are not specific to
pretrained V-JEPA representations, and must be reported as a major,
not minor, finding.

**What this task does NOT establish.** Any new scientific claim about geometric consistency itself — only
whether previously reported findings survive a fair, complete baseline
comparison.

---
