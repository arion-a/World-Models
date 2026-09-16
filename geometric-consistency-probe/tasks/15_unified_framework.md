# Task 15 — Unified evaluation framework

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 15 — UNIFIED EVALUATION FRAMEWORK" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Unify Tasks 6–14 into a coherent, reproducible evaluation framework.

**Why this task exists.** By this point, nine distinct experiment scripts (Tasks 6–14) exist,
each with its own slightly different result-JSON shape. Task 16's
report generation and Task 18's adversarial audit both need to consume
all nine consistently; without unification, both would have to
special-case nine formats, multiplying the chance of a transcription or
interpretation error entering the final claims.

**Relationship to previous tasks.** Unifies configuration, datasets, scene splits, transformations,
representation extraction, probes, baselines, metrics, controls, result
storage, and provenance across Tasks 6–14, extending `configs.config`'s
existing dataclass-based, YAML-loaded pattern rather than introducing a
new configuration mechanism. Does **not** re-run Tasks 6–14's
experiments to produce new numbers unless a genuine bug is found in the
process — this task is about the harness, not about re-deriving
results.

## Scientific question

**Scientific question.** None new — the individual experiments should become components of a
consistent measurement framework rather than unrelated scripts.

**Hypothesis.** None — infrastructure task.

**Mathematical formulation.** None new.

## Implementation requirements

**Inputs.** Tasks 6–14's existing result JSON files and experiment code.

**Outputs.** One shared result schema (Python dataclass or JSON Schema) that is a
strict superset of every field Tasks 6–14 actually produced (fields
extended, never dropped; optional fields for task-type-specific data
like `motion_config`); one CLI entry point (e.g. `python -m
evaluation.run_task --task 6..14`) that dispatches to each task's
existing experiment code; `state/task_15_result.json`.

**Experimental protocol.** 1. Define the shared schema as a strict superset of Tasks 6–14's
   existing fields.
2. Build one CLI entry point that imports and calls each task's
   existing experiment logic (no logic rewrite).
3. Re-validate every existing `state/task_0{6..9}_result.json`/
   `state/task_1{0..4}_result.json` against the new shared schema;
   migrate the files in place if field names changed, recording that
   migration explicitly as a protocol change (Global Invariant 27) even
   though it is a schema/naming change, not a re-run.
4. Experiments must be runnable through configuration rather than
   manual source-code editing from this point forward.

**Dataset requirements.** None new.

**Train/test protocol.** Unchanged from each unified task; this task must not alter any prior
split.

**Controls.** None new; every experiment's own controls are preserved unchanged
through the unification.

**Baselines.** None new; every experiment's own baselines are preserved unchanged
through the unification.

**Metrics.** None new; every experiment's own metrics are preserved unchanged.

## Required artifacts

`evaluation/` module (or similarly named) with the shared schema and
dispatcher; `state/task_15_result.json` confirming all nine prior
result files validate against the new schema.

## Tests required

**Required software tests.** - Schema validation tests for all of Tasks 6–14's (migrated) result
  files.
- Tasks 6–14 invocable through the unified interface; configuration
  parsing works; invalid configuration fails loudly (not silently);
  split validation; provenance fields present; unique experiment
  identification; no hidden defaults.
- Regression tests for Tasks 6–14 (their own prior tests still pass).
- No NaN/Inf; reproducibility.

**Required scientific-validity tests.** Confirm no previously recorded metric value was altered by this
consolidation (only structure/field names, if anything, and that
change is recorded) — a direct byte/value comparison between
pre-migration and post-migration result files for every unchanged
field.

**Required research-alignment checks.** Every one of Global Invariants 21–24 (seed, config, provenance,
machine-readable) must now be satisfiable via the unified schema for
every task, not just some.

## Leakage checks

None new (no new experiments run); a leakage regression introduced by
this refactor is itself an immediate failure.

## Acceptance criteria

**Failure conditions.** Any previously recorded metric value silently altered; a task's
experiment logic rewritten rather than wrapped; a methodological change
introduced without explicit documentation (Global Invariant 27); this
task used as a pretext to quietly redo a task whose result the operator
would prefer to be different.

**Acceptance criteria.** 1. Shared schema exists, is documented, and validates all of Tasks
   6–14's result files.
2. One CLI entry point can invoke each of Tasks 6–14's experiment code.
3. No prior task's recorded numbers were altered by this consolidation
   (only structure/field names, if anything, and that change is
   recorded).

## Prohibited shortcuts

Silently changing any previously recorded metric value while
"unifying" format; using this task as a pretext to quietly redo a task
whose result is undesired.

## Scientific interpretation limits

**Interpretation rules.** None — infrastructure only; no new scientific claim is made by this
task.

**What a positive result means.** Not applicable (infrastructure task) — "success" here means the unified
framework faithfully reproduces every prior task's recorded result
under a common interface.

**What a negative result means.** Not applicable in the scientific sense; a discovered discrepancy
between pre- and post-unification numbers is a software failure to
fix, not a scientific negative result.

**What this task does NOT establish.** Any new scientific claim about geometric consistency, accessibility,
invariance, scale, temporal consistency, occlusion, or counterfactual
structure — all such claims remain exactly as established (or not) by
Tasks 6–14 individually.

---
