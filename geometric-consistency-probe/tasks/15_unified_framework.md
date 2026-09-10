# Task 15 — Unified evaluation framework

**Depends on:** Tasks 6–14's individual experiment scripts, metrics, and
result-JSON schemas.

## Objective

Consolidate the by-now-nine distinct experiment scripts (Tasks 6–14)
into one reusable evaluation harness with a single command that can run
any of them, and one shared result-JSON schema, so Task 16's report
generation (and any future re-runs / independent audit in Task 18)
don't have to special-case nine slightly different formats.

## Scientific question

None new — infrastructure consolidation, same category as Task 10.

## Implementation requirements

1. Define one shared result schema (Python dataclass or JSON Schema)
   that is a strict superset of every field Tasks 6–14 actually
   produced — do not drop fields to simplify; extend, using optional
   fields where a given task type doesn't apply (e.g. `motion_config`
   is `null` for non-temporal tasks).
2. Provide one CLI entry point, e.g. `python -m
   evaluation.run_task --task 6..14`, that dispatches to each task's
   existing experiment code (import and call, don't rewrite the
   experiment logic itself) and writes output through the shared schema.
3. Re-validate every existing `state/task_0{6..9}_result.json` /
   `state/task_1{0..4}_result.json` against the new shared schema —
   migrate the files in place if field names changed, and record that
   migration explicitly (this is a protocol change, invariant 15,
   even though it's just a schema/naming change, not a re-run).
4. Do not re-run Tasks 6–14's experiments to produce new numbers unless
   a genuine bug is found in the process — this task is about the
   harness, not about re-deriving results.

## Required artifacts

`evaluation/` module (or similarly named) with the shared schema +
dispatcher; `state/task_15_result.json` describing what was
consolidated and confirming all nine prior result files validate
against the new schema.

## Tests required

- Schema validation tests for all of Tasks 6–14's (migrated) result
  files.
- Regression (existing suite).

## Leakage checks

None new (no new experiments run).

## Acceptance criteria

1. Shared schema exists, is documented, and validates all of Tasks
   6–14's result files.
2. One CLI entry point can invoke each of Tasks 6–14's experiment code.
3. No prior task's recorded numbers were altered by this
   consolidation (only structure/field names, if anything, and that
   change is recorded).

## Prohibited shortcuts

- Do not silently change any previously recorded metric value while
  "unifying" format.
- Do not use this task as an opportunity to quietly redo a task whose
  result you'd like to be different — that requires its own
  invariant-15 justification, not a schema refactor pretext.

## Scientific interpretation limits

None — infrastructure only.
