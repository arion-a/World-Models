# evaluation/SCHEMA.md — the unified Task 6-14 result schema (Task 15)

Task 15 unifies Tasks 6-14's nine independently-evolved experiment
result formats into one shared schema, per
`research/CANONICAL_RESEARCH_PROTOCOL.md`'s "TASK 15 — UNIFIED
EVALUATION FRAMEWORK" section. This document, `evaluation/schema.py`,
and `evaluation/run_task.py` are Task 15's required artifacts.

## What the schema is

`evaluation.schema.validate_result(data, expected_task=None) ->
list[str]` (empty list = valid) checks a parsed result dict against
`REQUIRED_FIELDS`, the 9 top-level keys present in *every* one of
Tasks 6-14's existing `state/task_NN_result.json` files (verified
empirically, not assumed — see `evaluation/validate_existing_results.py`):

| Field | Type | Notes |
|---|---|---|
| `task` | `int` | must equal the file's own task number |
| `implementation_status` | `str` | `"COMPLETE"` or `"BLOCKED"`; `"BLOCKED"` additionally requires `blocker_category` (one of `SCIENTIFIC_CONFLICT`, `DESTRUCTIVE_ACTION`, `MISSING_DEPENDENCY`, `BLOCKER` — matching `orchestrator/classify.py`) and a non-empty `blocking_issue` |
| `seed` | `int` or non-empty `list[int]` | Task 11 records one seed per scale point, hence the list form |
| `config` | `dict` | task's own config, opaque to this schema |
| `metrics` | `dict` | task's own metrics, opaque to this schema |
| `scientific_result` | `str`, >= 10 chars | the task's prose finding |
| `software_versions` | `dict` with `python` key | provenance |
| `tests` | `dict` with `passed`/`failed`/`returncode` (`int`) | this task's own self-reported test run |
| `artifacts` | `list[str]` | paths, checked to exist by `orchestrator/qa.py` layer A |

Every other field any of Tasks 6-14 used (`transform`, `dataset`,
`encoder`, `motion_config`, `scale_ladder`, `occlusion_scenario`,
`counterfactual_conditions`, ... — the full list is
`evaluation.schema.KNOWN_OPTIONAL_FIELDS`) stays **optional**: this is
what "strict superset ... fields extended, never dropped" means in
practice. A result may also carry a field no prior task used at all —
that is not a schema violation either. The schema is an allow-list of
what's *known*, not a deny-list of anything else.

This intentionally does not use the `jsonschema` package: it is not in
`requirements.txt` and nothing else in this repository depends on it;
the checks above are the plain-Python equivalent of the JSON Schema
that would otherwise be written (`"required": [...]`,
`"additionalProperties": true`, per-field type constraints).

## Global Invariants 21-24 (seed, config, provenance, machine-readable)

`REQUIRED_FIELDS` covers all four directly: `seed` (21), `config` (22),
`software_versions` (23, provenance), and the fact that `validate_file`
itself confirms the result parses as JSON (24, machine-readable).
`evaluation/validate_existing_results.py` confirms this holds for every
one of Tasks 6-14's actual result files, not just in principle.

## Re-validating Tasks 6-14's existing results

`python -m evaluation.validate_existing_results` re-validates
`state/task_06_result.json` .. `state/task_14_result.json` against the
schema above and prints a per-task report (errors, and a
`sha256_before`/`sha256_after` pair proving the file was not modified
by validation). As of Task 15, **no field rename was needed in any of
the nine files** — every one already used the field names
`REQUIRED_FIELDS` lists, so no in-place migration happened and no
previously recorded metric value changed. If a future task's result
ever did require a rename to conform, that would be an explicit
protocol change (research/RESEARCH_INVARIANTS.md invariant 15 / Global
Invariant 27), not a silent rewrite — `validate_all()`'s `migrated`
flag exists to make exactly that case detectable.

`state/task_07b_result.json` is deliberately excluded: it's Task 7b's
own ad hoc protocol/state document (`status`/`protocol_hash`/
`history`/`updated_at`), not one of Tasks 6-14's canonical experiment
results — the task spec's own file list is `state/
task_0{6..9}_result.json` / `state/task_1{0..4}_result.json`.

## Running a task through the unified interface

```bash
python -m evaluation.run_task --task 6              # single task
python -m evaluation.run_task --task 6,8,10          # comma list
python -m evaluation.run_task --task 6-9             # range
python -m evaluation.run_task --task 6..14           # range, task spec's own notation
python -m evaluation.run_task --task 6 --config configs/experiments/task6_camera_rotation.yaml
```

`evaluation/run_task.py` imports the target task's existing
`experiments.taskNN_*` module and calls its existing `main()` with
`sys.argv` set to `["python -m experiments.taskNN_*", *extra_argv]` —
identical to invoking `python -m experiments.taskNN_*` directly, so no
task's experiment logic was rewritten to fit this dispatcher. Any
arguments after `--task` (`--config`, `--skip_tests`, `--stage`, ...)
are forwarded verbatim to that task's own argument parser. An unknown
task id, an empty selector, or a malformed range raises `ValueError`
immediately rather than silently skipping or guessing.
