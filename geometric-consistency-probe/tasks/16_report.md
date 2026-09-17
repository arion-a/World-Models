# Task 16 — Final research report

**Authoritative source: `research/CANONICAL_RESEARCH_PROTOCOL.md`, "TASK 16 — FINAL RESEARCH REPORT" section.** This file is generated deterministically from that section by `orchestrator/sync.py` -- hand edits here are not preserved across a sync; edit the canonical document instead. If the two ever disagree, the canonical document wins.

## Objective

**Purpose.** Produce a rigorous research report based only on actual experimental
evidence.

**Why this task exists.** Nine-plus experiments (Tasks 6–14, unified in Task 15) produce numbers,
not conclusions. Task 16 exists to synthesize them into a single
document that states exactly what was found, with the same evidentiary
discipline DESIGN.md §13 already applies to Tasks 1–5 — every claim
traceable to a specific metric in a specific task result, and no claim
stronger than what §1's six operationally-testable components actually
support.

**Relationship to previous tasks.** Reads (not re-derives) every result in `state/task_06_result.json`
through `state/task_15_result.json` (or their Task-15-unified
equivalents). **The report must be written FROM THE RESULTS** — it must
not be written from expected conclusions and then populated with
supporting evidence.

## Scientific question

**Scientific question.** What, precisely, can and cannot be concluded about geometric
consistency and physical-state accessibility in this frozen encoder's
representation, across everything tested in Tasks 6–15?

**Hypothesis.** Not applicable — this task synthesizes, it does not test a new
hypothesis.

**Mathematical formulation.** Not applicable, beyond restating §2's central formulation for context.

## Implementation requirements

**Inputs.** `state/task_06_result.json` through `state/task_15_result.json` (or
Task 15's unified equivalents), and any generated reports/plots from
those tasks.

**Outputs.** `reports/final_research_report.md`, `state/task_16_result.json`.

**Experimental protocol.** Not an experiment; a synthesis protocol:

1. Restate the research question and method summary, one paragraph per
   Task 6–15, each citing its actual result file.
2. Build a results table aggregating every task's headline metric(s)
   against its baselines.
3. Explicitly reconcile any tasks whose results conflicted (e.g. strong
   equivariance in Task 6/7 but weak accessibility in Task 8) rather
   than silently picking the more favorable one to headline.
4. Write the required structure in full: Abstract; Research question;
   Motivation; Experimental setup; Controlled 3D environment;
   Rendering; Representation extraction; Geometric transformations;
   Physical-state probes; Appearance controls; Baselines; Scale
   experiments; Temporal experiments; Occlusion experiments;
   Counterfactual experiments; Results; Negative results; Failure
   analysis; Alternative explanations; Limitations; Conclusions; Future
   work.
5. Build the claim-evidence matrix for every major claim: CLAIM →
   EXPERIMENT → METRIC → CONTROL → ACTUAL RESULT → LIMITATION.

**Dataset requirements.** Not applicable.

**Train/test protocol.** Not applicable — reports on splits already used by Tasks 6–15.

**Controls.** Not applicable as a new element — the report cites, rather than
re-runs, each source task's own controls.

**Baselines.** Every result reported must include its baseline comparison (Global
Invariant 19) — no result may be reported in this document without
also reporting its baseline comparison.

**Metrics.** Not applicable beyond citing each task's own metrics verbatim.

## Required artifacts

`reports/final_research_report.md` with the full required structure
above and the claim-evidence matrix; `state/task_16_result.json`
pointing at the report path and confirming every cited task result
file was actually read, not assumed.

## Tests required

**Required software tests.** A test that every `state/task_0{6..9}|1{0..5}_result.json` referenced
by the report actually exists, and that at least one number from each
is quoted correctly (a simple string/number match between the report
and the JSON source, to guard against transcription drift).

**Required scientific-validity tests.** - No numerical claim in the report lacks a metric citation.
- No claim in the report describes a result that isn't backed by a
  metric actually computed and recorded in the cited task's result
  JSON.
- Every result reported includes its baseline comparison.

**Required research-alignment checks.** - No unsupported "understanding" claims: the phrase "understands 3D"
  (or equivalent) must never appear without the specific operational
  metric immediately following it (Global Invariant 29's interpretation
  discipline, applied at the report level).
- Negative results included, not omitted to make the narrative more
  positive (Global Invariants 16, 26).

## Leakage checks

Not applicable (no new data/splits introduced).

## Acceptance criteria

**Failure conditions.** Any claim not directly traceable to a result file; any negative result
omitted; any use of "understands 3D" (or equivalent) without an
immediate operational metric; any fabricated value, graph, statistical
significance claim, successful-experiment claim, comparison, or missing
result; any uncertainty silently converted into confidence.

**Acceptance criteria.** 1. Report covers every task 6–15 with a traceable citation.
2. Explicit "supported" vs. "not supported" claims section present
   (mirroring DESIGN.md §13's structure).
3. No claim in the report lacks a metric citation.
4. Every numerical result traces to machine-readable output; plots
   correspond to actual data; claims correspond to experiments;
   limitations included; negative results included; report
   reproducibility holds.

## Prohibited shortcuts

Writing the report from expected conclusions and retrofitting evidence;
fabricating any value, graph, comparison, or missing result; omitting a
task's negative result to improve the overall narrative; using
"understands 3D" or equivalent language without an immediate
operational metric.

## Scientific interpretation limits

**Interpretation rules.** This report's claims are bounded by everything already stated in each
task's own "What this task DOES NOT establish" section — the report may
not claim more than the sum of its parts. If an experiment failed
technically, state that. If an experiment produced a negative
scientific result, state that. If evidence is inconclusive, state that.

**What a positive result means.** Not applicable in the aggregate — the report's job is to state
precisely which of §1's six operationally-testable components were
positively, negatively, or inconclusively supported, per task, not to
render one aggregate verdict.

**What a negative result means.** Not applicable in the aggregate, for the same reason.

**What this task does NOT establish.** Anything beyond what Tasks 6–15 actually measured — the report is not
license to extrapolate, round up uncertainty, or claim general
understanding of 3D structure by a video representation model.

---
