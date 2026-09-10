# Task 16 — Final research report

**Depends on:** Every result in `state/task_06_result.json` through
`state/task_15_result.json`.

## Objective

Synthesize Tasks 6–15's findings into one final research report,
written to the same evidentiary standard as `DESIGN.md` §13 demands:
every claim traceable to a specific metric in a specific task result.

## Scientific question

What, precisely, can and cannot be concluded about geometric
consistency and physical-state accessibility in this frozen encoder's
representation, across everything tested?

## Implementation requirements

1. Produce `reports/final_research_report.md` (or similarly named),
   structured at minimum as: research question restated; method summary
   (one paragraph per task 6–15, each citing its actual result file);
   a results table aggregating every task's headline metric(s) against
   its baselines; a limitations section; and an explicit "claims
   supported" vs. "claims NOT supported" section, mirroring `DESIGN.md`
   §13's structure.
2. Every quantitative claim in the report must cite the exact
   `state/task_NN_result.json` field it comes from (a footnote/inline
   reference is enough — the point is traceability, not academic
   formatting).
3. Explicitly reconcile any tasks whose results conflicted (e.g. strong
   equivariance in Task 6/7 but weak accessibility in Task 8) rather
   than silently picking the more favorable one to headline.
4. No result may be reported without also reporting its baseline
   comparison, per invariant 14.

## Required artifacts

`reports/final_research_report.md`, `state/task_16_result.json`
(pointing at the report path and confirming every cited task result
file was actually read, not assumed).

## Tests required

- A test that every `state/task_0{6..9}|1{0..5}_result.json` referenced
  by the report actually exists and that at least one number from each
  is quoted correctly (a simple string/number match between the report
  and the JSON source, to guard against transcription drift).

## Leakage checks

None new.

## Acceptance criteria

1. Report covers every task 6–15 with a traceable citation.
2. Explicit "supported" vs. "not supported" claims section present.
3. No claim in the report lacks a metric citation.

## Prohibited shortcuts

- Do not write any claim not directly traceable to a result file.
- Do not omit a task's negative result to make the overall narrative
  more positive (invariant 11).
- Do not use the phrase "understands 3D" (or equivalent) without the
  specific operational metric immediately following it (invariant 13).

## Scientific interpretation limits

This report's claims are bounded by everything already stated in each
task's own "Scientific interpretation limits" section — the report may
not claim more than the sum of its parts.
