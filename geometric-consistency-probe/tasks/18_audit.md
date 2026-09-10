# Task 18 — Adversarial independent research audit

**Depends on:** The entire project, Tasks 1–17.

## Objective

An adversarial, independent pass over the whole project whose explicit
goal is to find leakage, confounds, or unsupported claims that Tasks
1–17's own QA missed — i.e. red-team the project's own science, not
just re-confirm it.

## Scientific question

Do any of the project's reported positive results survive an adversarial
attempt to break them? Concretely: can the auditor construct a
alternative, non-geometric explanation (appearance shortcut, scene-ID
leakage, rendering artifact, baseline miscalibration, split
contamination) for any headline result in
`reports/final_research_report.md`?

## Implementation requirements

1. Independently re-derive, from raw data (not from the reported
   numbers), at least one headline result from Task 6/7 and one from
   Task 8 — i.e. actually re-run the metric computation from the saved
   representations/manifests, rather than re-reading the JSON.
2. For every train/test split used anywhere in Tasks 6–14, independently
   re-verify disjointness and same-split-per-scene from the raw
   manifest/result files (do not trust each task's own leakage-check
   report; re-derive it).
3. Attempt at least one deliberate "attack": e.g. check whether a probe
   or `W_T`'s apparent performance survives when a scene-identity
   feature (not a physical-state feature) is substituted as the probe
   target — if a scene-ID-based control performs suspiciously close to
   the real probe, that's a leakage finding to report, not to bury.
4. Re-run every baseline for the audited results independently, using a
   different random seed than the original task used, and confirm the
   qualitative conclusion (learned map beats baselines, or does not) is
   seed-stable.
5. Review every task's "Prohibited shortcuts" section against what was
   actually implemented (read the code, not just the task's own
   self-report) and flag any shortcut that was taken anyway.

## Required artifacts

`reports/independent_audit.md`: every finding (confirmed, refuted, or
newly discovered issue), with severity, and — for anything found —
either a proposed fix or an explicit statement that a prior claim must
be downgraded/retracted. `state/task_18_result.json` summarizing pass/
fail per audited claim.

## Tests required

- The audit's own re-derivation code should itself have basic tests
  (it re-computes metrics; verify it reproduces a known, hand-computed
  toy example correctly before trusting it against real project data).

## Leakage checks

This entire task *is* a leakage check, applied adversarially across
Tasks 6–17.

## Acceptance criteria

1. At least one Task 6/7 result and one Task 8 result independently
   re-derived from raw data.
2. Seed-stability re-check performed for at least the flagship
   transform's headline result.
3. Scene-identity-substitution attack attempted and reported.
4. Every finding (positive or negative) recorded in
   `reports/independent_audit.md`, including "no issue found" for
   anything explicitly checked.

## Prohibited shortcuts

- Do not perform the audit by re-reading the same result JSON files and
  re-stating their conclusions — the entire point is independent
  re-derivation from more raw data.
- Do not soften or omit a finding that contradicts
  `reports/final_research_report.md`'s conclusions; the report must be
  corrected (with a recorded justification, invariant 15) if the audit
  overturns it, not the other way around.

## Scientific interpretation limits

This audit itself cannot prove the absence of all possible confounds —
only that the specific attacks attempted here did or didn't succeed.
Say so explicitly in `reports/independent_audit.md`'s own limitations
section.
