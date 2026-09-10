# Task 10 — Baseline framework

**Depends on:** `baselines/identity_baseline.py`,
`baselines/shuffled_pairing_baseline.py`, Task 6's new
`baselines/mean_baseline.py`, and every experiment task's ad hoc use of
them so far (6–9).

## Objective

Consolidate the baseline/control logic that Tasks 6–9 each called
individually into one shared, tested framework, so later tasks
(11–18) invoke one function instead of re-wiring three baseline calls
by hand each time — and so the set of "mandatory baselines" (invariant
14) is enforced in code, not just convention.

## Scientific question

None new — this is an infrastructure-consolidation task, not a new
experiment. Its "scientific validity" check is really about whether it
preserves the exact semantics of the baselines it consolidates.

## Implementation requirements

1. Add a single entry point, e.g. `baselines/run_all_baselines(Z_train,
   Z_prime_train, Z_test, Z_prime_test, transform_name, alpha, seed) ->
   dict[str, BaselineResultLike]`, that runs the persistence, mean, and
   shuffled-pairing baselines and returns them uniformly.
2. Refactor Tasks 6–9's experiment code to call this single function
   instead of the three separate calls — a real refactor, not a
   parallel implementation left to bit-rot.
3. Preserve exact numerical behavior: after refactoring, Task 6–9's
   previously recorded baseline numbers must reproduce bit-for-bit (or
   within floating-point tolerance) when re-run with the same seed/data
   — this is the regression check for this task specifically.
4. Keep the three baseline *implementations* themselves in their
   existing files; this task is about the calling convention, not about
   rewriting `identity_baseline.py` et al.

## Required artifacts

`state/task_10_result.json`, `baselines/run_all_baselines` (or similarly
named) implemented and used by Tasks 6–9's code.

## Tests required

- A unit test that `run_all_baselines` returns exactly the three
  baselines (persistence, mean, shuffled-pairing) with the same field
  names/types as calling them individually.
- A regression test that re-running Task 6 (or a small synthetic
  stand-in with the same seed/inputs) through the new consolidated call
  reproduces its previously recorded metrics.

## Leakage checks

None new — this task must not change which data reaches which
baseline; a leakage regression here would mean the refactor introduced
a bug, and must fail QA immediately.

## Acceptance criteria

1. `run_all_baselines` (or equivalent) exists, tested, and used by
   Tasks 6–9's experiment code (not just added and left unused).
2. Numerical parity with pre-refactor baseline numbers (documented
   tolerance, e.g. `np.allclose`).
3. Full existing suite passes, including Tasks 6–9's own tests.

## Prohibited shortcuts

- Do not change any baseline's math while "consolidating" it — that
  would silently invalidate Tasks 6–9's already-recorded results.
- Do not leave the old per-task baseline-calling code in place
  unused alongside the new function ("two ways to do it" rot).

## Scientific interpretation limits

None — this task produces no new scientific claim. Its only claim is
"the baseline framework used by every later task computes the same
three controls, correctly, in one place."
