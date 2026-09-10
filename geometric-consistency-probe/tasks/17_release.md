# Task 17 — Open-source release

**Depends on:** The entire repository as it stands after Task 16.

## Objective

Prepare the repository for external use and review: a clean, reproducible
setup path, license, and top-level documentation pointing readers to
`DESIGN.md`, `research/RESEARCH_PLAN.md`, and
`reports/final_research_report.md` in the right order.

## Scientific question

None — this is a release-readiness task, not an experiment. Its QA is
almost entirely SOFTWARE CORRECTNESS and REPRODUCIBILITY, not
scientific validity.

## Implementation requirements

1. Add a `LICENSE` file (ask the user which license if not already
   specified anywhere in the repo — do not assume one).
2. Verify `requirements.txt` is complete and pinned appropriately (the
   project already pins `bpy==5.0.1`; check `torch`/`transformers`
   version constraints are still accurate given Task 4/5's real
   pretrained-checkpoint work).
3. Verify a clean `git clone` + the documented `Setup` steps in
   `README.md` actually work end-to-end on a fresh checkout (run this
   for real, don't just re-read the instructions).
4. Add or verify a top-level "start here" pointer (in `README.md` or a
   new `CONTRIBUTING.md`) that sends a new reader to `DESIGN.md` first,
   then `research/RESEARCH_PLAN.md`, then the final report.
5. Confirm no credentials, API keys, or machine-specific paths are
   committed anywhere (grep the full history is out of scope; grepping
   the current tree is required).

## Required artifacts

`state/task_17_result.json` recording: license added, fresh-clone setup
verified (with the exact commands run and their exit codes), dependency
audit result, secret-scan result.

## Tests required

- An automated fresh-environment smoke test: create a new virtualenv,
  install `requirements.txt`, run `pytest -m "not slow"`, and record
  the outcome (this is the "did the documented setup actually work"
  check, done for real).

## Leakage checks

None applicable.

## Acceptance criteria

1. `LICENSE` present.
2. Fresh-clone setup verified end-to-end with recorded command output.
3. No secrets/credentials found in the current tree.
4. Top-level reading-order pointer present.

## Prohibited shortcuts

- Do not mark the fresh-clone verification complete without actually
  running it in an isolated environment.
- Do not pick a license without the user's explicit direction.

## Scientific interpretation limits

None — this task makes no scientific claims.
