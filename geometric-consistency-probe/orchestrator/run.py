"""Orchestrator CLI entry point.

    python -m orchestrator.run --dry-run     # show state/plan, change nothing
    python -m orchestrator.run               # actually run the next task
    python -m orchestrator.run --loop        # run tasks back-to-back while
                                              # the canonical protocol already
                                              # covers the next task number
    python -m orchestrator.run --reset-task 6  # explicit operator override
                                              # to retry a BLOCKED/FAILED task

Flow per task (see the module docstrings of state.py / sync.py / qa.py /
classify.py / prompts.py / claude_client.py / git_ops.py for the pieces
this wires together):

    load state -> determine next task
    -> SYNC: reconcile tasks/{NN}_*.md with research/
       CANONICAL_RESEARCH_PROTOCOL.md (deterministic, no judgment involved
       -- see orchestrator/sync.py)
    -> require a clean git tree -> RUNNING: invoke Claude with the task
       prompt (built from the canonical protocol, invariants, synced spec,
       current state, and the previous task's result)
    -> QA: independently verify (never trust Claude's own report)
    -> PASSED -> git checkpoint commit -> advance_task() -> next task's
       SYNC, automatically, if --loop
       QA FAILED and the task itself reported a genuine
           SCIENTIFIC_CONFLICT/DESTRUCTIVE_ACTION/MISSING_DEPENDENCY
           blocker (orchestrator/classify.py) -> BLOCKED immediately,
           no repair attempt wasted on something retrying cannot fix
       QA FAILED (ordinary implementation/methodological failure) and
           attempts remain -> FIXING: invoke Claude with a fix prompt,
           back to RUNNING
       QA FAILED and attempts exhausted -> BLOCKED, stop

None of this asks a human anything. The only way this stops short of
Task 18 completing is: (a) the canonical protocol has no section yet for
the next task number (nothing to derive a task from), (b) a task
reports a genuine SCIENTIFIC_CONFLICT/DESTRUCTIVE_ACTION/
MISSING_DEPENDENCY blocker, or (c) repair attempts are exhausted on an
ordinary failure. All three land in BLOCKED or a clean early return;
none of them is "ask the user for a routine decision."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from orchestrator import classify, claude_client, git_ops, qa, state, sync
from orchestrator.prompts import render_fix_prompt, render_previous_result_summary, render_state_summary, render_task_prompt, render_task_prompt_from_content

DEFAULT_MAX_FIX_ATTEMPTS = int(os.environ.get("MAX_FIX_ATTEMPTS", "3"))


def _repo_root() -> Path:
    # orchestrator/run.py lives at <repo_root>/orchestrator/run.py
    return Path(__file__).resolve().parent.parent


def _paths(repo_root: Path) -> dict[str, Path]:
    return {
        "state": repo_root / "state" / "progress.json",
        "tasks": repo_root / "tasks",
        "logs": repo_root / "logs",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous task orchestrator for the Geometric Consistency Probe.")
    parser.add_argument("--dry-run", action="store_true", help="Show state/plan without changing anything.")
    parser.add_argument("--loop", action="store_true", help="After a task passes, continue to the next one while the canonical protocol already covers it.")
    parser.add_argument("--allow-dirty", action="store_true", help="Proceed even if the git working tree has uncommitted changes.")
    parser.add_argument("--max-fix-attempts", type=int, default=DEFAULT_MAX_FIX_ATTEMPTS, help=f"Max repair attempts per task (default: {DEFAULT_MAX_FIX_ATTEMPTS}, env MAX_FIX_ATTEMPTS).")
    parser.add_argument("--repo-root", type=str, default=None, help="Repository root (default: the geometric-consistency-probe/ directory containing this package).")
    parser.add_argument("--reset-task", type=int, default=None, help="Explicit operator override: reset a BLOCKED/FAILED task back to READY (does not touch its files).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    paths = _paths(repo_root)

    try:
        current_state = state.load(paths["state"])
    except state.StateError as exc:
        print(f"FATAL: {paths['state']} is malformed and will not be auto-repaired: {exc}", file=sys.stderr)
        return 2

    if args.reset_task is not None:
        try:
            current_state.reset_task(args.reset_task)
        except state.StateError as exc:
            print(f"FATAL: cannot reset task {args.reset_task}: {exc}", file=sys.stderr)
            return 2
        state.save(current_state, paths["state"])
        print(f"Task {args.reset_task} reset to READY.")
        return 0

    if args.dry_run:
        return _dry_run(current_state, repo_root, paths, args)

    return _run_loop(current_state, repo_root, paths, args)


def _load_previous_result(repo_root: Path, task: int) -> tuple[int, dict | None]:
    prev_task = task - 1
    if prev_task < 1:
        return prev_task, None
    path = repo_root / "state" / f"task_{prev_task:02d}_result.json"
    if not path.exists():
        return prev_task, None
    try:
        return prev_task, json.loads(path.read_text())
    except json.JSONDecodeError:
        return prev_task, None


def _dry_run(current_state: state.OrchestratorState, repo_root: Path, paths: dict, args) -> int:
    print("=== DRY RUN (no changes will be made) ===")
    print(f"repo_root: {repo_root}")
    print(f"state file: {paths['state']} ({'exists' if paths['state'].exists() else 'will be bootstrapped'})")
    print(f"current status: {current_state.status}")
    print(f"completed_tasks: {current_state.completed_tasks}")
    print(f"current_task: {current_state.current_task}")

    if current_state.status in (state.BLOCKED, state.FAILED):
        blocker = current_state.blockers.get(str(current_state.current_task))
        print(f"Task {current_state.current_task} is {current_state.status} -- orchestrator would refuse to proceed without --reset-task.")
        if blocker:
            print(f"  blocker category: {blocker['category']} -- {blocker['reason']}")
        return 0

    next_task = current_state.next_task()
    if next_task is None:
        print("All tasks (1-18) complete. Nothing to do.")
        return 0

    print(f"next task: {next_task}")
    # dry_run=True: report what sync WOULD do without writing anything --
    # a genuinely stale/missing tasks/{NN}_*.md is exactly the kind of
    # fact --dry-run exists to surface, not silently fix on disk.
    try:
        sync_result = sync.sync_task_spec(next_task, repo_root, dry_run=True)
    except sync.CanonicalSectionNotFound as exc:
        print(f"Canonical protocol has no section for task {next_task} yet: {exc}")
        print("Orchestrator would stop here and wait for the canonical protocol to be extended.")
        return 0
    print(f"task spec sync: {'would update' if sync_result.changed else 'already up to date'} ({sync_result.path})")

    prev_task, prev_result = _load_previous_result(repo_root, next_task)
    prompt = render_task_prompt_from_content(
        next_task,
        sync_result.path.name,
        sync_result.content,
        render_state_summary(current_state.to_dict(), next_task, 1, args.max_fix_attempts),
        render_previous_result_summary(prev_result, prev_task),
        prev_task,
    )

    print(f"rendered prompt: {len(prompt)} chars (spec file located)")
    claude_config = claude_client.ClaudeConfig.from_env()
    argv = claude_client.build_command(prompt, claude_config)
    print(f"planned Claude command: {claude_client.redact_prompt_in_argv(argv, prompt)}")
    print(f"platform detected: {claude_client.detect_platform()}")

    q_config = qa.QAConfig()
    print("planned QA layers:")
    for name in (
        "A. SOFTWARE CORRECTNESS",
        "B. TASK ACCEPTANCE",
        "C. SCIENTIFIC VALIDITY",
        "D. RESEARCH ALIGNMENT",
        "E. DATA LEAKAGE",
        "F. REPRODUCIBILITY",
        "G. REGRESSION",
    ):
        print(f"  - {name}")
    print(f"test command QA will run: {list(q_config.test_command)}")
    print(f"max fix attempts: {args.max_fix_attempts}")
    print("blocker categories that skip remaining repair attempts (no human question, straight to BLOCKED):")
    print(f"  {list(classify.STOP_REPAIR_CATEGORIES)}")

    try:
        git_status = git_ops.get_status(repo_root)
        print(f"git: branch={git_status.branch} commit={git_status.commit[:12]} clean={git_status.clean}")
        if not git_status.clean and not args.allow_dirty:
            print("NOTE: working tree is dirty; a real run would refuse to start without --allow-dirty.")
    except git_ops.GitError as exc:
        print(f"git status check failed: {exc}")

    return 0


def _run_loop(current_state: state.OrchestratorState, repo_root: Path, paths: dict, args) -> int:
    while True:
        result = _run_one_task(current_state, repo_root, paths, args)
        if result != 0:
            return result
        if not args.loop:
            return 0
        if current_state.is_complete():
            print("All tasks (1-18) complete.")
            return 0
        # Only continue the loop while the canonical protocol already
        # covers the next task number -- otherwise stop cleanly and wait.
        # This is the one remaining "wait" condition: there is nothing to
        # derive a task from until its scientific design exists in the
        # canonical protocol (research/CANONICAL_RESEARCH_PROTOCOL.md's
        # own "IMPORTANT DISTINCTION" -- inventing scope is never routine).
        try:
            sync.sync_task_spec(current_state.next_task(), repo_root)
        except sync.CanonicalSectionNotFound:
            print(f"--loop: research/CANONICAL_RESEARCH_PROTOCOL.md has no section for task {current_state.next_task()} yet. Stopping and waiting.")
            return 0


def _run_one_task(current_state: state.OrchestratorState, repo_root: Path, paths: dict, args) -> int:
    if current_state.status in (state.BLOCKED, state.FAILED):
        blocker = current_state.blockers.get(str(current_state.current_task))
        print(f"Task {current_state.current_task} is {current_state.status}. Refusing to proceed automatically.")
        if blocker:
            print(f"  blocker category: {blocker['category']} -- {blocker['reason']}")
        print(f"Use `python -m orchestrator.run --reset-task {current_state.current_task}` after investigating, to retry.")
        return 1

    task = current_state.next_task()
    if task is None:
        print("All tasks (1-18) complete. Nothing to do.")
        return 0

    # A dirty tree is a DESTRUCTIVE-ACTION-adjacent risk (a checkpoint's
    # `git add -A` could fold in unrelated in-progress work), so this
    # check must happen before anything else -- including before the
    # otherwise-routine SYNC_SPECIFICATION step below -- and nothing may
    # be persisted to state/progress.json if it fails. Only relevant on
    # a fresh (READY) start: a resume of RUNNING/QA/FIXING legitimately
    # has the interrupted attempt's uncommitted changes still present.
    if current_state.status == state.READY:
        try:
            if args.allow_dirty:
                git_ops.get_status(repo_root)
            else:
                git_ops.require_clean_tree(repo_root)
        except git_ops.GitError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2

    # SYNC_SPECIFICATION: a routine, deterministic, non-interactive step
    # -- reconcile tasks/{NN}_*.md with the canonical protocol before
    # doing anything else. This is category A (ROUTINE DERIVABLE
    # DECISION) in the canonical protocol's own terms: never ask a human.
    try:
        sync_result = sync.sync_task_spec(task, repo_root)
    except sync.CanonicalSectionNotFound as exc:
        print(f"Task {task}: {exc}")
        print("Stopping and waiting for the canonical protocol to be extended -- state is unchanged.")
        return 0
    if sync_result.changed:
        print(f"Task {task}: synchronized {sync_result.path} with the canonical protocol.")
    current_state.record_sync(task, sync_result.changed, str(sync_result.path), sync_result.title)
    state.save(current_state, paths["state"])

    paths["logs"].mkdir(parents=True, exist_ok=True)
    log_path = paths["logs"] / f"task_{task:02d}.log"
    qa_log_path = paths["logs"] / f"task_{task:02d}_qa.log"
    result_path = repo_root / "state" / f"task_{task:02d}_result.json"
    claude_config = claude_client.ClaudeConfig.from_env()
    q_config = qa.QAConfig()
    max_attempts = args.max_fix_attempts

    # Resuming after an interruption that happened between QA passing
    # and the checkpoint/advance completing: QA already passed (it's
    # recorded in last_qa), so just finish checkpointing -- do not
    # re-invoke Claude or re-run QA for a task that already passed it.
    if current_state.status == state.PASSED:
        print(f"Resuming task {task}: QA already PASSED before the previous run was interrupted. Finishing checkpoint...")
        return _finish_passed_task(current_state, repo_root, paths, task)

    if current_state.status == state.READY:
        current_state.transition(state.RUNNING, task=task)
        state.save(current_state, paths["state"])
    else:
        # Resuming mid-attempt (RUNNING/QA/FIXING): the working tree is
        # expected to already carry the interrupted attempt's uncommitted
        # changes -- do not require a clean tree here, that would wrongly
        # reject a legitimate resume. No checkpoint commit has happened
        # yet in any of these statuses, so HEAD is still the correct
        # pre-task commit for QA's regression diff.
        print(f"Resuming task {task} from status {current_state.status} (previous run was interrupted).")

    try:
        pre_task_commit = git_ops.get_status(repo_root).commit
    except git_ops.GitError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    prev_task, prev_result = _load_previous_result(repo_root, task)
    attempts_already_made = current_state.attempts.get(str(task), 0)
    last_qa_dict = current_state.last_qa.get(str(task))

    # Resuming exactly mid-QA (interrupted after Claude's attempt already
    # finished and status moved to QA, but before qa.run_qa completed):
    # that attempt's work is already sitting in the working tree, so the
    # first loop iteration must re-run QA for THAT SAME attempt number
    # instead of invoking Claude again for a new one -- both because it
    # would waste an attempt with no QA report to build a fix prompt
    # from, and because a status of QA cannot legally transition to QA
    # again or to FAILED (ALLOWED_TRANSITIONS[QA] is only {PASSED,
    # FIXING, BLOCKED}) -- invoking Claude here and hitting a
    # ClaudeInvocationError used to crash with a StateError for exactly
    # this reason.
    #
    # Known limitation: status==QA on resume is also left behind by a
    # DIFFERENT case this can't distinguish from the one above -- a
    # later attempt's own invoke_claude call raising (e.g. a timeout)
    # before it could ever transition status away from QA, so nothing
    # from that attempt actually landed in the tree. Skip-and-verify
    # then just re-checks whatever the PRIOR attempt left behind, which
    # can only under-count progress (relabeling a redo as the same
    # attempt number) or fail QA again -- both fall through to the
    # ordinary fix-prompt retry loop just like any other QA failure, so
    # this never produces an incorrect PASS, only a possibly-wasted
    # iteration within the existing attempt budget. Precisely
    # distinguishing the two would need a same-attempt "Claude actually
    # produced new output" signal this state machine doesn't track;
    # not worth the added complexity for what degrades gracefully.
    resuming_mid_qa = current_state.status == state.QA
    loop_start = attempts_already_made if resuming_mid_qa else attempts_already_made + 1

    for attempt in range(loop_start, max_attempts + 1):
        skip_invoke = resuming_mid_qa and attempt == loop_start
        resuming_mid_qa = False  # only the first iteration can skip

        if not skip_invoke:
            if current_state.status == state.FIXING:
                current_state.transition(state.RUNNING, task=task)
                state.save(current_state, paths["state"])

            current_state.record_attempt(task)
            state.save(current_state, paths["state"])

            if attempt == 1:
                prompt = render_task_prompt(
                    task,
                    sync_result.path,
                    render_state_summary(current_state.to_dict(), task, attempt, max_attempts),
                    render_previous_result_summary(prev_result, prev_task),
                    prev_task,
                )
            else:
                prior_report = qa.render_report(last_qa_dict) if last_qa_dict else "(no prior QA report available -- resumed after interruption)"
                prompt = render_fix_prompt(task, sync_result.path, prior_report, attempt, max_attempts)

            print(f"Task {task}, attempt {attempt}/{max_attempts}: invoking Claude...")
            try:
                claude_result = claude_client.invoke_claude(prompt, repo_root, config=claude_config, log_path=log_path)
            except claude_client.ClaudeInvocationError as exc:
                print(f"FATAL: Claude invocation failed: {exc}", file=sys.stderr)
                current_state.transition(state.FAILED, task=task, reason=str(exc))
                state.save(current_state, paths["state"])
                return 2

            print(f"Task {task}, attempt {attempt}: Claude exited {claude_result.returncode} in {claude_result.duration_seconds:.1f}s")

            current_state.transition(state.QA, task=task)
            state.save(current_state, paths["state"])
        else:
            print(f"Task {task}: resuming mid-QA for attempt {attempt} -- Claude's work for this attempt already completed before the interruption, re-running QA without re-invoking Claude.")

        print(f"Task {task}, attempt {attempt}: running independent QA...")
        qa_result = qa.run_qa(task, repo_root, result_path=result_path, pre_task_commit=pre_task_commit, config=q_config)
        qa_log_path.parent.mkdir(parents=True, exist_ok=True)
        with qa_log_path.open("a") as f:
            f.write(f"=== attempt {attempt} ===\n{qa_result.render_report()}\n\n")
        current_state.record_qa(task, qa_result.to_dict())
        state.save(current_state, paths["state"])
        last_qa_dict = qa_result.to_dict()

        print(qa_result.render_report())

        if qa_result.passed:
            current_state.transition(state.PASSED, task=task)
            state.save(current_state, paths["state"])
            return _finish_passed_task(current_state, repo_root, paths, task, message_body=f"QA passed on attempt {attempt}/{max_attempts}.")

        # AUTOMATIC QA DECISION: a QA failure alone never means "ask the
        # user" -- classify whether the task itself reported a genuine
        # blocker (retrying cannot fix a missing credential or a real
        # scientific-protocol conflict) or an ordinary implementation
        # failure (the ordinary repair loop below is exactly for this).
        declared_result = _try_load_json(result_path)
        category = classify.classify_blocker(declared_result)
        if category in classify.STOP_REPAIR_CATEGORIES:
            reason = (declared_result or {}).get("blocking_issue", "(no blocking_issue field found in the result JSON)")
            print(f"Task {task}, attempt {attempt}: QA failed with a declared {category} blocker -- retrying will not help. Marking BLOCKED and stopping.")
            current_state.record_blocker(task, category, reason)
            current_state.transition(state.BLOCKED, task=task)
            state.save(current_state, paths["state"])
            return 1

        if attempt < max_attempts:
            print(f"Task {task}, attempt {attempt}: QA failed (ordinary implementation/methodological failure). Preparing a fix prompt for attempt {attempt + 1}.")
            current_state.transition(state.FIXING, task=task)
            state.save(current_state, paths["state"])
        else:
            print(f"Task {task}: QA failed after {max_attempts} attempts. Marking BLOCKED and stopping.")
            current_state.transition(state.BLOCKED, task=task)
            state.save(current_state, paths["state"])
            return 1

    print(f"Task {task}: no attempts remain (already used {attempts_already_made}/{max_attempts} before this run). Marking BLOCKED.")
    current_state.transition(state.BLOCKED, task=task)
    state.save(current_state, paths["state"])
    return 1


def _try_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _finish_passed_task(current_state: state.OrchestratorState, repo_root: Path, paths: dict, task: int, message_body: str = "resumed after interruption") -> int:
    """Create the checkpoint commit and advance state for a task whose
    status is already PASSED. Tolerates the working tree already being
    committed (crash happened between create_checkpoint() and
    advance_task() on a previous run) by falling back to the current
    HEAD commit instead of erroring on 'nothing to commit'."""
    try:
        commit_hash = git_ops.create_checkpoint(repo_root, task, message_body=message_body)
    except git_ops.GitError:
        # Nothing to commit almost certainly means a previous run already
        # committed the checkpoint before being interrupted -- use the
        # current HEAD rather than treating this as a new failure.
        try:
            commit_hash = git_ops.get_status(repo_root).commit
        except git_ops.GitError as exc:
            print(f"FATAL: task {task} is PASSED but no checkpoint commit could be created or found: {exc}", file=sys.stderr)
            current_state.transition(state.BLOCKED, task=task, reason=str(exc))
            state.save(current_state, paths["state"])
            return 2
    current_state.advance_task(commit_hash)
    state.save(current_state, paths["state"])
    git_ops.record_state_commit(repo_root, task, commit_hash)
    print(f"Task {task} PASSED. Checkpoint: {commit_hash[:12]}. Next task: {current_state.current_task}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
