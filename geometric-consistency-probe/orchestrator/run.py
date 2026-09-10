"""Orchestrator CLI entry point.

    python -m orchestrator.run --dry-run     # show state/plan, change nothing
    python -m orchestrator.run               # actually run the next task
    python -m orchestrator.run --loop        # run tasks back-to-back while
                                              # their spec files exist
    python -m orchestrator.run --reset-task 6  # explicit operator override
                                              # to retry a BLOCKED/FAILED task

Flow per task (see the module docstrings of state.py / qa.py / prompts.py
/ claude_client.py / git_ops.py for the pieces this wires together):

    load state -> determine next task -> require a clean git tree
    -> RUNNING: invoke Claude with the task prompt
    -> QA: independently verify (never trust Claude's own report)
    -> PASSED -> git checkpoint commit -> advance_task() -> done
       FAILED (QA) and attempts remain -> FIXING: invoke Claude with a
           fix prompt, back to RUNNING
       FAILED (QA) and attempts exhausted -> BLOCKED, stop

If tasks/{NN}_*.md does not exist yet for the next task, the orchestrator
stops cleanly without changing anything -- this is the intended way the
user's own workflow ("insert task N+1's prompt only after N is accepted")
gates the pipeline, not an error condition.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from orchestrator import claude_client, git_ops, qa, state
from orchestrator.prompts import TaskSpecNotFound, render_fix_prompt, render_task_prompt

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
    parser.add_argument("--loop", action="store_true", help="After a task passes, continue to the next one if its spec file already exists.")
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


def _dry_run(current_state: state.OrchestratorState, repo_root: Path, paths: dict, args) -> int:
    print("=== DRY RUN (no changes will be made) ===")
    print(f"repo_root: {repo_root}")
    print(f"state file: {paths['state']} ({'exists' if paths['state'].exists() else 'will be bootstrapped'})")
    print(f"current status: {current_state.status}")
    print(f"completed_tasks: {current_state.completed_tasks}")
    print(f"current_task: {current_state.current_task}")

    if current_state.status in (state.BLOCKED, state.FAILED):
        print(f"Task {current_state.current_task} is {current_state.status} -- orchestrator would refuse to proceed without --reset-task.")
        return 0

    next_task = current_state.next_task()
    if next_task is None:
        print("All tasks (1-18) complete. Nothing to do.")
        return 0

    print(f"next task: {next_task}")
    try:
        prompt = render_task_prompt(next_task, paths["tasks"])
    except TaskSpecNotFound as exc:
        print(f"Task spec not found: {exc}")
        print("Orchestrator would stop here and wait for the task file to be added.")
        return 0

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
        # Only continue the loop if the next task's spec file already
        # exists -- otherwise stop cleanly and wait, per the user's own
        # "insert the next task only after this one is accepted" workflow.
        try:
            render_task_prompt(current_state.next_task(), paths["tasks"])
        except TaskSpecNotFound:
            print(f"--loop: task {current_state.next_task()}'s spec file does not exist yet. Stopping and waiting.")
            return 0


def _run_one_task(current_state: state.OrchestratorState, repo_root: Path, paths: dict, args) -> int:
    if current_state.status in (state.BLOCKED, state.FAILED):
        print(f"Task {current_state.current_task} is {current_state.status}. Refusing to proceed automatically.")
        print(f"Use `python -m orchestrator.run --reset-task {current_state.current_task}` after investigating, to retry.")
        return 1

    task = current_state.next_task()
    if task is None:
        print("All tasks (1-18) complete. Nothing to do.")
        return 0

    try:
        render_task_prompt(task, paths["tasks"])  # just to check existence up front
    except TaskSpecNotFound as exc:
        print(f"Task {task}: {exc}")
        print("Stopping and waiting for the task spec file to be added -- state is unchanged.")
        return 0

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
        try:
            if args.allow_dirty:
                git_ops.get_status(repo_root)
            else:
                git_ops.require_clean_tree(repo_root)
        except git_ops.GitError as exc:
            print(f"FATAL: {exc}", file=sys.stderr)
            return 2
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

    attempts_already_made = current_state.attempts.get(str(task), 0)
    last_qa_dict = current_state.last_qa.get(str(task))

    for attempt in range(attempts_already_made + 1, max_attempts + 1):
        if current_state.status == state.FIXING:
            current_state.transition(state.RUNNING, task=task)
            state.save(current_state, paths["state"])

        current_state.record_attempt(task)
        state.save(current_state, paths["state"])

        if attempt == 1:
            prompt = render_task_prompt(task, paths["tasks"])
        else:
            prior_report = qa.render_report(last_qa_dict) if last_qa_dict else "(no prior QA report available -- resumed after interruption)"
            prompt = render_fix_prompt(task, paths["tasks"], prior_report, attempt, max_attempts)

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

        if attempt < max_attempts:
            print(f"Task {task}, attempt {attempt}: QA failed. Preparing a fix prompt for attempt {attempt + 1}.")
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
