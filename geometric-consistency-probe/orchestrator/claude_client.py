"""Non-interactive Claude Code invocation.

Builds and runs a command conceptually equivalent to:

    claude -p "<task prompt>" --output-format json --permission-mode acceptEdits --permission-prompts none

using `subprocess.run` with an argument list (never `shell=True`, never
a concatenated shell string -- avoids shell-injection entirely
regardless of what a task prompt contains).

Configuration is via environment variables, all optional:

  CLAUDE_COMMAND          executable name/path (default: "claude")
  CLAUDE_PERMISSION_MODE  (default: "acceptEdits" -- NOT a permission-
                           bypass mode; auto-accepts edit-type prompts
                           only, everything else still goes through
                           normal tool-permission rules)
  CLAUDE_TIMEOUT_SECONDS  per-invocation timeout (default: 3600)
  CLAUDE_EXTRA_ARGS       extra args, shlex-split then passed as
                           separate argv elements (still never through
                           a shell)

Deliberately never adds `--allow-dangerously-skip-permissions` or
`--permission-mode bypassPermissions` -- the orchestrator's brief is
explicit that dangerous permission-bypass flags must not be the
default. An operator who wants that can set CLAUDE_EXTRA_ARGS
themselves, which is their own explicit, visible choice, logged like
everything else here.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import uuid
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COMMAND = "claude"
DEFAULT_PERMISSION_MODE = "acceptEdits"
DEFAULT_TIMEOUT_SECONDS = 3600

# Flags this module will never add on its own, no matter what
# CLAUDE_PERMISSION_MODE is set to -- an operator has to add these via
# CLAUDE_EXTRA_ARGS explicitly if they really want them.
_DANGEROUS_FLAGS = ("--allow-dangerously-skip-permissions",)


class ClaudeInvocationError(Exception):
    """Could not even start the Claude CLI (not found, bad config)."""


@dataclass
class ClaudeConfig:
    command: str = DEFAULT_COMMAND
    permission_mode: str = DEFAULT_PERMISSION_MODE
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        extra = os.environ.get("CLAUDE_EXTRA_ARGS", "")
        return cls(
            command=os.environ.get("CLAUDE_COMMAND", DEFAULT_COMMAND),
            permission_mode=os.environ.get("CLAUDE_PERMISSION_MODE", DEFAULT_PERMISSION_MODE),
            timeout_seconds=int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
            extra_args=tuple(shlex.split(extra)) if extra else (),
        )


@dataclass
class ClaudeResult:
    returncode: int
    stdout: str
    stderr: str
    parsed_json: dict | None
    duration_seconds: float
    command: list[str]
    session_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def detect_platform() -> str:
    """Informational only (logged, never used to change behavior beyond
    the Windows .cmd fallback in build_command): 'linux', 'wsl',
    'darwin', or 'windows'."""
    system = platform.system().lower()
    if system == "linux":
        try:
            if "microsoft" in Path("/proc/version").read_text().lower():
                return "wsl"
        except OSError:
            pass
        return "linux"
    if system == "darwin":
        return "darwin"
    if system == "windows":
        return "windows"
    return system


def build_command(prompt: str, config: ClaudeConfig, session_id: str | None = None) -> list[str]:
    """Pure function: prompt + config (+ an optional pre-generated
    session_id) -> argv list. No subprocess call, no I/O -- kept separate
    from invoke_claude() so command construction can be unit-tested
    without ever touching the real CLI.

    `session_id`, when given, is passed as `--session-id`. This matters:
    plain `claude -p "..."` with no session flag defaults to *continuing
    the most recent session tied to the working directory* rather than
    starting a fresh one -- confirmed empirically, not assumed, by
    running it from this very directory and observing it reuse this
    orchestrating session's own session_id. Task implementations must
    run in a genuinely fresh session, isolated from whatever session
    (this one included) happens to have last touched the repo, so
    invoke_claude() always generates and passes a fresh UUID.
    """
    command = config.command
    if platform.system().lower() == "windows" and detect_platform() == "windows":
        # Native Windows (not WSL): a plain "claude" on PATH is often the
        # npm-installed .cmd shim; subprocess without shell=True needs the
        # exact executable, so try the .cmd form only if the plain name
        # wasn't explicitly overridden to something else via CLAUDE_COMMAND.
        if command == DEFAULT_COMMAND:
            import shutil

            resolved = shutil.which("claude.cmd") or shutil.which("claude")
            if resolved:
                command = resolved

    argv = [
        command,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--permission-mode",
        config.permission_mode,
        "--permission-prompts",
        "none",
    ]
    if session_id:
        argv.extend(["--session-id", session_id])
    for flag in config.extra_args:
        if flag in _DANGEROUS_FLAGS:
            raise ClaudeInvocationError(
                f"refusing to pass {flag!r} via CLAUDE_EXTRA_ARGS automatically -- "
                "dangerous permission-bypass flags are not enabled by this orchestrator"
            )
    argv.extend(config.extra_args)
    return argv


def invoke_claude(
    prompt: str,
    repo_root: str | Path,
    config: ClaudeConfig | None = None,
    log_path: str | Path | None = None,
) -> ClaudeResult:
    """Run Claude Code non-interactively on `prompt`, cwd=repo_root.

    Never raises on a nonzero exit or a Claude-side failure -- that is
    reported via ClaudeResult.ok / returncode so the orchestrator's QA
    layer (never Claude's own claim) decides what happened. Only raises
    ClaudeInvocationError if the process could not even be started
    (executable not found) or timed out.
    """
    config = config or ClaudeConfig.from_env()
    session_id = str(uuid.uuid4())
    argv = build_command(prompt, config, session_id=session_id)

    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ClaudeInvocationError(
            f"Claude executable {config.command!r} not found on PATH. "
            f"Set CLAUDE_COMMAND to the correct executable for this platform ({detect_platform()})."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        _write_log(log_path, argv, prompt, stdout=exc.stdout or "", stderr=exc.stderr or "", returncode=None, duration=duration)
        raise ClaudeInvocationError(f"Claude invocation timed out after {config.timeout_seconds}s") from exc

    duration = time.monotonic() - started
    parsed = _try_parse_json(proc.stdout)
    _write_log(log_path, argv, prompt, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode, duration=duration)

    return ClaudeResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        parsed_json=parsed,
        duration_seconds=duration,
        command=argv,
        session_id=session_id,
    )


def redact_prompt_in_argv(argv: list[str], prompt: str) -> list[str]:
    """For display/logging only: replace the (potentially huge) prompt
    argument with a short placeholder so terminal output and dry-run
    summaries stay readable."""
    return [a if a != prompt else f"<prompt, {len(prompt)} chars>" for a in argv]


def _try_parse_json(stdout: str) -> dict | None:
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def _write_log(log_path, argv, prompt, *, stdout, stderr, returncode, duration) -> None:
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Log the argv with the prompt redacted to a length note, not the raw
    # text twice over -- the full prompt is logged once, clearly labeled,
    # never duplicated into a shell-reconstructable command line.
    safe_argv = redact_prompt_in_argv(argv, prompt)
    lines = [
        f"=== Claude invocation ===",
        f"command: {safe_argv}",
        f"returncode: {returncode}",
        f"duration_seconds: {duration:.2f}",
        "--- prompt ---",
        prompt,
        "--- stdout ---",
        stdout,
        "--- stderr ---",
        stderr,
        "",
    ]
    with log_path.open("a") as f:
        f.write("\n".join(lines))
