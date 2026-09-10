"""Unit tests for orchestrator/claude_client.py: command construction
(argv arrays, never shell strings), dangerous-flag rejection, and
invoke_claude's handling of success/failure/timeout -- all via a mocked
subprocess.run, never a real `claude` CLI call.
"""

from __future__ import annotations

import subprocess

import pytest

from orchestrator import claude_client as cc


# --- build_command: pure function, no subprocess involved -------------------


def test_build_command_is_an_argv_list_not_a_shell_string():
    argv = cc.build_command("do the thing", cc.ClaudeConfig())
    assert isinstance(argv, list)
    assert all(isinstance(a, str) for a in argv)


def test_build_command_never_includes_dangerous_bypass_by_default():
    argv = cc.build_command("prompt", cc.ClaudeConfig())
    joined = " ".join(argv)
    assert "--allow-dangerously-skip-permissions" not in joined
    assert "bypassPermissions" not in joined


def test_build_command_uses_configured_permission_mode():
    argv = cc.build_command("prompt", cc.ClaudeConfig(permission_mode="acceptEdits"))
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"


def test_build_command_respects_claude_command_override():
    argv = cc.build_command("prompt", cc.ClaudeConfig(command="/custom/path/claude"))
    assert argv[0] == "/custom/path/claude"


def test_build_command_rejects_dangerous_extra_arg():
    cfg = cc.ClaudeConfig(extra_args=("--allow-dangerously-skip-permissions",))
    with pytest.raises(cc.ClaudeInvocationError):
        cc.build_command("prompt", cfg)


def test_build_command_prompt_is_a_single_argv_element_not_interpolated():
    """A prompt containing shell metacharacters must pass through as one
    literal argument, never get parsed/expanded -- this is what
    argument-array subprocess invocation buys over a shell string."""
    dangerous_prompt = "hello; rm -rf /; echo $(whoami) `id` && cat /etc/passwd"
    argv = cc.build_command(dangerous_prompt, cc.ClaudeConfig())
    assert dangerous_prompt in argv
    assert argv.count(dangerous_prompt) == 1


def test_config_from_env_reads_environment(monkeypatch):
    monkeypatch.setenv("CLAUDE_COMMAND", "my-claude")
    monkeypatch.setenv("CLAUDE_PERMISSION_MODE", "dontAsk")
    monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "42")
    cfg = cc.ClaudeConfig.from_env()
    assert cfg.command == "my-claude"
    assert cfg.permission_mode == "dontAsk"
    assert cfg.timeout_seconds == 42


def test_config_from_env_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("CLAUDE_COMMAND", raising=False)
    monkeypatch.delenv("CLAUDE_PERMISSION_MODE", raising=False)
    cfg = cc.ClaudeConfig.from_env()
    assert cfg.command == cc.DEFAULT_COMMAND
    assert cfg.permission_mode == cc.DEFAULT_PERMISSION_MODE


def test_redact_prompt_in_argv_hides_prompt_text():
    argv = ["claude", "-p", "a very long secret-ish prompt"]
    redacted = cc.redact_prompt_in_argv(argv, "a very long secret-ish prompt")
    assert "a very long secret-ish prompt" not in redacted
    assert any("chars" in a for a in redacted)


# --- invoke_claude: subprocess.run is mocked, never real ---------------------


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_invoke_claude_success(monkeypatch, tmp_path):
    def fake_run(argv, cwd, capture_output, text, timeout, check):
        assert isinstance(argv, list)
        return _FakeCompletedProcess(returncode=0, stdout='{"result": "ok"}', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = cc.invoke_claude("do it", tmp_path, config=cc.ClaudeConfig())
    assert result.ok
    assert result.returncode == 0
    assert result.parsed_json == {"result": "ok"}


def test_invoke_claude_nonzero_exit_is_reported_not_raised(monkeypatch, tmp_path):
    """A nonzero exit is data for QA to act on, not an exception -- the
    orchestrator must never crash just because Claude's own run failed."""

    def fake_run(argv, cwd, capture_output, text, timeout, check):
        return _FakeCompletedProcess(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = cc.invoke_claude("do it", tmp_path, config=cc.ClaudeConfig())
    assert not result.ok
    assert result.returncode == 1
    assert "boom" in result.stderr


def test_invoke_claude_unparseable_output_is_not_fatal(monkeypatch, tmp_path):
    def fake_run(argv, cwd, capture_output, text, timeout, check):
        return _FakeCompletedProcess(returncode=0, stdout="not json at all", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = cc.invoke_claude("do it", tmp_path, config=cc.ClaudeConfig())
    assert result.ok
    assert result.parsed_json is None


def test_invoke_claude_missing_executable_raises_clear_error(monkeypatch, tmp_path):
    def fake_run(argv, cwd, capture_output, text, timeout, check):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(cc.ClaudeInvocationError):
        cc.invoke_claude("do it", tmp_path, config=cc.ClaudeConfig(command="not-a-real-claude-binary"))


def test_invoke_claude_timeout_raises_clear_error(monkeypatch, tmp_path):
    def fake_run(argv, cwd, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(cc.ClaudeInvocationError):
        cc.invoke_claude("do it", tmp_path, config=cc.ClaudeConfig(timeout_seconds=1))


def test_invoke_claude_writes_a_log_with_prompt_redacted_from_argv_line(monkeypatch, tmp_path):
    def fake_run(argv, cwd, capture_output, text, timeout, check):
        return _FakeCompletedProcess(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    log_path = tmp_path / "log.txt"
    cc.invoke_claude("THE_SECRET_PROMPT_TEXT", tmp_path, config=cc.ClaudeConfig(), log_path=log_path)
    log_text = log_path.read_text()
    assert "THE_SECRET_PROMPT_TEXT" in log_text  # logged once, in the prompt section
    assert "command: " in log_text
    command_line = [l for l in log_text.splitlines() if l.startswith("command: ")][0]
    assert "THE_SECRET_PROMPT_TEXT" not in command_line  # but not duplicated into the argv line
