"""Tests for evaluation.run_task -- Task 15's unified CLI dispatcher over
Tasks 6-14's existing experiment code.

These never actually run a task's real experiment (rendering/encoding):
`resolve_task_module`/`run_task` are exercised against monkeypatched
`main()` functions so the test suite stays in the fast ("not slow")
tier while still proving the dispatcher correctly imports and calls
each task's own entry point rather than reimplementing it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from evaluation.run_task import (
    TASK_MODULES,
    main,
    parse_task_selector,
    resolve_task_module,
    run_task,
)


def test_task_modules_covers_tasks_6_through_14():
    assert set(TASK_MODULES.keys()) == set(range(6, 15))


@pytest.mark.parametrize("task_id", sorted(TASK_MODULES))
def test_every_task_module_is_importable_and_has_main(task_id):
    module = importlib.import_module(TASK_MODULES[task_id])
    assert hasattr(module, "main"), f"{TASK_MODULES[task_id]} has no main()"


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("6", [6]),
        ("6,7,8", [6, 7, 8]),
        ("6-9", [6, 7, 8, 9]),
        ("6..9", [6, 7, 8, 9]),
        ("6..14", list(range(6, 15))),
        ("6,8-10", [6, 8, 9, 10]),
        (" 6 , 7 ", [6, 7]),
        ("6,6,7", [6, 7]),  # de-duplicated, order preserved
    ],
)
def test_parse_task_selector_accepts_valid_forms(selector, expected):
    assert parse_task_selector(selector) == expected


@pytest.mark.parametrize(
    "selector",
    [
        "",
        "  ",
        "5",  # not one of Tasks 6-14
        "6,5",
        "15",  # Task 15 itself is not in TASK_MODULES (dispatches OTHER tasks)
        "6-",
        "-9",
        "abc",
        "6..abc",
        "9-6",  # start > end
        "6,,7",
    ],
)
def test_parse_task_selector_rejects_invalid_forms_loudly(selector):
    with pytest.raises(ValueError):
        parse_task_selector(selector)


def test_resolve_task_module_rejects_unknown_task_id():
    with pytest.raises(ValueError):
        resolve_task_module(999)


def test_resolve_task_module_returns_real_module_for_known_task():
    module = resolve_task_module(6)
    assert module.__name__ == "experiments.task6_camera_rotation"


def test_run_task_dispatches_to_the_right_module_without_running_the_real_experiment(monkeypatch):
    module = importlib.import_module(TASK_MODULES[6])
    calls = []

    def fake_main():
        calls.append(list(sys.argv))

    monkeypatch.setattr(module, "main", fake_main)
    run_task(6, ["--config", "configs/experiments/task6_camera_rotation.yaml", "--skip_tests"])

    assert len(calls) == 1
    assert calls[0][0] == "python -m experiments.task6_camera_rotation"
    assert "--config" in calls[0]
    assert "configs/experiments/task6_camera_rotation.yaml" in calls[0]
    assert "--skip_tests" in calls[0]


def test_run_task_restores_sys_argv_even_if_the_task_raises(monkeypatch):
    module = importlib.import_module(TASK_MODULES[7])

    def failing_main():
        raise RuntimeError("simulated task failure")

    monkeypatch.setattr(module, "main", failing_main)
    original_argv = list(sys.argv)
    with pytest.raises(RuntimeError):
        run_task(7, [])
    assert sys.argv == original_argv


def test_main_requires_task_argument():
    with pytest.raises(SystemExit):
        main([])


def test_main_dispatches_multiple_tasks_in_order(monkeypatch):
    order = []
    for task_id in (6, 7):
        module = importlib.import_module(TASK_MODULES[task_id])
        monkeypatch.setattr(module, "main", lambda tid=task_id: order.append(tid))

    main(["--task", "6,7"])
    assert order == [6, 7]
