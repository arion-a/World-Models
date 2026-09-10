"""Unit tests for orchestrator/qa.py's seven independent QA layers.

Each test builds a fake repo (tests/orchestrator/conftest.py) and checks
one specific way a task's result can be wrong -- missing file, NaN,
leaking split, stub markers, protected-file changes without
justification, etc. -- and confirms QA catches it. This is the test
suite for "the orchestrator must not mark a task PASS solely because
Claude reports success" (research/RESEARCH_INVARIANTS.md invariant 18).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from orchestrator import git_ops, qa
from tests.orchestrator.conftest import make_fake_repo, write_valid_result


def _commit_all(repo, message="wip"):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    return make_fake_repo(tmp_path / "repo", task=6)


def _base_commit(repo):
    return git_ops.get_status(repo).commit


# --- a fully valid result passes everything ---------------------------------


def test_valid_result_passes_all_layers(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    assert result.passed, result.render_report()
    assert all(layer.passed for layer in result.layers)


# --- Layer A: software correctness ------------------------------------------


def test_missing_result_file_fails_layer_a(repo):
    pre = _base_commit(repo)
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    assert not result.passed
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed


def test_invalid_json_fails_layer_a(repo):
    pre = _base_commit(repo)
    (repo / "state" / "task_06_result.json").write_text("{not valid json")
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed


def test_nan_in_result_fails_layer_a(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    data["metrics"]["learned_W_T"]["r2"] = float("nan")
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed
    assert any("NaN" in d or "nan" in d for d in layer_a.details)


def test_missing_declared_artifact_fails_layer_a(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6, artifact_name="report.md")
    path = repo / "state" / "task_06_result.json"
    (repo / "report.md").unlink()  # declared but now missing
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed


def test_stub_marker_fails_layer_a(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6, extra_fields={"scientific_result": "TODO: implement this properly later"})
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed


def test_all_zero_metrics_looks_like_a_stub(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    for block in data["metrics"].values():
        for k in block:
            block[k] = 0.0
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed


def test_failing_test_suite_fails_layer_a(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    (repo / "tests" / "test_dummy.py").write_text("def test_ok():\n    assert False\n")
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_a = next(l for l in result.layers if l.name.startswith("A."))
    assert not layer_a.passed


# --- Layer B: task acceptance ------------------------------------------------


def test_missing_required_field_fails_layer_b(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    del data["dataset"]["train_scene_ids"]
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_b = next(l for l in result.layers if l.name.startswith("B."))
    assert not layer_b.passed


def test_implementation_status_not_complete_fails_layer_b(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6, extra_fields={"implementation_status": "IN_PROGRESS"})
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_b = next(l for l in result.layers if l.name.startswith("B."))
    assert not layer_b.passed


def test_wrong_task_number_in_result_fails_layer_b(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6, extra_fields={"task": 7})
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_b = next(l for l in result.layers if l.name.startswith("B."))
    assert not layer_b.passed


# --- Layer C: scientific validity --------------------------------------------


def test_short_scientific_result_fails_layer_c(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6, extra_fields={"scientific_result": "ok"})
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_c = next(l for l in result.layers if l.name.startswith("C."))
    assert not layer_c.passed


def test_missing_baseline_comparison_fails_layer_c(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    data["metrics"] = {"only_learned_map": {"r2": 0.9}}  # no baseline/control anywhere
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_c = next(l for l in result.layers if l.name.startswith("C."))
    assert not layer_c.passed


def test_negative_scientific_result_still_passes_layer_c():
    """SCIENTIFIC NEGATIVE RESULT must not be penalized -- only the
    presence/structure of the measurement is checked, never its sign."""
    result_json = {
        "task": 6,
        "scientific_result": "The learned map performed no better than the persistence baseline; no equivariance detected.",
        "metrics": {"learned_W_T": {"r2": -0.3}, "persistence_baseline": {"r2": 0.1}},
    }
    layer = qa._layer_c_scientific_validity(6, result_json)
    assert layer.passed, layer.details


# --- Layer D: research alignment ---------------------------------------------


def test_missing_invariants_file_fails_layer_d(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    (repo / "research" / "RESEARCH_INVARIANTS.md").unlink()
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_d = next(l for l in result.layers if l.name.startswith("D."))
    assert not layer_d.passed


def test_watered_down_invariants_fails_layer_d(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    text = (repo / "research" / "RESEARCH_INVARIANTS.md").read_text()
    # remove invariant 18's line entirely
    lines = [l for l in text.splitlines() if not l.startswith("18. **")]
    (repo / "research" / "RESEARCH_INVARIANTS.md").write_text("\n".join(lines))
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_d = next(l for l in result.layers if l.name.startswith("D."))
    assert not layer_d.passed


def test_failing_research_alignment_tests_fails_layer_d(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    (repo / "tests" / "research" / "test_fake_alignment.py").write_text("def test_ok():\n    assert False\n")
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_d = next(l for l in result.layers if l.name.startswith("D."))
    assert not layer_d.passed


# --- Layer E: data leakage ----------------------------------------------------


def test_overlapping_train_test_scenes_fails_layer_e(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    data["dataset"]["test_scene_ids"].append(data["dataset"]["train_scene_ids"][0])  # leak!
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_e = next(l for l in result.layers if l.name.startswith("E."))
    assert not layer_e.passed
    assert any("leakage" in d for d in layer_e.details)


def test_missing_split_fields_fails_layer_e_for_split_bearing_task(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    del data["dataset"]  # no train/test ids anywhere now
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_e = next(l for l in result.layers if l.name.startswith("E."))
    assert not layer_e.passed


def test_infra_task_is_exempt_from_leakage_layer():
    """Task 10 (baseline framework) is infrastructure-only -- it declares
    no train/test split, and that must not be treated as a failure."""
    layer = qa._layer_e_data_leakage(10, {"task": 10, "implementation_status": "COMPLETE"})
    assert layer.passed


# --- Layer F: reproducibility --------------------------------------------------


def test_missing_seed_fails_layer_f(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    del data["seed"]
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_f = next(l for l in result.layers if l.name.startswith("F."))
    assert not layer_f.passed


def test_missing_config_fails_layer_f(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    path = repo / "state" / "task_06_result.json"
    data = json.loads(path.read_text())
    data["config"] = None
    path.write_text(json.dumps(data))
    result = qa.run_qa(6, repo, path, pre_task_commit=pre)
    layer_f = next(l for l in result.layers if l.name.startswith("F."))
    assert not layer_f.passed


# --- Layer G: regression / protected files -----------------------------------


def test_unjustified_protected_file_change_fails_layer_g(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    (repo / "generation" / "core.py").write_text("# protected placeholder module\nVALUE = 2  # changed!\n")
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_g = next(l for l in result.layers if l.name.startswith("G."))
    assert not layer_g.passed
    assert any("generation/core.py" in d for d in layer_g.details)


def test_justified_protected_file_change_passes_layer_g(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6, extra_fields={"protected_files_justification": "fixed a real bug in generation/core.py needed for this task"})
    (repo / "generation" / "core.py").write_text("# protected placeholder module\nVALUE = 2  # fixed bug\n")
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_g = next(l for l in result.layers if l.name.startswith("G."))
    assert layer_g.passed


def test_new_untracked_protected_file_is_also_caught(repo):
    """A brand new (untracked) file under a protected path must be
    caught too, not just modifications to existing tracked files."""
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    (repo / "generation" / "new_module.py").write_text("# sneaky new file\n")
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_g = next(l for l in result.layers if l.name.startswith("G."))
    assert not layer_g.passed


def test_unrelated_file_change_does_not_trigger_protected_files_check(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)  # writes report.md and state/task_06_result.json, both unprotected
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    layer_g = next(l for l in result.layers if l.name.startswith("G."))
    assert layer_g.passed


# --- QAResult reporting -------------------------------------------------------


def test_render_report_from_persisted_dict_matches_live_object(repo):
    pre = _base_commit(repo)
    write_valid_result(repo, task=6)
    result = qa.run_qa(6, repo, repo / "state" / "task_06_result.json", pre_task_commit=pre)
    reconstructed = qa.render_report(result.to_dict())
    assert reconstructed == result.render_report()
