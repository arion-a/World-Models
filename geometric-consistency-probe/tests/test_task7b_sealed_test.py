"""QA gate (contract Sec. 12): 'Verify the test evaluation is sealed and
executed once; a rerun is allowed only for a documented software failure
before result inspection.' Also exercises the sealed test's own metric
correctness on a well-determined synthetic problem."""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.task7b_latent_transformation_discovery import sealed_test


def _synthetic_representations(n_scenes: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=D) * 0.5 + 1.0
    Z, Zp = {}, {}
    for i in range(n_scenes):
        sid = f"scene_{i:04d}"
        z = rng.normal(size=D)
        Z[sid] = z
        Zp[sid] = z * w_true
    return Z, Zp, [f"scene_{i:04d}" for i in range(n_scenes)]


def test_sealed_test_runs_once_and_reports_sensible_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed_test, "SEALED_TEST_PATH", tmp_path / "sealed_test_result.json")
    Z, Zp, all_ids = _synthetic_representations(70, D=8, seed=0)
    train_ids, val_ids, test_ids = all_ids[:40], all_ids[40:50], all_ids[50:]

    result = sealed_test.run_sealed_test("M2_diagonal", {"alpha": 0.01}, train_ids, val_ids, test_ids, Z, Zp, seed=0)
    assert result["selected_model_key"] == "M2_diagonal"
    assert result["selected_config_test_metrics"]["r2"] > 0.9
    assert result["n_test"] == len(test_ids)
    assert result["n_val"] == len(val_ids)
    assert (tmp_path / "sealed_test_result.json").exists()


def test_sealed_test_refuses_to_run_twice_without_force(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed_test, "SEALED_TEST_PATH", tmp_path / "sealed_test_result.json")
    Z, Zp, all_ids = _synthetic_representations(70, D=6, seed=1)
    train_ids, val_ids, test_ids = all_ids[:40], all_ids[40:50], all_ids[50:]

    sealed_test.run_sealed_test("M1_scalar", {}, train_ids, val_ids, test_ids, Z, Zp, seed=0)
    with pytest.raises(RuntimeError):
        sealed_test.run_sealed_test("M1_scalar", {}, train_ids, val_ids, test_ids, Z, Zp, seed=0)


def test_sealed_test_force_rerun_is_available_but_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed_test, "SEALED_TEST_PATH", tmp_path / "sealed_test_result.json")
    Z, Zp, all_ids = _synthetic_representations(70, D=6, seed=2)
    train_ids, val_ids, test_ids = all_ids[:40], all_ids[40:50], all_ids[50:]

    sealed_test.run_sealed_test("M1_scalar", {}, train_ids, val_ids, test_ids, Z, Zp, seed=0)
    # only succeeds because force=True is passed explicitly
    result = sealed_test.run_sealed_test("M1_scalar", {}, train_ids, val_ids, test_ids, Z, Zp, seed=0, force=True)
    assert result["sealed"] is True


def test_sealed_test_reports_per_scene_metrics_and_ci(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed_test, "SEALED_TEST_PATH", tmp_path / "sealed_test_result.json")
    Z, Zp, all_ids = _synthetic_representations(70, D=6, seed=3)
    train_ids, val_ids, test_ids = all_ids[:40], all_ids[40:50], all_ids[50:]
    result = sealed_test.run_sealed_test("M5_affine", {"alpha": 1.0}, train_ids, val_ids, test_ids, Z, Zp, seed=0)
    assert len(result["per_scene_r2"]) == len(test_ids)
    assert "mean" in result["bootstrap_ci_over_scenes"]


def test_sealed_test_result_file_is_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(sealed_test, "SEALED_TEST_PATH", tmp_path / "sealed_test_result.json")
    Z, Zp, all_ids = _synthetic_representations(70, D=6, seed=4)
    train_ids, val_ids, test_ids = all_ids[:40], all_ids[40:50], all_ids[50:]
    sealed_test.run_sealed_test("M0_persistence", {}, train_ids, val_ids, test_ids, Z, Zp, seed=0)
    reloaded = json.loads((tmp_path / "sealed_test_result.json").read_text())
    assert reloaded["contamination_status"].startswith("CLEAN")


def test_sealed_test_never_uses_test_data_for_m6_early_stopping(tmp_path, monkeypatch):
    """The critical leakage case: if the selected model is M6, its early
    stopping must be validated against val_ids, never test_ids. Spies on
    learning_curve._fit_candidate (the sole place early-stopping data is
    threaded through) and asserts the val arrays it receives match
    Z_val/Zp_val, not Z_test/Zp_test -- this is exactly the bug that was
    found and fixed here: the original implementation passed Z_test as
    the val argument, which every other model in this hierarchy ignores
    (so tests using M0/M1/M2/M5 above could not have caught it)."""
    monkeypatch.setattr(sealed_test, "SEALED_TEST_PATH", tmp_path / "sealed_test_result.json")
    Z, Zp, all_ids = _synthetic_representations(70, D=6, seed=5)
    train_ids, val_ids, test_ids = all_ids[:40], all_ids[40:50], all_ids[50:]

    seen_val_arrays = {}
    real_fit_candidate = sealed_test.learning_curve._fit_candidate

    def spy_fit_candidate(model_key, kwargs, Z_train, Zp_train, Z_val_arg, Zp_val_arg, seed):
        seen_val_arrays["Z_val"] = Z_val_arg
        seen_val_arrays["Zp_val"] = Zp_val_arg
        return real_fit_candidate(model_key, kwargs, Z_train, Zp_train, Z_val_arg, Zp_val_arg, seed)

    monkeypatch.setattr(sealed_test.learning_curve, "_fit_candidate", spy_fit_candidate)

    sealed_test.run_sealed_test("M6_residual_mlp", {"h": 4}, train_ids, val_ids, test_ids, Z, Zp, seed=0)

    expected_Z_val, expected_Zp_val = sealed_test.learning_curve._stack(val_ids, Z, Zp)
    expected_Z_test, expected_Zp_test = sealed_test.learning_curve._stack(test_ids, Z, Zp)

    assert np.array_equal(seen_val_arrays["Z_val"], expected_Z_val)
    assert np.array_equal(seen_val_arrays["Zp_val"], expected_Zp_val)
    assert not np.array_equal(seen_val_arrays["Z_val"], expected_Z_test)
    assert not np.array_equal(seen_val_arrays["Zp_val"], expected_Zp_test)
