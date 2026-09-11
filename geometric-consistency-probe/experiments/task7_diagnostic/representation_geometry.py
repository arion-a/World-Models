"""Diagnostic E: representation geometry -- extends the forensic audit's
descriptive statistics with a TRAIN-FITTED PCA/whitening comparison.

Critical difference from a naive whitening: the PCA basis and the
whitening scale are fit on Z_train ONLY, then applied (transform, not
re-fit) to Z_test -- exactly analogous to how Ridge's own internal
centering is train-only (per the forensic audit's normalization
finding). This is a diagnostic of whether representation anisotropy
itself (not sample size) is bottlenecking the regression: whitening
equalizes the variance across the ~18 effective dimensions before
fitting, which is a legitimate, well-justified transformation to test
(unlike, say, selecting a target-informed subspace, which would leak).

We do NOT select whitening/PCA based on TEST performance anywhere in
this script -- the only test-set contact is the final read-only R^2
evaluation, once, per transform, per method.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
AUDIT_DIR = REPO_ROOT / "experiments" / "task7_forensic_audit"


def r_squared(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def cosine_sim(pred, target):
    num = np.sum(pred * target, axis=-1)
    denom = np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1)
    denom = np.where(denom == 0, 1e-12, denom)
    return float(np.mean(num / denom))


def effective_rank(eigvals):
    ev = eigvals[eigvals > 1e-12]
    p = ev / ev.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


class TrainFittedWhitener:
    """PCA + optional whitening, fit on TRAIN ONLY. transform() applies
    the stored (train-derived) mean/basis/scale to any array, train or
    test -- never re-fit on what it is given."""

    def __init__(self, n_components: int, whiten: bool):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X_train: np.ndarray) -> "TrainFittedWhitener":
        self.mean_ = X_train.mean(axis=0)
        centered = X_train - self.mean_
        # SVD-based PCA (more stable than eig(cov) for N < D)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.components_ = Vt[: self.n_components]  # (k, D)
        n = X_train.shape[0]
        self.explained_variance_ = (S[: self.n_components] ** 2) / (n - 1)
        self.full_eigvals_ = (S**2) / (n - 1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        projected = (X - self.mean_) @ self.components_.T  # (N, k)
        if self.whiten:
            scale = np.sqrt(np.clip(self.explained_variance_, 1e-12, None))
            projected = projected / scale
        return projected

    def inverse_transform(self, Xp: np.ndarray) -> np.ndarray:
        if self.whiten:
            scale = np.sqrt(np.clip(self.explained_variance_, 1e-12, None))
            Xp = Xp * scale
        return Xp @ self.components_ + self.mean_


def main():
    recorded = json.loads((REPO_ROOT / "state" / "task_07_result.json").read_text())
    transforms = list(recorded["transforms_evaluated"])
    train_ids = recorded["dataset"]["train_scene_ids"]
    test_ids = recorded["dataset"]["test_scene_ids"]

    npz = np.load(AUDIT_DIR / "recomputed_representations.npz")
    Z_train = np.stack([npz[f"Z__{sid}"] for sid in train_ids])
    Z_test = np.stack([npz[f"Z__{sid}"] for sid in test_ids])
    n_train = Z_train.shape[0]

    # geometry, fit on TRAIN only (this is the number that matters for
    # deciding what "effective rank" the regression actually has to work
    # with, since the regression itself only ever sees Z_train's span).
    train_pca = TrainFittedWhitener(n_components=min(n_train - 1, Z_train.shape[1]), whiten=False).fit(Z_train)
    train_only_geometry = {
        "n_train": n_train,
        "effective_rank_train_only": effective_rank(train_pca.full_eigvals_),
        "explained_variance_by_top_k": {
            k: float(train_pca.full_eigvals_[:k].sum() / train_pca.full_eigvals_.sum())
            for k in (1, 5, 10, 20, 31) if k <= len(train_pca.full_eigvals_)
        },
        "note": "PCA fit on the 32 TRAIN scenes only (N_train-1=31 components max) -- distinct from the forensic audit's N=40 (train+test pooled) effective-rank figure; this is what the regression itself actually has to work with.",
    }
    print(json.dumps(train_only_geometry, indent=2))

    results = {"train_only_geometry": train_only_geometry, "per_transform": {}}

    # Try whitening in a modest number of retained components (k << 32,
    # since with only 32 train samples, using all 31 non-trivial
    # components and then whitening to unit variance is itself prone to
    # amplifying noise in low-variance directions -- k is a design
    # choice made WITHOUT looking at test performance, using only the
    # explained-variance-by-top-k figure above as the criterion (a
    # standard, test-blind rule: keep enough components for ~90% of
    # train variance).
    k_90pct = next(
        k for k in range(1, len(train_pca.full_eigvals_) + 1)
        if train_pca.full_eigvals_[:k].sum() / train_pca.full_eigvals_.sum() >= 0.90
    )
    print(f"\nk retained for >=90% train-explained variance: {k_90pct}")
    results["k_selected_for_90pct_train_variance"] = k_90pct

    ridge_alpha = recorded["config"]["ridge_alpha"]

    for t in transforms:
        Zp_train = np.stack([npz[f"Zp__{t}__{sid}"] for sid in train_ids])
        Zp_test = np.stack([npz[f"Zp__{t}__{sid}"] for sid in test_ids])

        entry = {}

        # baseline: original space, same fixed alpha (sanity re-check)
        ridge_raw = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(Z_train, Zp_train)
        pred_raw = ridge_raw.predict(Z_test)
        entry["raw_space_ridge"] = {"r2": r_squared(pred_raw, Zp_test), "cosine": cosine_sim(pred_raw, Zp_test)}

        # PCA-reduced (no whitening), k=k_90pct, fit on TRAIN only, ridge in reduced space, map back
        pca = TrainFittedWhitener(n_components=k_90pct, whiten=False).fit(Z_train)
        Zt_train_reduced = pca.transform(Z_train)
        Zt_test_reduced = pca.transform(Z_test)
        Zpt_train_reduced = pca.transform(Zp_train)  # project TARGET using the SAME Z-fitted basis
        ridge_pca = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(Zt_train_reduced, Zpt_train_reduced)
        pred_pca_reduced = ridge_pca.predict(Zt_test_reduced)
        pred_pca = pca.inverse_transform(pred_pca_reduced)
        entry["pca_reduced_ridge"] = {
            "k_components": k_90pct,
            "r2": r_squared(pred_pca, Zp_test),
            "cosine": cosine_sim(pred_pca, Zp_test),
        }

        # Whitened (unit variance per retained component), same k, fit on TRAIN only
        whitener = TrainFittedWhitener(n_components=k_90pct, whiten=True).fit(Z_train)
        Zw_train = whitener.transform(Z_train)
        Zw_test = whitener.transform(Z_test)
        Zpw_train = whitener.transform(Zp_train)
        ridge_whitened = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(Zw_train, Zpw_train)
        pred_whitened_reduced = ridge_whitened.predict(Zw_test)
        pred_whitened = whitener.inverse_transform(pred_whitened_reduced)
        entry["whitened_ridge"] = {
            "k_components": k_90pct,
            "r2": r_squared(pred_whitened, Zp_test),
            "cosine": cosine_sim(pred_whitened, Zp_test),
        }

        results["per_transform"][t] = entry
        print(f"{t}: raw r2={entry['raw_space_ridge']['r2']:.4f}  "
              f"pca(k={k_90pct}) r2={entry['pca_reduced_ridge']['r2']:.4f}  "
              f"whitened(k={k_90pct}) r2={entry['whitened_ridge']['r2']:.4f}")

    (OUT_DIR / "representation_geometry.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
