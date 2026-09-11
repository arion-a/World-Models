"""Task 7B model hierarchy (contract Sec. 6): M0-M6, fixed order, every
candidate fit on train only, regularization/early-stopping selected on
validation only -- never on test. Fitting a model returns a FittedModel
whose .predict(Z) is a pure function of the frozen, already-fit
parameters; nothing in .predict ever looks at labels.

Parameter counts are computed from the ACTUAL fitted arrays' shapes, not
hard-coded, so a test can assert they match the contract's stated
formulas (tests/test_task7b_models.py) -- and so a discrepancy is
visible rather than silently assumed. See that test file's docstring for
one PRE-EXISTING inconsistency in the contract's own worked example for
M6 (its stated formula "2Dh+2D+h" does not itself arithmetically produce
the specific example counts it also lists) -- this module implements the
model EQUATION exactly as given (Zhat'=Z+B tanh(AZ+a)+b) and reports the
true parameter count for that equation (2Dh+D+h), rather than force-fitting
either of the contract's two mutually-inconsistent numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge

LOW_RANK_RANKS = (2, 4, 8, 16, 32)
MLP_HIDDEN_SIZES = (8, 16, 32)


@dataclass
class FittedModel:
    name: str
    n_params: int
    predict_fn: Callable[[np.ndarray], np.ndarray]
    hyperparams: dict = field(default_factory=dict)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.predict_fn(Z)


# --- M0: persistence ---------------------------------------------------------


def fit_m0_persistence(Z_train: np.ndarray, Zp_train: np.ndarray) -> FittedModel:
    return FittedModel("M0_persistence", 0, lambda Z: Z, {})


# --- M1: scalar ---------------------------------------------------------------


def fit_m1_scalar(Z_train: np.ndarray, Zp_train: np.ndarray) -> FittedModel:
    """a = argmin_a ||aZ - Z'||^2, closed form: a = <Z,Z'> / <Z,Z>."""
    num = float(np.sum(Z_train * Zp_train))
    den = float(np.sum(Z_train * Z_train))
    a = num / den if den > 1e-12 else 0.0
    return FittedModel("M1_scalar", 1, lambda Z, a=a: a * Z, {"a": a})


# --- M2: diagonal linear --------------------------------------------------


def fit_m2_diagonal(Z_train: np.ndarray, Zp_train: np.ndarray, alpha: float = 0.0) -> FittedModel:
    """Per-dimension ridge regression, closed form:
    w_d = sum_n Z[n,d] Z'[n,d] / (sum_n Z[n,d]^2 + alpha)."""
    num = np.sum(Z_train * Zp_train, axis=0)
    den = np.sum(Z_train**2, axis=0) + alpha
    w = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    D = Z_train.shape[1]
    return FittedModel("M2_diagonal", D, lambda Z, w=w: Z * w, {"alpha": alpha})


# --- M3: low-rank residual linear ------------------------------------------


def fit_m3_low_rank(Z_train: np.ndarray, Zp_train: np.ndarray, r: int, alpha: float = 1.0) -> FittedModel:
    """Zhat' = (I + UV^T) Z. Fit via reduced-rank regression: ridge-regress
    the residual (Z' - Z) on Z (full D x D operator L such that
    residual ~= Z @ L^T), then keep its best rank-r approximation (SVD
    truncation of L) -- the standard closed-form solution to a rank-
    constrained linear regression, not an ad hoc heuristic."""
    D = Z_train.shape[1]
    if r >= D:
        raise ValueError(f"low-rank r={r} must be < D={D}")
    residual_train = Zp_train - Z_train
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Z_train, residual_train)
    L_full = ridge.coef_  # (D, D): residual ~= Z @ L_full.T
    U_svd, S, Vt = np.linalg.svd(L_full, full_matrices=False)
    U = U_svd[:, :r] * S[:r]  # (D, r)
    V = Vt[:r, :]  # (r, D)
    L_r = U @ V  # (D, D), rank <= r

    def predict(Z, L_r=L_r):
        return Z + Z @ L_r.T

    return FittedModel(f"M3_low_rank_r{r}", 2 * D * r, predict, {"r": r, "alpha": alpha})


# --- M4: full linear (no bias) ---------------------------------------------


def fit_m4_full_linear(Z_train: np.ndarray, Zp_train: np.ndarray, alpha: float) -> FittedModel:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Z_train, Zp_train)
    W = ridge.coef_  # (D, D)
    D = Z_train.shape[1]
    return FittedModel("M4_full_linear", D * D, lambda Z, W=W: Z @ W.T, {"alpha": alpha})


# --- M5: affine ---------------------------------------------------------------


def fit_m5_affine(Z_train: np.ndarray, Zp_train: np.ndarray, alpha: float) -> FittedModel:
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(Z_train, Zp_train)
    W, b = ridge.coef_, ridge.intercept_
    D = Z_train.shape[1]
    return FittedModel("M5_affine", D * D + D, lambda Z, W=W, b=b: Z @ W.T + b, {"alpha": alpha})


# --- M6: small residual MLP, one hidden layer ------------------------------


def fit_m6_residual_mlp(
    Z_train: np.ndarray,
    Zp_train: np.ndarray,
    Z_val: np.ndarray,
    Zp_val: np.ndarray,
    h: int,
    seed: int = 0,
    max_epochs: int = 3000,
    patience: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> FittedModel:
    """Zhat' = Z + B tanh(A Z + a) + b -- exactly one hidden layer, no
    wider/deeper alternative. Early stopping on VALIDATION loss only
    (contract Sec. 6): training stops once `patience` epochs pass with no
    validation-loss improvement, and the returned model uses the
    best-validation-loss weights, never the last epoch's."""
    import torch

    torch.manual_seed(seed)
    D = Z_train.shape[1]

    Zt = torch.tensor(Z_train, dtype=torch.float32)
    Zpt = torch.tensor(Zp_train, dtype=torch.float32)
    Zv = torch.tensor(Z_val, dtype=torch.float32)
    Zpv = torch.tensor(Zp_val, dtype=torch.float32)

    A = torch.nn.Parameter(torch.randn(h, D) * (1.0 / np.sqrt(D)))
    a = torch.nn.Parameter(torch.zeros(h))
    B = torch.nn.Parameter(torch.randn(D, h) * (1.0 / np.sqrt(h)))
    b = torch.nn.Parameter(torch.zeros(D))

    def forward(Z, A=A, a=a, B=B, b=b):
        return Z + torch.tanh(Z @ A.T + a) @ B.T + b

    optimizer = torch.optim.Adam([A, a, B, b], lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0
    epochs_trained = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred = forward(Zt)
        loss = torch.mean((pred - Zpt) ** 2)
        loss.backward()
        optimizer.step()
        epochs_trained = epoch + 1

        with torch.no_grad():
            val_loss = torch.mean((forward(Zv) - Zpv) ** 2).item()
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = (A.detach().clone(), a.detach().clone(), B.detach().clone(), b.detach().clone())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break

    A_best, a_best, B_best, b_best = best_state
    A_np, a_np, B_np, b_np = A_best.numpy(), a_best.numpy(), B_best.numpy(), b_best.numpy()

    def predict(Z, A=A_np, a=a_np, B=B_np, b=b_np):
        return Z + np.tanh(Z @ A.T + a) @ B.T + b

    n_params = 2 * D * h + D + h  # A: h*D, a: h, B: D*h, b: D -- see module docstring
    return FittedModel(
        f"M6_residual_mlp_h{h}",
        n_params,
        predict,
        {"h": h, "seed": seed, "epochs_trained": epochs_trained, "best_val_loss": best_val_loss},
    )


# --- controls (contract Sec. 6's required baselines, fit like any model) ---


def fit_train_mean_control(Z_train: np.ndarray, Zp_train: np.ndarray) -> FittedModel:
    mean_vec = Zp_train.mean(axis=0)
    return FittedModel("control_train_mean", Zp_train.shape[1], lambda Z, m=mean_vec: np.broadcast_to(m, Z.shape).copy(), {})


def fit_shuffled_pair_control(Z_train: np.ndarray, Zp_train: np.ndarray, alpha: float, seed: int = 0) -> FittedModel:
    """Fits the SAME functional form (affine ridge) as the primary
    candidate would, but on deliberately shuffled (Z_i, Z'_perm(i)) pairs
    -- a negative control on the fitting procedure itself, not a strawman
    (uses the identical fitting path as M5)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Zp_train))
    if len(perm) > 1 and np.all(perm == np.arange(len(perm))):
        perm = np.roll(perm, 1)
    shuffled = Zp_train[perm]
    fitted = fit_m5_affine(Z_train, shuffled, alpha)
    return FittedModel("control_shuffled_pair", fitted.n_params, fitted.predict_fn, {"alpha": alpha, "seed": seed})
