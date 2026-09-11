"""Section 7 (representation statistics) + Section 14 (pooled vs dense)
analysis, run against the REAL Z/Z' arrays independently re-derived by
full_reproduction.py (never the ones Task 7 cached internally -- those
were never persisted to disk in the first place, which is itself an
audit-relevant fact: see audit_report.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent


def effective_rank(cov_eigvals: np.ndarray) -> float:
    """exp(entropy of normalized eigenvalue spectrum) -- a continuous
    'how many dimensions are actually doing work' measure (Roy & Vetterli
    2007), 1 = totally collapsed to one direction, D = perfectly isotropic."""
    ev = cov_eigvals[cov_eigvals > 1e-12]
    p = ev / ev.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def analyze(Z: np.ndarray, label: str) -> dict:
    norms = np.linalg.norm(Z, axis=1)
    mean_vec = Z.mean(axis=0)
    centered = Z - mean_vec
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)[::-1]
    total_var = eigvals.sum()
    explained_by_top = {
        k: float(eigvals[:k].sum() / total_var) for k in (1, 5, 10, 50, 100) if k <= len(eigvals)
    }
    # pairwise cosine similarity (mean off-diagonal)
    normed = Z / np.clip(norms, 1e-12, None)[:, None]
    cos_matrix = normed @ normed.T
    n = Z.shape[0]
    off_diag_mean = float((cos_matrix.sum() - n) / (n * (n - 1)))

    return {
        "n_samples": n,
        "dim": Z.shape[1],
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "total_variance": float(total_var),
        "per_dimension_variance_mean": float(np.diag(cov).mean()),
        "per_dimension_variance_std": float(np.diag(cov).std()),
        "pca_explained_variance_by_top_k_components": explained_by_top,
        "effective_rank": effective_rank(eigvals),
        "ambient_dim": Z.shape[1],
        "mean_pairwise_cosine_similarity_off_diagonal": off_diag_mean,
        "note_on_anisotropy": (
            "A high mean pairwise cosine similarity combined with a small effective_rank relative "
            "to ambient_dim indicates the representation cloud occupies a narrow cone / low-dimensional "
            "subspace rather than being spread isotropically -- common in pretrained transformer "
            "embeddings ('representation anisotropy'), and relevant here because it can make a linear "
            "regression's job either easier (if the transform's effect lies along the dominant "
            "directions) or effectively impossible (if the transform's effect is a small perturbation "
            "orthogonal to, or smaller than, the dominant directions' scale)."
        ),
    }


def main():
    npz = np.load(OUT_DIR / "recomputed_representations.npz")
    recorded = json.loads((OUT_DIR.parent.parent / "state" / "task_07_result.json").read_text())
    transforms = list(recorded["transforms_evaluated"])
    all_scene_ids = sorted(set(recorded["dataset"]["train_scene_ids"]) | set(recorded["dataset"]["test_scene_ids"]))

    Z = np.stack([npz[f"Z__{sid}"] for sid in all_scene_ids])
    stats = {"Z_original": analyze(Z, "Z_original")}
    for t in transforms:
        Zp = np.stack([npz[f"Zp__{t}__{sid}"] for sid in all_scene_ids])
        stats[f"Zp_{t}"] = analyze(Zp, f"Zp_{t}")
        # also: per-scene delta = Zp - Z, does the TRANSFORM's effect have
        # a much smaller norm than Z itself? (small effect could be
        # swamped by mean-pooling / by the dominant representation
        # directions unrelated to the transform)
        delta = Zp - Z
        delta_norms = np.linalg.norm(delta, axis=1)
        z_norms = np.linalg.norm(Z, axis=1)
        stats[f"Zp_{t}"]["delta_vs_Z_norm_ratio_mean"] = float((delta_norms / z_norms).mean())
        stats[f"Zp_{t}"]["delta_norm_mean"] = float(delta_norms.mean())

    # pooled vs dense: token-level stats for one real encode, straight
    # from the real cached token sequence shape already established.
    stats["pooling_summary"] = {
        "native_token_sequence_shape_per_clip": [512, 1024],
        "pooling_operation": "unweighted mean over all 512 tokens (both the 2 temporal tubelets AND all 256 spatial patches, collapsed together) -> a single (1024,) vector",
        "tokens_discarded_by_pooling": "512 - 1 = 511 of 512 token positions' individual identity/location (spatial row/col, temporal tubelet index) is discarded; only their mean survives",
        "consequence": (
            "A camera rotation, for instance, primarily changes WHICH spatial patches see WHICH content "
            "(content shifts across the 16x16 spatial grid) rather than changing the marginal distribution "
            "of patch content much -- mean-pooling is specifically insensitive to this kind of purely "
            "positional/geometric rearrangement, since summing over all patch positions destroys "
            "information about which position held which content. This is a plausible, methodologically "
            "significant contributor to a negative geometric-equivariance result, independent of whether "
            "the token-level (unpooled) representations do encode geometric structure."
        ),
    }

    (OUT_DIR / "representation_statistics.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "pooling_summary"}, indent=2)[:6000])
    print("... (truncated; full detail in representation_statistics.json)")


if __name__ == "__main__":
    main()
