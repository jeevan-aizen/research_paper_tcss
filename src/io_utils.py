"""Every model writes class probabilities for val and test. All evaluation,
triage and fairness analysis reads only these files, so a single change to a
model never requires re-running the analysis code by hand."""
import json
from pathlib import Path

import numpy as np

from . import config as C


def pred_dir(dataset: str, model: str, seed: int) -> Path:
    d = C.RESULTS_DIR / dataset / model / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_preds(dataset, model, seed, val_probs, test_probs, meta=None):
    d = pred_dir(dataset, model, seed)
    for name, p in [("val_probs", val_probs), ("test_probs", test_probs)]:
        p = np.asarray(p, dtype=float)
        assert p.ndim == 2 and p.shape[1] == 3, f"{name} must be (n, 3), got {p.shape}"
        assert np.allclose(p.sum(1), 1.0, atol=1e-4), f"{name} rows must sum to 1"
        np.save(d / f"{name}.npy", p)
    (d / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))


def load_preds(dataset, model, seed):
    d = pred_dir(dataset, model, seed)
    return np.load(d / "val_probs.npy"), np.load(d / "test_probs.npy")


def available_models(dataset: str) -> list[str]:
    root = C.RESULTS_DIR / dataset
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and any((q / "test_probs.npy").exists() for q in p.glob("seed*")))


def done(dataset: str, model: str, seed: int) -> bool:
    """True if this (dataset, model, seed) already has saved test predictions,
    so interrupted Colab sessions can resume where they stopped."""
    return (C.RESULTS_DIR / dataset / model / f"seed{seed}" / "test_probs.npy").exists()
