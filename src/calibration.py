"""Confidence calibration. Triage depends on max-probability being meaningful,
so every model is temperature-scaled on the validation split (Guo et al. 2017)
before thresholds are applied. Log-probabilities are used as logits, which
works for any model that outputs probabilities."""
import numpy as np
from scipy.optimize import minimize_scalar


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(val_probs, y_val) -> float:
    logits = np.log(np.clip(val_probs, 1e-12, 1))

    def nll(T):
        p = _softmax(logits / T)
        return -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1)))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(probs, T):
    return _softmax(np.log(np.clip(probs, 1e-12, 1)) / T)


def ece(probs, y, n_bins=15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)
