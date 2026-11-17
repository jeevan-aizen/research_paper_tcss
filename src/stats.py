"""Metrics and statistics. zero_division is fixed to 0 everywhere: a class that
is never predicted gets precision 0, which is stated in the paper."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from . import config as C


def metrics(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": p.mean(), "recall_macro": r.mean(),
        "f1_macro": f.mean(),
        "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        **{f"f1_{C.LABELS[k]}": f[k] for k in range(3)},
        **{f"recall_{C.LABELS[k]}": r[k] for k in range(3)},
        **{f"precision_{C.LABELS[k]}": p[k] for k in range(3)},
    }


def bootstrap_ci(y, pred, fn=None, n=C.N_BOOTSTRAP, seed=0, alpha=0.05):
    fn = fn or (lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = np.array([fn(y[i], pred[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap(y, pred_a, pred_b, n=C.N_BOOTSTRAP, seed=0):
    """Two-sided p-value for H0: macro-F1(A) == macro-F1(B) on the same items."""
    rng = np.random.default_rng(seed)
    f = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
    obs = f(y, pred_a) - f(y, pred_b)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = np.array([f(y[i], pred_a[i]) - f(y[i], pred_b[i]) for i in idx])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(min(1.0, p))


def mcnemar(y, pred_a, pred_b):
    """Exact McNemar test on per-item correctness."""
    a, b = pred_a == y, pred_b == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for k repeated
    random splits, which overlap and so violate the plain t-test."""
    from scipy.stats import t as tdist
    d = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(d)
    var = d.var(ddof=1)
    if k < 2 or var == 0:
        return float(d.mean()), 1.0
    stat = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(d.mean()), float(2 * tdist.sf(abs(stat), df=k - 1))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjustment for a family of comparisons."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
