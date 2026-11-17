"""Human-AI moderation triage.

A moderation system receives every post. The model handles the posts it is
confident about; the least-confident fraction (the human review budget b) is
deferred to a human moderator. We simulate the human with the real annotator
votes: a deferred post receives the label of ONE randomly drawn annotator of
that post, so the simulated moderator makes realistic, item-specific errors
instead of being an oracle. An oracle variant (gold label) is reported as an
upper bound.

Three system designs are compared:
  single   one model + human review of the b least-confident posts
  cascade  cheap TF-IDF model first; posts below its confidence threshold are
           escalated to a transformer; human reviews the b least-confident
           posts at the final stage. Reports the share of posts needing GPU.
  fuzzy    model + Mamdani fuzzy resolver on low-confidence posts, no human
           (the configuration of the original multi-stage pipeline).
"""
import numpy as np

from . import config as C
from .stats import metrics


def cost_matrix(c_miss=C.COST_MISSED_HATE, c_ff=C.COST_FALSE_FLAG, c_conf=0.5):
    """cost[true, pred]. Missing hate costs c_miss; flagging non-hate as hate
    costs c_ff; offensive/neither confusion costs c_conf."""
    M = np.full((3, 3), c_conf)
    np.fill_diagonal(M, 0.0)
    M[C.HATE, [C.OFFENSIVE, C.NEITHER]] = c_miss
    M[[C.OFFENSIVE, C.NEITHER], C.HATE] = c_ff
    return M


def uncertainty(probs, kind="maxprob"):
    if kind == "maxprob":
        return 1 - probs.max(1)
    if kind == "entropy":
        p = np.clip(probs, 1e-12, 1)
        return -(p * np.log(p)).sum(1)
    raise ValueError(kind)


def defer_by_budget(probs, budget, kind="maxprob"):
    """Boolean mask of the `budget` fraction of least-confident items."""
    k = int(round(budget * len(probs)))
    mask = np.zeros(len(probs), bool)
    if k > 0:
        mask[np.argsort(-uncertainty(probs, kind), kind="stable")[:k]] = True
    return mask


def human_labels(votes, rng, oracle_y=None):
    if oracle_y is not None:
        return oracle_y.copy()
    p = votes / votes.sum(1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return (u > np.cumsum(p, 1)).sum(1).clip(0, 2)


def _summary(y, pred, deferred, cm, gpu_frac=0.0):
    m = metrics(y, pred)
    hate = y == C.HATE
    nonhate = ~hate
    return {
        **m,
        "deferral_rate": float(deferred.mean()),
        "reviews_per_10k": float(deferred.mean() * 10_000),
        "missed_hate_rate": float((pred[hate] != C.HATE).mean()) if hate.any() else np.nan,
        "false_flag_rate": float((pred[nonhate] == C.HATE).mean()),
        "cost_per_item": float(cm[y, pred].mean() + C.COST_HUMAN_REVIEW * deferred.mean()),
        "gpu_fraction": float(gpu_frac),
    }


def simulate_single(y, probs, votes, budget, n_draws=20, seed=0, oracle=False, cm=None):
    cm = cost_matrix() if cm is None else cm
    rng = np.random.default_rng(seed)
    deferred = defer_by_budget(probs, budget)
    rows = []
    for _ in range(1 if oracle else n_draws):
        pred = probs.argmax(1).copy()
        h = human_labels(votes[deferred], rng, y[deferred] if oracle else None)
        pred[deferred] = h
        rows.append(_summary(y, pred, deferred, cm))
    return _mean_rows(rows)


def choose_cascade_threshold(val_cheap, val_strong, y_val, tol=0.005, grid=None):
    """Lowest escalation share whose validation macro-F1 is within `tol` of
    running the strong model on everything. Chosen on validation only."""
    grid = grid or C.THETA_GRID
    full = metrics(y_val, val_strong.argmax(1))["f1_macro"]
    best = (1.0, None)
    for th in grid:
        esc = val_cheap.max(1) < th
        pred = np.where(esc, val_strong.argmax(1), val_cheap.argmax(1))
        if metrics(y_val, pred)["f1_macro"] >= full - tol and esc.mean() < best[0]:
            best = (esc.mean(), th)
    return best[1] if best[1] is not None else 1.01


def simulate_cascade(y, cheap, strong, votes, theta, budget, n_draws=20, seed=0,
                     oracle=False, cm=None):
    cm = cost_matrix() if cm is None else cm
    rng = np.random.default_rng(seed)
    esc = cheap.max(1) < theta
    final = np.where(esc[:, None], strong, cheap)
    deferred = defer_by_budget(final, budget)
    rows = []
    for _ in range(1 if oracle else n_draws):
        pred = final.argmax(1).copy()
        pred[deferred] = human_labels(votes[deferred], rng, y[deferred] if oracle else None)
        rows.append(_summary(y, pred, deferred, cm, gpu_frac=esc.mean()))
    return _mean_rows(rows)


def simulate_fuzzy(y, probs, fuzzy_labels, theta, cm=None):
    """Posts with max-probability < theta take the fuzzy label; no human."""
    cm = cost_matrix() if cm is None else cm
    routed = probs.max(1) < theta
    pred = np.where(routed, fuzzy_labels, probs.argmax(1))
    out = _summary(y, pred, np.zeros(len(y), bool), cm)
    out["routed_rate"] = float(routed.mean())
    out["routed_by_class"] = {C.LABELS[k]: int((routed & (y == k)).sum()) for k in range(3)}
    return out


def _mean_rows(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def annotator_agreement_ceiling(y, votes, n_draws=50, seed=0):
    """Macro-F1 of a single random annotator against the majority label: an
    empirical ceiling on what the 'human' in the loop achieves."""
    rng = np.random.default_rng(seed)
    return float(np.mean([metrics(y, human_labels(votes, rng))["f1_macro"]
                          for _ in range(n_draws)]))
