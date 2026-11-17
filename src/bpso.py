"""Binary PSO feature selection (Kennedy & Eberhart 1997) with linearly
decreasing inertia (Shi & Eberhart 1998), velocity clamping and a ring
neighbourhood. Fitness is computed on the VALIDATION split only; the test
split is never passed to this module."""
import hashlib

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

from . import config as C
from .classical import make_model


def _fitness(mask, Xtr, ytr, Xva, yva, base, seed, alpha):
    n_sel = int(mask.sum())
    if n_sel == 0:
        return 1.0, 0.0
    cols = np.flatnonzero(mask)
    m = make_model(base, seed).fit(Xtr[:, cols], ytr)
    f1 = f1_score(yva, m.predict(Xva[:, cols]), average="macro", zero_division=0)
    return alpha * (1 - f1) + (1 - alpha) * n_sel / mask.size, f1


def run_bpso(Xtr, ytr, Xva, yva, base="lr_bal", seed=42, n_jobs=-1, cfg=None, log=print):
    cfg = {**C.BPSO, **(cfg or {})}
    rng = np.random.default_rng(seed)
    n, d = cfg["n_particles"], Xtr.shape[1]
    X = (rng.random((n, d)) < 0.5).astype(np.int8)
    V = rng.uniform(-1, 1, (n, d))
    cache: dict[str, tuple[float, float]] = {}

    def evaluate(P):
        keys = [hashlib.md5(p.tobytes()).hexdigest() for p in P]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        res = Parallel(n_jobs=n_jobs)(
            delayed(_fitness)(P[i], Xtr, ytr, Xva, yva, base, seed, cfg["alpha"]) for i in todo)
        for i, r in zip(todo, res):
            cache[keys[i]] = r
        return np.array([cache[k][0] for k in keys]), np.array([cache[k][1] for k in keys])

    fit, f1 = evaluate(X)
    pbest, pbest_fit, pbest_f1 = X.copy(), fit.copy(), f1.copy()
    half = cfg["ring_k"] // 2
    history = []
    for t in range(cfg["n_iters"]):
        w = cfg["w_start"] - (cfg["w_start"] - cfg["w_end"]) * t / max(1, cfg["n_iters"] - 1)
        # ring neighbourhood best for each particle
        lbest = np.empty_like(X)
        for i in range(n):
            nb = [(i + o) % n for o in range(-half, half + 1)]
            lbest[i] = pbest[nb[int(np.argmin(pbest_fit[nb]))]]
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        V = w * V + cfg["c1"] * r1 * (pbest - X) + cfg["c2"] * r2 * (lbest - X)
        V = np.clip(V, -cfg["v_max"], cfg["v_max"])
        X = (rng.random((n, d)) < 1 / (1 + np.exp(-V))).astype(np.int8)
        fit, f1 = evaluate(X)
        better = fit < pbest_fit
        pbest[better], pbest_fit[better], pbest_f1[better] = X[better], fit[better], f1[better]
        g = int(np.argmin(pbest_fit))
        history.append(dict(iter=t + 1, w=round(w, 4), best_fitness=float(pbest_fit[g]),
                            best_val_f1=float(pbest_f1[g]), n_features=int(pbest[g].sum())))
        log(f"  BPSO iter {t+1:02d}  fit={pbest_fit[g]:.4f}  valF1={pbest_f1[g]:.4f}  k={pbest[g].sum()}")
    g = int(np.argmin(pbest_fit))
    return pbest[g].astype(bool), history
