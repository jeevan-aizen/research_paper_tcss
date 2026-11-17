"""Genetic-algorithm hyper-parameter search over the same space as the
original TPOT run (MNB alpha, LR C), extended with the class-weighting switch.
Fitness = mean macro-F1 of stratified k-fold CV on the TRAINING split only.

A small self-contained GA is used instead of TPOT so the search is fully
specified in the paper and does not depend on TPOT's changing API."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C
from .classical import make_model

SPACE = {"mnb": (-3.0, 2.0),    # log10 alpha in [1e-3, 1e2]
         "lr": (-4.0, np.log10(25.0))}  # log10 C in [1e-4, 25]


def _decode(g):
    fam, logv, bal = g
    name = fam + ("_bal" if bal else "")
    param = {"alpha": 10 ** logv} if fam == "mnb" else {"C": 10 ** logv}
    return name, param


def _random_genome(rng):
    fam = rng.choice(list(SPACE))
    lo, hi = SPACE[fam]
    return (fam, float(rng.uniform(lo, hi)), bool(rng.random() < 0.5))


def run_ga(Xtr, ytr, seed=42, cfg=None, n_jobs=-1, log=print):
    cfg = {**C.GA, **(cfg or {})}
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=seed)
    cache = {}

    def fitness(g):
        key = (g[0], round(g[1], 4), g[2])
        if key not in cache:
            name, p = _decode(g)
            cache[key] = float(cross_val_score(make_model(name, seed, **p), Xtr, ytr, cv=cv,
                                               scoring="f1_macro", n_jobs=n_jobs).mean())
        return cache[key]

    pop = [_random_genome(rng) for _ in range(cfg["pop_size"])]
    history = []
    for gen in range(cfg["n_generations"]):
        scores = np.array([fitness(g) for g in pop])
        order = np.argsort(-scores)
        best = pop[order[0]]
        history.append(dict(generation=gen + 1, best_cv_f1=float(scores[order[0]]),
                            mean_cv_f1=float(scores.mean()), best=_decode(best)))
        log(f"  GA gen {gen+1:02d}  best={scores[order[0]]:.4f}  {_decode(best)}")
        new = [pop[i] for i in order[:cfg["elite"]]]
        while len(new) < cfg["pop_size"]:
            def tour():
                c = rng.choice(len(pop), cfg["tournament"], replace=False)
                return pop[c[np.argmax(scores[c])]]
            a, b = tour(), tour()
            fam = a[0] if rng.random() < 0.5 else b[0]
            # blend crossover on the log-parameter when families match
            if a[0] == b[0]:
                v = a[1] + rng.random() * (b[1] - a[1])
            else:
                v = a[1] if fam == a[0] else b[1]
            bal = a[2] if rng.random() < 0.5 else b[2]
            if rng.random() < cfg["mutation_rate"]:
                v += rng.normal(0, 0.5)
            if rng.random() < cfg["mutation_rate"] / 3:
                bal = not bal
            lo, hi = SPACE[fam]
            new.append((fam, float(np.clip(v, lo, hi)), bal))
        pop = new
    scores = np.array([fitness(g) for g in pop])
    best = pop[int(np.argmax(scores))]
    return _decode(best), history
