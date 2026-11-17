"""Who bears the errors of automated moderation?

1. Dialect (Davidson). Tweets are assigned a dialect group with the
   demographic dialect model of Blodgett et al. (2016). We compare
   AAE-aligned and White-aligned tweets on: rate flagged as hate, false
   hate-flag rate among gold non-hate tweets, false abusive-flag rate
   (hate or offensive) among gold 'neither' tweets, and the share of each
   group's posts deferred to human review.

2. Target group (HateXplain). For each targeted community, the share of hate
   posts against that community that the system misses.
"""
import sys

import numpy as np
import pandas as pd

from . import config as C


def dialect_posteriors(texts, twitteraae_repo: str) -> np.ndarray:
    """Return (n, 4) posteriors [AA, Hispanic, Other, White] using the
    released model from github.com/slanglab/twitteraae (clone it and pass
    the repo path). Rows with no in-vocabulary token are NaN."""
    import os
    sys.path.insert(0, f"{twitteraae_repo}/code")
    import predict  # noqa: E402  (module shipped with the TwitterAAE repo)
    cwd = os.getcwd()
    os.chdir(f"{twitteraae_repo}/code")   # predict.py loads ../model/ relative to code/
    try:
        predict.load_model()
    finally:
        os.chdir(cwd)
    out = np.full((len(texts), 4), np.nan)
    for i, t in enumerate(texts):
        p = predict.predict(str(t).split())
        if p is not None:
            out[i] = p
    return out


def assign_dialect(post: np.ndarray, thr=C.DIALECT_THRESHOLD) -> np.ndarray:
    g = np.full(len(post), "other", dtype=object)
    g[post[:, 0] >= thr] = "AAE"
    g[post[:, 3] >= thr] = "WAE"   # White-aligned English
    return g


def dialect_table(y, pred, groups, deferred=None) -> pd.DataFrame:
    rows = []
    for g in ["AAE", "WAE"]:
        m = groups == g
        if not m.any():
            continue
        nonhate = m & (y != C.HATE)
        neither = m & (y == C.NEITHER)
        rows.append({
            "group": g, "n": int(m.sum()),
            "gold_hate_rate": float((y[m] == C.HATE).mean()),
            "pred_hate_rate": float((pred[m] == C.HATE).mean()),
            "false_hate_flag_rate": float((pred[nonhate] == C.HATE).mean()) if nonhate.any() else np.nan,
            "false_abusive_flag_rate": float((pred[neither] != C.NEITHER).mean()) if neither.any() else np.nan,
            "deferral_rate": float(deferred[m].mean()) if deferred is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if len(df) == 2:
        ratio = {"group": "AAE/WAE ratio", "n": np.nan}
        for c in df.columns[2:]:
            a, w = df.loc[0, c], df.loc[1, c]
            ratio[c] = a / w if w else np.nan
        df = pd.concat([df, pd.DataFrame([ratio])], ignore_index=True)
    return df


def target_table(y, pred, targets: pd.Series, min_hate=30) -> pd.DataFrame:
    rows = []
    groups = sorted({t for s in targets for t in s.split(";") if t})
    for g in groups:
        m = targets.str.split(";").apply(lambda xs: g in xs).to_numpy()
        hate = m & (y == C.HATE)
        if hate.sum() < min_hate:
            continue
        rows.append({"target": g, "n_hate": int(hate.sum()),
                     "missed_hate_rate": float((pred[hate] != C.HATE).mean())})
    return pd.DataFrame(rows).sort_values("missed_hate_rate", ascending=False)
