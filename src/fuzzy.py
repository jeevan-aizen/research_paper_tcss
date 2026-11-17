"""Three-class Mamdani fuzzy resolver for low-confidence predictions.

Fixes the original design flaw: the old resolver could only output Hate or
Neither, so every routed Offensive tweet was guaranteed to be wrong. Here the
output universe (severity 0-10) has three consequents (Neither, Offensive,
Hate) and the inputs include the classifier's own probabilities as well as
VADER negativity.

Inputs
    pH   classifier P(hate)          in [0, 1]
    pN   classifier P(neither)       in [0, 1]
    neg  VADER negative proportion   in [0, 1]
"""
import numpy as np

from . import config as C

PROB_MF = {"low": (0.0, 0.0, 0.4), "med": (0.2, 0.5, 0.8), "high": (0.6, 1.0, 1.0)}
OUT_MF = {C.NEITHER: (0.0, 0.0, 4.0), C.OFFENSIVE: (3.0, 5.0, 7.0), C.HATE: (6.0, 10.0, 10.0)}
CUTS = (3.5, 6.5)   # severity < 3.5 -> Neither, < 6.5 -> Offensive, else Hate

# (antecedents, consequent). Antecedent = list of (input, level) ANDed with min.
# Level "notlow" = max(med, high).
RULES = [
    ([("pH", "high")], C.HATE),                                   # R1
    ([("pH", "med"), ("neg", "high")], C.HATE),                   # R2
    ([("pH", "med"), ("neg", "med")], C.OFFENSIVE),               # R3
    ([("pH", "med"), ("neg", "low")], C.OFFENSIVE),               # R4
    ([("pH", "med"), ("pN", "high")], C.NEITHER),                 # R5
    ([("pH", "low"), ("pN", "high")], C.NEITHER),                 # R6
    ([("pH", "low"), ("pN", "med"), ("neg", "low")], C.NEITHER),  # R7
    ([("pH", "low"), ("pN", "med"), ("neg", "notlow")], C.OFFENSIVE),  # R8
    ([("pH", "low"), ("pN", "low")], C.OFFENSIVE),                # R9
]


def tri(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= a).astype(float))
    right = np.where(c > b, (c - x) / (c - b + 1e-12), (x <= c).astype(float))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def neg_mf_from_train(neg_train: np.ndarray) -> dict:
    """VADER 'neg' is concentrated near 0, so fixed [0,1] breakpoints would put
    almost everything in 'low'. Breakpoints are set from TRAINING-set
    quantiles (33rd/66th/90th percentile of non-zero scores)."""
    nz = neg_train[neg_train > 0]
    q33, q66, q90 = np.quantile(nz, [0.33, 0.66, 0.90]) if len(nz) else (0.1, 0.2, 0.4)
    return {"low": (0.0, 0.0, q66), "med": (q33 / 2, q66, q90), "high": (q66, q90, 1.0)}


def _memberships(x, mf):
    m = {k: tri(x, *v) for k, v in mf.items()}
    m["notlow"] = np.maximum(m["med"], m["high"])
    return m


def fuzzy_resolve(probs: np.ndarray, neg: np.ndarray, neg_mf: dict, grid_pts: int = 201):
    """Return (labels, severity, rule_strengths[n, 9])."""
    inputs = {"pH": _memberships(probs[:, C.HATE], PROB_MF),
              "pN": _memberships(probs[:, C.NEITHER], PROB_MF),
              "neg": _memberships(neg, neg_mf)}
    strengths = np.stack([np.min(np.stack([inputs[i][lvl] for i, lvl in ante]), axis=0)
                          for ante, _ in RULES], axis=1)                # (n, R)
    grid = np.linspace(0, 10, grid_pts)
    out_mf = {k: tri(grid, *v) for k, v in OUT_MF.items()}
    agg = np.zeros((len(probs), grid_pts))
    for r, (_, cons) in enumerate(RULES):                               # Mamdani: min-implication,
        agg = np.maximum(agg, np.minimum(strengths[:, [r]], out_mf[cons][None, :]))  # max-aggregation
    mass = agg.sum(1)
    severity = np.where(mass > 0, (agg * grid).sum(1) / np.maximum(mass, 1e-12), np.nan)
    labels = np.where(severity < CUTS[0], C.NEITHER,
                      np.where(severity < CUTS[1], C.OFFENSIVE, C.HATE))
    # no rule fired -> fall back to the classifier's argmax
    labels = np.where(np.isnan(severity), probs.argmax(1), labels)
    return labels.astype(int), severity, strengths


def vader_neg(texts) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(str(t))["neg"] for t in texts])
