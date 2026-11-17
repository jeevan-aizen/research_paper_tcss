"""Central configuration. Every experiment reads from here so the paper's
Table of settings can be generated from one place."""
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parents[1]
# On Colab, point these at Google Drive so results survive disconnects:
#   os.environ["CH_DATA_DIR"] = "/content/drive/MyDrive/cyberhate/data"
DATA_DIR = Path(os.environ.get("CH_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("CH_RESULTS_DIR", ROOT / "results"))
# Transformer checkpoints are large and temporary; keep them on local disk.
CKPT_DIR = Path(os.environ.get("CH_CKPT_DIR", ROOT / "ckpt"))

# Five independent stratified splits. Each seed controls the split AND model init.
SEEDS = [42, 123, 456, 789, 1011]

# Split proportions (of the full dataset). Validation is carved out of the
# training portion and is the ONLY data used for PSO fitness, GA selection,
# temperature scaling and threshold choice. The test split is touched once.
TEST_SIZE = 0.20
VAL_SIZE = 0.10

LABELS = {0: "hate", 1: "offensive", 2: "neither"}
HATE, OFFENSIVE, NEITHER = 0, 1, 2

# ---- Stage 1: TF-IDF ----
TFIDF = dict(max_features=5000, ngram_range=(1, 2), stop_words="english",
             sublinear_tf=False, min_df=2)

# ---- Stage 2: classical models ----
MNB_ALPHA = 1.0
LR_C = 1.0
SVM_C = 1.0

# ---- Binary PSO (Shi & Eberhart linearly decreasing inertia, clamped velocity) ----
BPSO = dict(n_particles=20, n_iters=30, c1=2.0, c2=2.0,
            w_start=0.9, w_end=0.4, v_max=4.0, alpha=0.88, ring_k=5)

# ---- Genetic algorithm hyper-parameter search ----
GA = dict(pop_size=20, n_generations=10, cv_folds=5, tournament=3,
          mutation_rate=0.3, elite=2)

# ---- Transformers ----
TRANSFORMERS = {
    "hatebert": "GroNLP/hateBERT",
    "roberta": "roberta-base",
    "bertweet": "vinai/bertweet-base",
}
FT = dict(epochs=3, lr=2e-5, batch_size=16, max_len=128, weight_decay=0.01,
          warmup_ratio=0.06, fp16=True)

# ---- Zero-shot LLM ----
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"   # model used in the paper (16-bit)
LLM_CHUNK = 1000   # posts per cached chunk (resume point after a disconnect)

# ---- Triage ----
THETA_GRID = [round(x, 2) for x in [i / 100 for i in range(34, 100, 2)]]
# Cost of each (true, predicted) error, relative to a false flag of 1.0.
# Missed hate is swept in the analysis; this is only the default.
COST_MISSED_HATE = 5.0
COST_FALSE_FLAG = 1.0
COST_HUMAN_REVIEW = 0.2   # per item reviewed, relative units

# ---- Dialect fairness (Blodgett et al. TwitterAAE) ----
DIALECT_THRESHOLD = 0.50   # posterior share needed to assign a tweet to a group (0.6, 0.7 reported as robustness)

N_BOOTSTRAP = 1000
