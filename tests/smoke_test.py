"""End-to-end smoke test on SYNTHETIC data (no downloads, no GPU).
Checks that every CPU stage runs and writes its outputs. The numbers it
produces are meaningless and must never appear in the paper.

    python tests/smoke_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def synth_davidson(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    vocab = {0: ["vile", "subhuman", "scum", "go back"], 1: ["damn", "trash", "stupid", "hell"],
             2: ["sunny", "coffee", "game", "music"]}
    filler = ["the", "a", "today", "really", "just", "lol", "people", "this", "that", "so"]
    labels = rng.choice(3, n, p=[0.06, 0.77, 0.17])
    rows = []
    for lab in labels:
        words = list(rng.choice(filler, 6)) + list(rng.choice(vocab[lab], 2))
        if rng.random() < 0.3:      # label noise / overlap
            words += list(rng.choice(vocab[rng.integers(3)], 1))
        cnt = 3
        v = rng.multinomial(cnt, np.eye(3)[lab] * 0.8 + 0.2 / 3)
        rows.append({"count": cnt, "hate_speech": v[0], "offensive_language": v[1], "neither": v[2],
                     "class": int(lab), "tweet": "@user " + " ".join(words) + " http://t.co/x"})
    return pd.DataFrame(rows)


def synth_hatexplain(n=600, seed=1):
    rng = np.random.default_rng(seed)
    names = ["hatespeech", "offensive", "normal"]
    groups = ["African", "Women", "Islam", "Jewish"]
    out = {}
    for i in range(n):
        lab = rng.choice(3, p=[0.3, 0.3, 0.4])
        tgt = [str(rng.choice(groups))] if lab < 2 else ["None"]
        ann = [{"label": names[lab if rng.random() < 0.8 else rng.integers(3)], "annotator_id": j,
                "target": tgt} for j in range(3)]
        toks = ["word"] * 4 + [["hateful", "slur"][rng.integers(2)] if lab == 0 else
                               ["rude", "crap"][rng.integers(2)] if lab == 1 else "nice"]
        out[f"p{i}"] = {"post_id": f"p{i}", "annotators": ann, "rationales": [], "post_tokens": toks}
    return out


def main():
    from src import config as C
    tmp = Path(tempfile.mkdtemp())
    C.DATA_DIR, C.RESULTS_DIR = tmp / "data", tmp / "results"
    csv = tmp / "labeled_data.csv"
    synth_davidson().to_csv(csv, index=False)
    hx = tmp / "dataset.json"
    hx.write_text(json.dumps(synth_hatexplain()))

    import run
    run.TABLES, run.FIGS = C.RESULTS_DIR / "tables", C.RESULTS_DIR / "figures"
    C.BPSO.update(n_particles=6, n_iters=3)
    C.GA.update(pop_size=6, n_generations=2, cv_folds=3)

    class A:  # minimal argparse stand-in
        seeds = "42,123,456"; davidson_csv = str(csv); hatexplain_json = str(hx); base = "lr_bal"
        leaky = False; model = "hatebert"; weighted = False; llm = None; twitteraae = None
        cheap = "lr_bal"; strong = "hatebert"; min_hate = 5; force = False; llm_4bit = False

    for ds in ["davidson", "hatexplain"]:
        A.dataset = ds
        run.cmd_prepare(A)
        df = pd.read_pickle(C.DATA_DIR / ds / "processed.pkl")
        if "vader_neg" not in df:   # VADER not installed in the test box
            df["vader_neg"] = np.random.default_rng(0).beta(1, 6, len(df))
            df.to_pickle(C.DATA_DIR / ds / "processed.pkl")
        run.cmd_classical(A)
        run.cmd_classical(A)   # second call must skip (resume)
        run.cmd_bpso(A)
        A.leaky = True; run.cmd_bpso(A); A.leaky = False
        run.cmd_ga(A)
        run.cmd_randsub(A)
        # fake "transformer" outputs so the cascade code path is exercised
        from src.io_utils import save_preds
        from src.data import load_split
        y = df["label"].values
        for s in [42, 123, 456]:
            sp = load_split(ds, s)
            def fake(idx):
                p = np.full((len(idx), 3), 0.1); p[np.arange(len(idx)), y[idx]] = 0.8
                p += np.random.default_rng(s).random(p.shape) * 0.3
                return p / p.sum(1, keepdims=True)
            save_preds(ds, "hatebert", s, fake(sp["val"]), fake(sp["test"]), {"fake": True})
        if ds == "davidson":   # fake dialect posteriors
            post = np.random.default_rng(0).dirichlet([1, 0.3, 0.3, 1], len(df))
            np.save(C.DATA_DIR / ds / "dialect_posteriors.npy", post)
        run.cmd_evaluate(A)
        run.cmd_triage(A)
        run.cmd_fairness(A)

    produced = sorted(p.name for p in (C.RESULTS_DIR / "tables").iterdir())
    figs = sorted(p.name for p in (C.RESULTS_DIR / "figures").iterdir())
    print("\nTABLES:", produced, "\nFIGURES:", figs)
    expected = {"davidson_summary.csv", "davidson_triage.csv", "davidson_fuzzy.csv",
                "davidson_fairness_dialect.csv", "hatexplain_fairness_target.csv",
                "davidson_significance.csv", "davidson_cost_sweep.csv"}
    missing = expected - set(produced)
    assert not missing, f"missing outputs: {missing}"
    fz = pd.read_csv(C.RESULTS_DIR / "tables" / "davidson_fuzzy.csv")
    assert (fz["routed_offensive"] >= 0).all()
    shutil.rmtree(tmp)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
