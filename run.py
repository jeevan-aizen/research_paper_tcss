"""Single entry point for every experiment in the paper.

    python run.py prepare    --dataset davidson [--davidson-csv labeled_data.csv]
    python run.py prepare    --dataset hatexplain --hatexplain-json dataset.json
    python run.py classical  --dataset davidson
    python run.py bpso       --dataset davidson --base lr_bal [--leaky]
    python run.py ga         --dataset davidson
    python run.py transformer --dataset davidson --model hatebert [--weighted]
    python run.py llm        --dataset davidson
    python run.py dialect    --dataset davidson --twitteraae /path/to/twitteraae
    python run.py evaluate   --dataset davidson
    python run.py triage     --dataset davidson
    python run.py fairness   --dataset davidson
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src.io_utils import available_models, done, load_preds, save_preds

TABLES = C.RESULTS_DIR / "tables"
FIGS = C.RESULTS_DIR / "figures"


def _seeds(a):
    return [int(s) for s in a.seeds.split(",")] if a.seeds else C.SEEDS


def _texts(df, idx, col="text_clf"):
    return df[col].values[idx]


# ------------------------------------------------------------------ prepare
def cmd_prepare(a):
    desc_path = C.DATA_DIR / a.dataset / "description.json"
    if desc_path.exists() and not a.force:
        print(f"[{a.dataset}] already prepared (use --force to redo):")
        print(desc_path.read_text())
        return
    if a.dataset == "davidson":
        df = D.load_davidson(a.davidson_csv)
    elif a.dataset == "hatexplain":
        df = D.load_hatexplain(a.hatexplain_json)
    else:
        raise SystemExit("unknown dataset")
    try:
        from src.fuzzy import vader_neg
        df["vader_neg"] = vader_neg(df["text_raw"])
    except ImportError:
        print("! vaderSentiment not installed; fuzzy stage will be unavailable")
    out = C.DATA_DIR / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out / "processed.pkl")
    desc = D.describe(df)
    for s in C.SEEDS:
        sp = D.make_split(df, s)
        D.save_split(a.dataset, s, sp)
        desc.setdefault("split_sizes", {})[s] = {k: int(len(v)) for k, v in sp.items()}
    (out / "description.json").write_text(json.dumps(desc, indent=2))
    print(json.dumps(desc, indent=2))


# ---------------------------------------------------------------- classical
def cmd_classical(a):
    from src.classical import CLASSICAL_MODELS, fit_tfidf, make_model, predict_proba_3
    df = D.load_processed(a.dataset)
    y = df["label"].values
    for s in _seeds(a):
        if all(done(a.dataset, m, s) for m in CLASSICAL_MODELS) and not a.force:
            print(f"[{a.dataset} seed {s}] classical already done, skipping"); continue
        sp = D.load_split(a.dataset, s)
        vec, Xtr = fit_tfidf(_texts(df, sp["train"]))
        Xva, Xte = vec.transform(_texts(df, sp["val"])), vec.transform(_texts(df, sp["test"]))
        for name in CLASSICAL_MODELS:
            t0 = time.time()
            m = make_model(name, s).fit(Xtr, y[sp["train"]])
            fit_s = time.time() - t0
            t1 = time.time()
            pte = predict_proba_3(m, Xte)
            inf_s = time.time() - t1
            save_preds(a.dataset, name, s, predict_proba_3(m, Xva), pte,
                       {"fit_seconds": round(fit_s, 3), "ms_per_post": round(1000 * inf_s / Xte.shape[0], 4),
                        "hardware": "CPU", "vocab_size": len(vec.vocabulary_)})
            print(f"[{a.dataset} seed {s}] {name:8s} fit {fit_s:6.1f}s")


# --------------------------------------------------------------------- bpso
def cmd_bpso(a):
    from src.bpso import run_bpso
    from src.classical import fit_tfidf, make_model, predict_proba_3
    df = D.load_processed(a.dataset)
    y = df["label"].values
    name = f"bpso_{a.base}" + ("_LEAKY" if a.leaky else "")
    for s in _seeds(a):
        if done(a.dataset, name, s) and not a.force:
            print(f"[{a.dataset} seed {s}] {name} already done, skipping"); continue
        sp = D.load_split(a.dataset, s)
        vec, Xtr = fit_tfidf(_texts(df, sp["train"]))
        Xva, Xte = vec.transform(_texts(df, sp["val"])), vec.transform(_texts(df, sp["test"]))
        # --leaky reproduces the ORIGINAL protocol (fitness on test) only to
        # measure how much it inflates results. Never reported as a result.
        Xfit, yfit = (Xte, y[sp["test"]]) if a.leaky else (Xva, y[sp["val"]])
        t0 = time.time()
        mask, hist = run_bpso(Xtr, y[sp["train"]], Xfit, yfit, base=a.base, seed=s)
        cols = np.flatnonzero(mask)
        m = make_model(a.base, s).fit(Xtr[:, cols], y[sp["train"]])
        save_preds(a.dataset, name, s, predict_proba_3(m, Xva[:, cols]), predict_proba_3(m, Xte[:, cols]),
                   {"n_features": int(mask.sum()), "seconds": round(time.time() - t0, 1),
                    "history": hist, "leaky": a.leaky})


def cmd_randsub(a):
    """Control for BPSO: a RANDOM feature subset of the same size that BPSO
    selected on the same split. If BPSO ~= random, the swarm did not learn."""
    from src.classical import fit_tfidf, make_model, predict_proba_3
    df = D.load_processed(a.dataset)
    y = df["label"].values
    name = f"rand_{a.base}"
    for s in _seeds(a):
        if done(a.dataset, name, s) and not a.force:
            print(f"[{a.dataset} seed {s}] {name} already done, skipping"); continue
        meta = json.loads((C.RESULTS_DIR / a.dataset / f"bpso_{a.base}" / f"seed{s}" / "meta.json").read_text())
        k = int(meta["n_features"])
        sp = D.load_split(a.dataset, s)
        vec, Xtr = fit_tfidf(_texts(df, sp["train"]))
        Xva, Xte = vec.transform(_texts(df, sp["val"])), vec.transform(_texts(df, sp["test"]))
        cols = np.sort(np.random.default_rng(s + 7).choice(Xtr.shape[1], k, replace=False))
        m = make_model(a.base, s).fit(Xtr[:, cols], y[sp["train"]])
        save_preds(a.dataset, name, s, predict_proba_3(m, Xva[:, cols]), predict_proba_3(m, Xte[:, cols]),
                   {"n_features": k, "matched_to": f"bpso_{a.base}"})
        print(f"[{a.dataset} seed {s}] {name}: random {k} features")


# ----------------------------------------------------------------------- ga
def cmd_ga(a):
    from src.classical import fit_tfidf, make_model, predict_proba_3
    from src.ga import run_ga
    df = D.load_processed(a.dataset)
    y = df["label"].values
    for s in _seeds(a):
        if done(a.dataset, "ga_best", s) and not a.force:
            print(f"[{a.dataset} seed {s}] ga_best already done, skipping"); continue
        sp = D.load_split(a.dataset, s)
        vec, Xtr = fit_tfidf(_texts(df, sp["train"]))
        Xva, Xte = vec.transform(_texts(df, sp["val"])), vec.transform(_texts(df, sp["test"]))
        t0 = time.time()
        (name, params), hist = run_ga(Xtr, y[sp["train"]], seed=s)
        m = make_model(name, s, **params).fit(Xtr, y[sp["train"]])
        save_preds(a.dataset, "ga_best", s, predict_proba_3(m, Xva), predict_proba_3(m, Xte),
                   {"best_model": name, "params": params, "seconds": round(time.time() - t0, 1),
                    "history": hist})
        print(f"[{a.dataset} seed {s}] GA best: {name} {params}")


# ------------------------------------------------------------- transformers
def cmd_transformer(a):
    from src.transformers_ft import finetune
    df = D.load_processed(a.dataset)
    y = df["label"].values
    name = a.model + ("_w" if a.weighted else "")
    for s in _seeds(a):
        if done(a.dataset, name, s) and not a.force:
            print(f"[{a.dataset} seed {s}] {name} already done, skipping"); continue
        sp = D.load_split(a.dataset, s)
        out = C.CKPT_DIR / a.dataset / name / f"seed{s}"
        vp, tp, meta = finetune(a.model, _texts(df, sp["train"], "text_tf"), y[sp["train"]],
                                _texts(df, sp["val"], "text_tf"), y[sp["val"]],
                                _texts(df, sp["test"], "text_tf"), s, out, a.weighted)
        save_preds(a.dataset, name, s, vp, tp, meta)
        print(f"[{a.dataset} seed {s}] {name} done: {meta}")


def cmd_llm(a):
    from src.llm_zeroshot import PROMPT, score_all_cached
    df = D.load_processed(a.dataset)
    slug = a.llm.split("/")[-1] + ("-4bit" if a.llm_4bit else "")
    probs, secs = score_all_cached(list(df["text_tf"]), C.DATA_DIR / a.dataset / f"llm_cache_{slug}",
                                   a.llm, a.llm_4bit)
    for s in _seeds(a):
        sp = D.load_split(a.dataset, s)
        save_preds(a.dataset, "llm_zeroshot", s, probs[sp["val"]], probs[sp["test"]],
                   {"model": a.llm, "four_bit": a.llm_4bit, "prompt": PROMPT,
                    "scoring_seconds_this_session": round(secs, 1),
                    "ms_per_post": round(1000 * secs / len(df), 2) if secs else None})
    print(f"[{a.dataset}] LLM predictions saved for seeds {_seeds(a)}")


# ------------------------------------------------------------------ dialect
def cmd_dialect(a):
    from src.fairness import dialect_posteriors
    df = D.load_processed(a.dataset)
    post = dialect_posteriors(df["text_raw"].tolist(), a.twitteraae)
    np.save(C.DATA_DIR / a.dataset / "dialect_posteriors.npy", post)
    print("in-vocabulary tweets:", int((~np.isnan(post[:, 0])).sum()), "/", len(post))


# ---------------------------------------------------------------- evaluate
def _calibrated(dataset, model, seed, y_val):
    from src.calibration import apply_temperature, fit_temperature
    vp, tp = load_preds(dataset, model, seed)
    T = fit_temperature(vp, y_val)
    return apply_temperature(vp, T), apply_temperature(tp, T), T, tp


def cmd_evaluate(a):
    from src.calibration import ece
    from src.stats import bootstrap_ci, corrected_resampled_ttest, holm, mcnemar, metrics
    df = D.load_processed(a.dataset)
    y = df["label"].values
    rows = []
    preds = {}
    for model in available_models(a.dataset):
        for s in _seeds(a):
            try:
                sp = D.load_split(a.dataset, s)
                _, tp_cal, T, tp_raw = _calibrated(a.dataset, model, s, y[sp["val"]])
            except FileNotFoundError:
                continue
            yt = y[sp["test"]]
            pred = tp_raw.argmax(1)     # temperature scaling never changes argmax
            preds[(model, s)] = pred
            r = {"model": model, "seed": s, **metrics(yt, pred),
                 "ece_raw": ece(tp_raw, yt), "ece_cal": ece(tp_cal, yt), "temperature": T}
            if s == C.SEEDS[0]:
                r["f1_macro_ci_lo"], r["f1_macro_ci_hi"] = bootstrap_ci(yt, pred)
            rows.append(r)
    per_seed = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(TABLES / f"{a.dataset}_per_seed.csv", index=False)

    num = per_seed.drop(columns=["seed"]).groupby("model")
    summary = num.mean(numeric_only=True).add_suffix("_mean").join(
        num.std(numeric_only=True).add_suffix("_std"))
    summary["n_seeds"] = num.size()
    summary.to_csv(TABLES / f"{a.dataset}_summary.csv")

    # Significance vs reference models (corrected resampled t-test across
    # seeds + per-seed McNemar), Holm-adjusted within each reference family.
    sp0 = D.load_split(a.dataset, C.SEEDS[0])
    n_tr, n_te = len(sp0["train"]), len(sp0["test"])
    sig = []
    for ref in [r for r in ["lr_bal", "hatebert"] if r in summary.index]:
        raw = {}
        for m in summary.index:
            if m == ref:
                continue
            common = sorted(set(per_seed[per_seed.model == m].seed) & set(per_seed[per_seed.model == ref].seed))
            if len(common) < 2:
                continue
            fa = per_seed.set_index(["model", "seed"]).loc[[(m, s) for s in common], "f1_macro"].values
            fb = per_seed.set_index(["model", "seed"]).loc[[(ref, s) for s in common], "f1_macro"].values
            diff, p = corrected_resampled_ttest(fa, fb, n_tr, n_te)
            mc = [mcnemar(y[D.load_split(a.dataset, s)["test"]], preds[(m, s)], preds[(ref, s)])
                  for s in common]
            raw[m] = p
            sig.append({"reference": ref, "model": m, "delta_f1_macro": diff, "p_ttest": p,
                        "mcnemar_p_median": float(np.median(mc)),
                        "seeds_mcnemar_lt_0.05": int(sum(q < 0.05 for q in mc)), "n_seeds": len(common)})
        adj = holm(raw)
        for r in sig:
            if r["reference"] == ref and r["model"] in adj:
                r["p_holm"] = adj[r["model"]]
    pd.DataFrame(sig).to_csv(TABLES / f"{a.dataset}_significance.csv", index=False)
    print(summary[["f1_macro_mean", "f1_macro_std", "f1_hate_mean", "ece_cal_mean", "n_seeds"]]
          .sort_values("f1_macro_mean", ascending=False).round(4).to_string())


# ------------------------------------------------------------------ triage
BUDGETS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def cmd_triage(a):
    from src.fuzzy import fuzzy_resolve, neg_mf_from_train
    from src.stats import metrics
    from src.triage import (annotator_agreement_ceiling, choose_cascade_threshold, cost_matrix,
                            simulate_cascade, simulate_fuzzy, simulate_single)
    df = D.load_processed(a.dataset)
    y, V = df["label"].values, D.votes(df)
    models = available_models(a.dataset)
    models = [m for m in models if not m.endswith("_LEAKY")]
    rows, fz_rows, cost_rows = [], [], []
    for s in _seeds(a):
        sp = D.load_split(a.dataset, s)
        te, yv, yt = sp["test"], y[sp["val"]], y[sp["test"]]
        cal = {}
        for m in models:
            try:
                vp, tp, _, _ = _calibrated(a.dataset, m, s, yv)
                cal[m] = (vp, tp)
            except FileNotFoundError:
                pass
        # (1) single model + human
        for m, (_, tp) in cal.items():
            for b in BUDGETS:
                for oracle in (False, True):
                    r = simulate_single(yt, tp, V[te], b, seed=s, oracle=oracle)
                    rows.append({"system": "single", "model": m, "seed": s, "budget": b,
                                 "human": "oracle" if oracle else "annotator", **r})
        # (2) cascade cheap -> strong + human
        if a.cheap in cal and a.strong in cal:
            th = choose_cascade_threshold(cal[a.cheap][0], cal[a.strong][0], yv)
            for b in BUDGETS:
                r = simulate_cascade(yt, cal[a.cheap][1], cal[a.strong][1], V[te], th, b, seed=s)
                rows.append({"system": "cascade", "model": f"{a.cheap}->{a.strong}", "seed": s,
                             "budget": b, "human": "annotator", "cascade_theta": th, **r})
        # (3) fuzzy resolver (original multi-stage design, now 3-class)
        if "vader_neg" in df.columns:
            neg = df["vader_neg"].values
            nmf = neg_mf_from_train(neg[sp["train"]])
            for m in [x for x in ["mnb", "lr", "lr_bal", "ga_best"] if x in cal]:
                vp, tp = cal[m]
                fz_val, _, _ = fuzzy_resolve(vp, neg[sp["val"]], nmf)
                fz_te, _, strengths = fuzzy_resolve(tp, neg[te], nmf)
                # theta chosen on validation, plus the original fixed 0.6
                best_th = max(C.THETA_GRID, key=lambda th: simulate_fuzzy(yv, vp, fz_val, th)["f1_macro"])
                for th, how in [(0.6, "fixed"), (best_th, "val-tuned")]:
                    r = simulate_fuzzy(yt, tp, fz_te, th)
                    rbc = r.pop("routed_by_class")
                    fz_rows.append({"model": m, "seed": s, "theta": th, "theta_choice": how,
                                    "f1_macro_no_fuzzy": metrics(yt, tp.argmax(1))["f1_macro"],
                                    **{f"routed_{k}": v for k, v in rbc.items()}, **r,
                                    "mean_rule_strength": strengths.mean(0).round(4).tolist()})
        # (4) cost sweep: best review budget for each missed-hate cost
        for m, (_, tp) in cal.items():
            for c_miss in [1, 2, 5, 10, 20]:
                cm = cost_matrix(c_miss=c_miss)
                best = min(BUDGETS, key=lambda b: simulate_single(yt, tp, V[te], b, seed=s, cm=cm)["cost_per_item"])
                cost_rows.append({"model": m, "seed": s, "c_miss": c_miss, "best_budget": best,
                                  "cost_at_best": simulate_single(yt, tp, V[te], best, seed=s, cm=cm)["cost_per_item"],
                                  "cost_no_human": simulate_single(yt, tp, V[te], 0.0, seed=s, cm=cm)["cost_per_item"]})
    TABLES.mkdir(parents=True, exist_ok=True)
    tri = pd.DataFrame(rows)
    tri.to_csv(TABLES / f"{a.dataset}_triage.csv", index=False)
    pd.DataFrame(fz_rows).to_csv(TABLES / f"{a.dataset}_fuzzy.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(TABLES / f"{a.dataset}_cost_sweep.csv", index=False)
    ceiling = annotator_agreement_ceiling(y, V)
    (TABLES / f"{a.dataset}_annotator_ceiling.json").write_text(json.dumps({"single_annotator_macro_f1": ceiling}))
    _plot_triage(tri, a.dataset)
    print("single-annotator macro-F1 vs majority:", round(ceiling, 4))


def _plot_triage(tri, dataset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    d = tri[tri.human == "annotator"].groupby(["system", "model", "budget"]).mean(numeric_only=True).reset_index()
    for metric, ylabel in [("missed_hate_rate", "Missed hate rate"), ("f1_macro", "Macro-F1"),
                           ("cost_per_item", "Expected cost per post")]:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for (sys_, m), g in d.groupby(["system", "model"]):
            ax.plot(100 * g.budget, g[metric], marker="o", ms=3, lw=1, label=m)
        ax.set_xlabel("Posts sent to human review (%)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / f"{dataset}_triage_{metric}.pdf")
        plt.close(fig)


# ---------------------------------------------------------------- fairness
def cmd_fairness(a):
    from src.fairness import assign_dialect, dialect_table, target_table
    from src.triage import defer_by_budget, human_labels
    df = D.load_processed(a.dataset)
    y, V = df["label"].values, D.votes(df)
    models = [m for m in available_models(a.dataset) if not m.endswith("_LEAKY")]
    out = []
    groups = None
    post_path = C.DATA_DIR / a.dataset / "dialect_posteriors.npy"
    if post_path.exists():
        post = np.load(post_path)
        groups = assign_dialect(np.nan_to_num(post, nan=0.0))
        print("dialect groups:", pd.Series(groups).value_counts().to_dict())
    for s in _seeds(a):
        sp = D.load_split(a.dataset, s)
        te = sp["test"]
        rng = np.random.default_rng(s)
        for m in models:
            try:
                _, tp, _, _ = _calibrated(a.dataset, m, s, y[sp["val"]])
            except FileNotFoundError:
                continue
            for b in [0.0, 0.10]:
                deferred = defer_by_budget(tp, b)
                pred = tp.argmax(1).copy()
                pred[deferred] = human_labels(V[te][deferred], rng)
                if groups is not None:
                    t = dialect_table(y[te], pred, groups[te], deferred)
                    t.insert(0, "budget", b); t.insert(0, "seed", s); t.insert(0, "model", m)
                    out.append(("dialect", t))
                if df["targets"].str.len().gt(0).any():
                    t = target_table(y[te], pred, df["targets"].iloc[te].reset_index(drop=True), a.min_hate)
                    t.insert(0, "budget", b); t.insert(0, "seed", s); t.insert(0, "model", m)
                    out.append(("target", t))
    TABLES.mkdir(parents=True, exist_ok=True)
    for kind in ["dialect", "target"]:
        frames = [t for k, t in out if k == kind]
        if frames:
            pd.concat(frames).to_csv(TABLES / f"{a.dataset}_fairness_{kind}.csv", index=False)
            print(f"wrote {a.dataset}_fairness_{kind}.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--dataset", default="davidson")
    p.add_argument("--seeds", default=None, help="comma-separated, default all")
    p.add_argument("--davidson-csv", default=None)
    p.add_argument("--hatexplain-json", default=None)
    p.add_argument("--base", default="lr_bal")
    p.add_argument("--leaky", action="store_true")
    p.add_argument("--model", default="hatebert")
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--llm", default=C.LLM_MODEL)
    p.add_argument("--llm-4bit", action="store_true", help="load the LLM in 4-bit (needed for 7B on a T4)")
    p.add_argument("--force", action="store_true", help="re-run even if results already exist")
    p.add_argument("--twitteraae", default=None)
    p.add_argument("--cheap", default="lr_bal")
    p.add_argument("--strong", default="hatebert")
    p.add_argument("--min-hate", type=int, default=30)
    a = p.parse_args()
    globals()[f"cmd_{a.command}"](a)


if __name__ == "__main__":
    main()
