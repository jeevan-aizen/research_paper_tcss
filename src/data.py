"""Dataset loading, text cleaning and leakage-safe splits.

Every dataset is converted to one common DataFrame:
    text_raw     original text (used for dialect estimation)
    text_clf     cleaned text for TF-IDF models
    text_tf      lightly normalised text for transformers / LLM
    label        0 = hate, 1 = offensive, 2 = neither
    votes_hate, votes_off, votes_neither   annotator vote counts
    targets      ';'-joined target groups (HateXplain only, else '')
"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
RT_RE = re.compile(r"^\s*rt\b[:\s]*", re.IGNORECASE)
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def clean_for_classical(t: str) -> str:
    t = html.unescape(str(t))
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = HASHTAG_RE.sub(r" \1 ", t)          # keep hashtag word, drop '#'
    t = RT_RE.sub(" ", t)
    t = t.lower()
    t = NON_ALPHA_RE.sub(" ", t)
    return SPACE_RE.sub(" ", t).strip()


def clean_for_transformer(t: str) -> str:
    t = html.unescape(str(t))
    t = URL_RE.sub("HTTPURL", t)
    t = MENTION_RE.sub("@USER", t)
    t = RT_RE.sub("", t)
    return SPACE_RE.sub(" ", t).strip()


# --------------------------------------------------------------------------
# Davidson et al. (2017)
# --------------------------------------------------------------------------
def load_davidson(path: str | None = None) -> pd.DataFrame:
    """Load the FULL Davidson corpus (24,783 tweets).

    `path` may point to labeled_data.csv from github.com/t-davidson/
    hate-speech-and-offensive-language. If None, the HuggingFace copy
    `tdavidson/hate_speech_offensive` is used."""
    if path:
        df = pd.read_csv(path)
    else:
        from datasets import load_dataset
        df = load_dataset("tdavidson/hate_speech_offensive", split="train").to_pandas()

    rename = {"hate_speech": "votes_hate", "hate_speech_count": "votes_hate",
              "offensive_language": "votes_off", "offensive_language_count": "votes_off",
              "neither": "votes_neither", "neither_count": "votes_neither",
              "class": "label", "tweet": "text_raw"}
    df = df.rename(columns=rename)
    need = ["text_raw", "label", "votes_hate", "votes_off", "votes_neither"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Davidson file missing columns {missing}; got {list(df.columns)}")
    df = df[need].copy()
    df["targets"] = ""
    return _finalise(df, "davidson")


# --------------------------------------------------------------------------
# HateXplain (Mathew et al., 2021)
# --------------------------------------------------------------------------
HX_MAP = {"hatespeech": C.HATE, "offensive": C.OFFENSIVE, "normal": C.NEITHER}


def load_hatexplain(dataset_json: str) -> pd.DataFrame:
    """`dataset_json` = Data/dataset.json from github.com/hate-alert/HateXplain.
    Posts without a majority label are dropped (as in the original paper).
    A target group is kept if at least 2 of the 3 annotators named it."""
    raw = json.loads(Path(dataset_json).read_text())
    rows = []
    for pid, post in raw.items():
        labs = [HX_MAP[a["label"]] for a in post["annotators"]]
        counts = np.bincount(labs, minlength=3)
        if counts.max() < 2:
            continue
        tgt_counts: dict[str, int] = {}
        for a in post["annotators"]:
            for t in set(a.get("target", [])):
                if t and t != "None":
                    tgt_counts[t] = tgt_counts.get(t, 0) + 1
        targets = sorted(t for t, n in tgt_counts.items() if n >= 2)
        rows.append(dict(post_id=pid, text_raw=" ".join(post["post_tokens"]),
                         label=int(counts.argmax()), votes_hate=int(counts[0]),
                         votes_off=int(counts[1]), votes_neither=int(counts[2]),
                         targets=";".join(targets)))
    return _finalise(pd.DataFrame(rows), "hatexplain")


def _finalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["text_clf"] = df["text_raw"].map(clean_for_classical)
    df["text_tf"] = df["text_raw"].map(clean_for_transformer)
    df.attrs["name"] = name
    return df


def describe(df: pd.DataFrame) -> dict:
    """Numbers for the dataset table in the paper, including duplicates and
    empty-after-cleaning rows so the paper can state them explicitly."""
    counts = df["label"].value_counts().sort_index()
    return {
        "n": int(len(df)),
        "per_class": {C.LABELS[k]: int(v) for k, v in counts.items()},
        "per_class_pct": {C.LABELS[k]: round(100 * v / len(df), 2) for k, v in counts.items()},
        "exact_duplicate_texts": int(df["text_raw"].duplicated().sum()),
        "empty_after_cleaning": int((df["text_clf"].str.len() == 0).sum()),
        "unanimous_label_pct": round(100 * float(
            (df[["votes_hate", "votes_off", "votes_neither"]].max(1)
             == df[["votes_hate", "votes_off", "votes_neither"]].sum(1)).mean()), 2),
    }


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def make_split(df: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Stratified train / val / test indices. Val is a fraction of the whole
    dataset carved from the non-test part."""
    idx = np.arange(len(df))
    y = df["label"].values
    trval, test = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y, random_state=seed)
    val_frac = C.VAL_SIZE / (1 - C.TEST_SIZE)
    train, val = train_test_split(trval, test_size=val_frac, stratify=y[trval], random_state=seed)
    assert not (set(train) & set(test)) and not (set(val) & set(test)) and not (set(train) & set(val))
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def split_dir(dataset: str, seed: int) -> Path:
    d = C.DATA_DIR / dataset / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_split(dataset: str, seed: int, split: dict) -> None:
    np.savez(split_dir(dataset, seed) / "split.npz", **split)


def load_split(dataset: str, seed: int) -> dict:
    z = np.load(split_dir(dataset, seed) / "split.npz")
    return {k: z[k] for k in z.files}


def load_processed(dataset: str) -> pd.DataFrame:
    return pd.read_pickle(C.DATA_DIR / dataset / "processed.pkl")


def votes(df: pd.DataFrame) -> np.ndarray:
    return df[["votes_hate", "votes_off", "votes_neither"]].to_numpy(dtype=float)
