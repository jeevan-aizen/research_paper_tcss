"""Fine-tuning HateBERT / RoBERTa / BERTweet for 3-class classification.
Model selection uses the validation split only (best epoch by val macro-F1)."""
import time
from pathlib import Path

import numpy as np

from . import config as C


def finetune(model_key, train_texts, y_train, val_texts, y_val, test_texts, seed,
             out_dir, class_weighted=False):
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    set_seed(seed)
    out_dir = Path(out_dir)
    name = C.TRANSFORMERS[model_key]
    tok = None
    if model_key == "bertweet":
        try:   # BERTweet's own normaliser (older transformers); text_tf is already normalised otherwise
            tok = AutoTokenizer.from_pretrained(name, normalization=True, use_fast=False)
        except Exception as e:
            print(f"  (BERTweet slow tokenizer unavailable: {e}; using default tokenizer)")
    if tok is None:
        tok = AutoTokenizer.from_pretrained(name)
    id2label = {i: C.LABELS[i] for i in range(3)}
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=3, id2label=id2label, label2id={v: k for k, v in id2label.items()})

    class DS(Dataset):
        def __init__(self, texts, labels=None):
            self.enc = tok(list(texts), truncation=True, max_length=C.FT["max_len"])
            self.labels = labels

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            if self.labels is not None:
                item["labels"] = int(self.labels[i])
            return item

    weights = None
    if class_weighted:
        counts = np.bincount(y_train, minlength=3)
        weights = torch.tensor(len(y_train) / (3 * counts), dtype=torch.float)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            w = weights.to(out.logits.device) if weights is not None else None
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
            return (loss, out) if return_outputs else loss

    def compute_metrics(ev):
        return {"f1_macro": f1_score(ev.label_ids, ev.predictions.argmax(-1), average="macro")}

    # warmup given in steps: `warmup_ratio` was removed in transformers 5
    steps_per_epoch = -(-len(y_train) // C.FT["batch_size"])
    warmup_steps = int(C.FT["warmup_ratio"] * steps_per_epoch * C.FT["epochs"])
    args = TrainingArguments(
        output_dir=str(out_dir / "ckpt"), num_train_epochs=C.FT["epochs"],
        learning_rate=C.FT["lr"], per_device_train_batch_size=C.FT["batch_size"],
        per_device_eval_batch_size=64, weight_decay=C.FT["weight_decay"],
        warmup_steps=warmup_steps, fp16=C.FT["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", save_total_limit=1, seed=seed,
        report_to="none", logging_steps=100)
    trainer = WTrainer(model=model, args=args, train_dataset=DS(train_texts, y_train),
                       eval_dataset=DS(val_texts, y_val), processing_class=tok,
                       compute_metrics=compute_metrics)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    def probs(texts):
        t1 = time.time()
        logits = trainer.predict(DS(texts)).predictions
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True), time.time() - t1

    val_p, _ = probs(val_texts)
    test_p, test_s = probs(test_texts)
    import shutil
    shutil.rmtree(out_dir / "ckpt", ignore_errors=True)   # checkpoints are ~1 GB each
    meta = {"model": name, "seed": seed, "class_weighted": class_weighted,
            "train_seconds": round(train_s, 1),
            "test_inference_seconds": round(test_s, 2),
            "ms_per_post": round(1000 * test_s / len(test_texts), 3),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    return val_p, test_p, meta
