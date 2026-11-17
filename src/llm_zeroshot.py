"""Zero-shot instruction-tuned LLM baseline. Instead of parsing free text, we
read the next-token probabilities of the three option letters, which gives a
proper probability vector usable for calibration and triage.

The prompt does not depend on the split, so every post is scored ONCE and the
per-seed val/test predictions are sliced from that cache. Scores are written in
chunks so a Colab disconnect loses at most one chunk."""
import time
from pathlib import Path

import numpy as np

from . import config as C

PROMPT = """You are a content moderator. Classify the social media post into exactly one category.

A) Hate speech: language that expresses hatred towards a targeted group, or is intended to be derogatory, to humiliate, or to insult the members of the group based on attributes such as race, religion, ethnicity, sexual orientation, disability, or gender.
B) Offensive language: offensive, vulgar or insulting language that is not hate speech as defined above.
C) Neither: not offensive and not hate speech.

Post: \"\"\"{text}\"\"\"

Answer with a single letter (A, B, or C)."""

LETTER_TO_LABEL = {"A": C.HATE, "B": C.OFFENSIVE, "C": C.NEITHER}


def load_llm(model_name=C.LLM_MODEL, four_bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # T4 / P100 (free Colab and Kaggle GPUs) lack bfloat16; fall back to float16.
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else torch.float16
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"   # renamed in newer versions
    kw = {dkey: dtype, "device_map": "auto", "low_cpu_mem_usage": True}
    if four_bit:   # a 7B model in fp16 (~15 GB) does not fit a 15 GB T4
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
    letter_ids = []
    for L in "ABC":
        ids = tok.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"letter {L} is not a single token for {model_name}"
        letter_ids.append(ids[0])
    return tok, model, letter_ids


def score(texts, bundle, batch_size=16, max_chars=1000) -> np.ndarray:
    import torch
    tok, model, letter_ids = bundle
    out = np.zeros((len(texts), 3))
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=str(t)[:max_chars])}],
            tokenize=False, add_generation_prompt=True) for t in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :]
        p = torch.softmax(logits[:, letter_ids].float(), dim=-1).cpu().numpy()
        for j, L in enumerate("ABC"):
            out[s:s + len(batch), LETTER_TO_LABEL[L]] = p[:, j]
    return out


def score_all_cached(texts, cache_dir: Path, model_name=C.LLM_MODEL, four_bit=False,
                     chunk=C.LLM_CHUNK, log=print):
    """Score every post, resuming from cached chunks. Returns (probs, seconds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(texts) + chunk - 1) // chunk
    todo = [i for i in range(n_chunks) if not (cache_dir / f"chunk{i:04d}.npy").exists()]
    seconds = 0.0
    if todo:
        bundle = load_llm(model_name, four_bit)
        for i in todo:
            t0 = time.time()
            np.save(cache_dir / f"chunk{i:04d}.npy", score(texts[i * chunk:(i + 1) * chunk], bundle))
            seconds += time.time() - t0
            log(f"  LLM chunk {i + 1}/{n_chunks} done ({time.time() - t0:.0f}s)")
    probs = np.concatenate([np.load(cache_dir / f"chunk{i:04d}.npy") for i in range(n_chunks)])
    assert len(probs) == len(texts)
    return probs, seconds
