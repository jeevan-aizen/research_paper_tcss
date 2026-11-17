# Who Handles the Uncertain Cases? Confidence-Based Human–AI Triage for Hate Speech Moderation

Code, results and LaTeX source for the paper by **K. Jeevan, V. S. A. Paul and A. N. Rao**
(Geethanjali College of Engineering and Technology, Hyderabad), submitted to
*IEEE Transactions on Computational Social Systems*.

The study evaluates hate speech moderation as a human–AI pipeline: classifiers decide the
posts they are confident about, and the least-confident posts are deferred to a simulated
human moderator drawn from the datasets' real annotator votes. It measures missed hate,
review workload, expected cost, GPU usage, and how errors and review are distributed across
dialects (Davidson) and targeted communities (HateXplain).

## Repository layout

```
run.py                 single entry point for every experiment (see below)
src/                   data loading, models, BPSO, GA, fuzzy resolver, calibration,
                       triage simulation, fairness analysis, statistics
tests/smoke_test.py    end-to-end test on synthetic data (CPU, ~1 min)
colab_runner.ipynb     step-by-step notebook used to run everything on Google Colab
paper/                 LaTeX source (author photos not included); make_tables.py builds
                       every table and figure
results/               result tables (CSV), figures, and per-run metadata (timings, memory)
data/*/description.json  dataset statistics (raw data is downloaded, not stored here)
```

## Setup

```bash
pip install -r requirements.txt
python tests/smoke_test.py          # should end with: SMOKE TEST PASSED
```

## Data

| Dataset | Source |
|---|---|
| Davidson et al. (2017), 24,783 tweets | `labeled_data.csv` from https://github.com/t-davidson/hate-speech-and-offensive-language |
| HateXplain (Mathew et al., 2021) | `Data/dataset.json` from https://github.com/hate-alert/HateXplain |
| TwitterAAE dialect model (Blodgett et al., 2016) | https://github.com/slanglab/twitteraae |

The TwitterAAE code is Python 2. Before use, patch it for Python 3 and make its model folder
reachable (this is what the Colab notebook does):

```bash
git clone https://github.com/slanglab/twitteraae raw/twitteraae
ln -sfn "$PWD/raw/twitteraae/model" "$PWD/model"
sed -i 's/\.decode("utf-8")//g; s/\bxrange(/range(/g; s/\.iteritems()/.items()/g; s/\bunicode\b/str/g; s/\bbasestring\b/str/g' raw/twitteraae/code/predict.py
```

## Running the experiments

```bash
python run.py prepare --dataset davidson   --davidson-csv raw/labeled_data.csv
python run.py prepare --dataset hatexplain --hatexplain-json raw/hatexplain.json

for D in davidson hatexplain; do
  python run.py classical   --dataset $D
  python run.py ga          --dataset $D
  python run.py bpso        --dataset $D --base lr_bal
  python run.py bpso        --dataset $D --base mnb
  python run.py bpso        --dataset $D --base lr_bal --leaky   # leakage audit only
  python run.py randsub     --dataset $D --base lr_bal           # random-subset control
  python run.py randsub     --dataset $D --base mnb
  python run.py transformer --dataset $D --model hatebert        # GPU
  python run.py transformer --dataset $D --model roberta
  python run.py transformer --dataset $D --model bertweet
  python run.py transformer --dataset $D --model hatebert --weighted
  python run.py llm         --dataset $D --llm Qwen/Qwen2.5-3B-Instruct
done
python run.py dialect --dataset davidson --twitteraae raw/twitteraae
for D in davidson hatexplain; do
  python run.py evaluate --dataset $D
  python run.py triage   --dataset $D
  python run.py fairness --dataset $D
done
```

Every step resumes: finished (dataset, model, split) runs are skipped, so an interrupted
session can simply be restarted. Set `CH_DATA_DIR` and `CH_RESULTS_DIR` to keep outputs on
persistent storage (e.g. Google Drive). The dialect robustness check re-runs `fairness`
with `DIALECT_THRESHOLD` in `src/config.py` set to 0.7, 0.6 and 0.5.

## Reproducing the paper's tables and figure

```bash
python paper/make_tables.py .
```

reads `results/tables/*.csv`, `results/*/*/seed*/meta.json` and `data/*/description.json`
and writes `paper/tables/*.tex` and `paper/figures/triage.pdf`. No number in the paper's
tables is typed by hand.

## Protocol in brief

* Five stratified 70/10/20 splits (seeds 42, 123, 456, 789, 1011). The validation split is
  the only data used for feature-selection fitness, temperature scaling, and the cascade and
  fuzzy thresholds; each test split is used once.
* Metrics use `zero_division=0`. Models are compared with the corrected resampled t-test
  (Nadeau & Bengio, 2003), Holm-adjusted within each family of comparisons.
* Compute: Google Colab, 2-core CPU runtime and one NVIDIA T4 GPU; Python 3.13,
  scikit-learn 1.6.1, PyTorch 2.11.0, Transformers 5.18.0 (fine-tuning) / 5.22.0 (LLM).

## Ethics

Only publicly released, de-identified research datasets are used; no new data is collected
and the raw posts are not redistributed here. Dialect estimates describe language variety
and are used only in aggregate.
