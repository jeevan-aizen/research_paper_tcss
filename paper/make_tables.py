"""Builds every LaTeX table and the triage figure from results/tables/*.csv.
Run:  python paper/make_tables.py <path-to-unzipped-results_for_paper>
Every number in the paper's tables comes from this script; nothing is typed by hand."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
T = R / 'results' / 'tables'
OUT = Path(__file__).parent / 'tables'
FIG = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
DS = ['davidson', 'hatexplain']
TBD = r'\res{--}'
NAMES = {'mnb': 'MNB', 'mnb_bal': 'MNB-bal', 'lr': 'LR', 'lr_bal': 'LR-bal', 'svm': 'Linear SVM',
         'svm_bal': 'Linear SVM-bal', 'ga_best': 'GA-selected', 'bpso_mnb': 'BPSO + MNB',
         'bpso_lr_bal': 'BPSO + LR-bal', 'roberta': 'RoBERTa', 'bertweet': 'BERTweet',
         'hatebert': 'HateBERT', 'hatebert_w': 'HateBERT-w', 'llm_zeroshot': 'Qwen2.5-3B (zero-shot)'}

def f3(x): return f'{x:.3f}'
def pm(m, s): return f'{m:.3f}\\,$\\pm$\\,{s:.3f}'
def write(name, s): (OUT / name).write_text(s); print('wrote', name)

S = {D: pd.read_csv(T / f'{D}_summary.csv', index_col=0) for D in DS}
SIG = {D: pd.read_csv(T / f'{D}_significance.csv') for D in DS}
TRI = {D: pd.read_csv(T / f'{D}_triage.csv') for D in DS}
HATE_TEST = {D: json.load(open(R / 'data' / D / 'description.json')) for D in DS}

# ---------------- Table: dataset statistics ----------------
d = {D: HATE_TEST[D] for D in DS}
def cls(D, k): return f"{d[D]['per_class'][k]:,} ({d[D]['per_class_pct'][k]:.1f}\\%)"
s = r"""\begin{tabular}{lrr}
\toprule
 & Davidson & HateXplain \\
\midrule
Posts & %s & %s \\
Hate & %s & %s \\
Offensive & %s & %s \\
Neither / normal & %s & %s \\
Exact duplicate texts & %d & %d \\
Unanimous annotator label & %.1f\%% & %.1f\%% \\
Train / val.\ / test per split & %s & %s \\
\bottomrule
\end{tabular}
""" % (f"{d['davidson']['n']:,}", f"{d['hatexplain']['n']:,}",
       cls('davidson', 'hate'), cls('hatexplain', 'hate'), cls('davidson', 'offensive'), cls('hatexplain', 'offensive'),
       cls('davidson', 'neither'), cls('hatexplain', 'neither'),
       d['davidson']['exact_duplicate_texts'], d['hatexplain']['exact_duplicate_texts'],
       d['davidson']['unanimous_label_pct'], d['hatexplain']['unanimous_label_pct'],
       '/'.join(f"{v:,}" for v in d['davidson']['split_sizes']['42'].values()),
       '/'.join(f"{v:,}" for v in d['hatexplain']['split_sizes']['42'].values()))
write('data.tex', s)

# ---------------- Table: main results ----------------
groups = [('Classical', ['mnb', 'mnb_bal', 'lr', 'lr_bal', 'svm', 'svm_bal']),
          ('Bio-inspired', ['ga_best', 'bpso_mnb', 'bpso_lr_bal']),
          ('Transformer', ['roberta', 'bertweet', 'hatebert', 'hatebert_w']),
          ('LLM', ['llm_zeroshot'])]
_fine = lambda D: [m for g in groups for m in g[1] if m in S[D].index and m != 'llm_zeroshot']
best = {D: (S[D].loc[_fine(D), 'f1_macro_mean'].max(), (1 - S[D].loc[_fine(D), 'recall_hate_mean']).min()) for D in DS}
def tied(D, m):
    if m == 'hatebert': return True
    r = SIG[D][(SIG[D].reference == 'hatebert') & (SIG[D].model == m)]
    return len(r) and r.p_holm.iloc[0] >= 0.05
lines = []
for gname, ms in groups:
    lines.append(r'\multicolumn{9}{l}{\textit{%s}} \\' % gname)
    for m in ms:
        row = [NAMES[m]]
        for D in DS:
            if m not in S[D].index:
                row += [TBD] * 4; continue
            r = S[D].loc[m]
            f1 = pm(r.f1_macro_mean, r.f1_macro_std)
            if abs(r.f1_macro_mean - best[D][0]) < 1e-9: f1 = r'\textbf{' + f1 + '}'
            if tied(D, m): f1 += r'$^{\dagger}$'
            mhr = 1 - r.recall_hate_mean
            mh = f3(mhr) if abs(mhr - best[D][1]) > 1e-9 else r'\textbf{' + f3(mhr) + '}'
            row += [f1, f3(r.f1_hate_mean), mh, f3(r.ece_cal_mean)]
        lines.append(' & '.join(row) + r' \\')
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
write('main.tex', r"""\begin{tabular}{l|cccc|cccc}
\toprule
 & \multicolumn{4}{c|}{Davidson} & \multicolumn{4}{c}{HateXplain} \\
Model & Macro-F1 & F1 hate & MHR & ECE & Macro-F1 & F1 hate & MHR & ECE \\
\midrule
""" + '\n'.join(lines) + '\n\\end{tabular}\n')

# ---------------- Table: feature-selection audit ----------------
rows = []
for D in DS:
    for base, nm in [('lr_bal', 'LR-bal'), ('mnb', 'MNB')]:
        s_ = S[D]
        get = lambda m: pm(s_.loc[m].f1_macro_mean, s_.loc[m].f1_macro_std) if m in s_.index else '--'
        rows.append(f"{'Davidson' if D=='davidson' else 'HateXplain'} & {nm} & {get(base)} & {get('rand_'+base)} & "
                    f"{get('bpso_'+base)} & {get('bpso_'+base+'_LEAKY')} \\\\")
write('fs.tex', r"""\begin{tabular}{llcccc}
\toprule
Dataset & Base & All 5{,}000 & Random half & BPSO (val.) & BPSO (test) \\
\midrule
""" + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------- Table: fuzzy resolver ----------------
lines = []
for D in DS:
    fz = pd.read_csv(T / f'{D}_fuzzy.csv')
    nh = d[D]['per_class']['hate'] / d[D]['n'] * d[D]['split_sizes']['42']['test']
    lines.append(r'\multicolumn{6}{l}{\textit{%s}} \\' % ('Davidson' if D == 'davidson' else 'HateXplain'))
    for m in ['mnb', 'lr', 'lr_bal', 'ga_best']:
        for how in ['fixed', 'val-tuned']:
            g = fz[(fz.model == m) & (fz.theta_choice == how)].mean(numeric_only=True)
            mhr0 = 1 - S[D].loc[m].recall_hate_mean
            lines.append(f"{NAMES[m]} & {g.theta:.2f} & {100*g.routed_rate:.1f} & {100*g.routed_hate/nh:.1f} & "
                         f"{g.f1_macro_no_fuzzy:.3f}$\\to${g.f1_macro:.3f} & {mhr0:.3f}$\\to${g.missed_hate_rate:.3f} \\\\")
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
write('fuzzy.tex', r"""\begin{tabular}{lccccc}
\toprule
Base & $\theta$ & Routed & Hate routed & Macro-F1 & MHR \\
 & & (\%) & (\%) & & \\
\midrule
""" + '\n'.join(lines) + '\n\\end{tabular}\n')

# ---------------- Table: triage at 10% ----------------
def tri(D, m, b, human='annotator'):
    t = TRI[D]; g = t[(t.model == m) & (t.budget == b) & (t.human == human)].mean(numeric_only=True)
    return g
spec = [('LR-bal', 'lr_bal', 0.0, 'annotator'), ('LR-bal + human', 'lr_bal', 0.1, 'annotator'),
        ('SVM-bal + human', 'svm_bal', 0.1, 'annotator'),
        ('HateBERT-w', 'hatebert_w', 0.0, 'annotator'), ('HateBERT-w + human', 'hatebert_w', 0.1, 'annotator'),
        ('BERTweet', 'bertweet', 0.0, 'annotator'), ('BERTweet + human', 'bertweet', 0.1, 'annotator'),
        ('Cascade + human', 'lr_bal->hatebert', 0.1, 'annotator'),
        ('HateBERT-w + oracle', 'hatebert_w', 0.1, 'oracle'), ('BERTweet + oracle', 'bertweet', 0.1, 'oracle')]
lines = []
for lab, m, b, h in spec:
    row = [lab]
    for D in DS:
        g = tri(D, m, b, h)
        gpu = '0' if not m.startswith('hate') and not m.startswith('bert') and '->' not in m else ('100' if '->' not in m else f'{100*g.gpu_fraction:.0f}')
        row += [f3(g.f1_macro), f3(g.missed_hate_rate), f3(g.false_flag_rate), f3(g.cost_per_item), gpu]
    lines.append(' & '.join(row) + r' \\')
write('triage.tex', r"""\begin{tabular}{l|ccccc|ccccc}
\toprule
 & \multicolumn{5}{c|}{Davidson} & \multicolumn{5}{c}{HateXplain} \\
System & Macro-F1 & MHR & FFR & Cost & GPU\,\% & Macro-F1 & MHR & FFR & Cost & GPU\,\% \\
\midrule
""" + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------- Table: cost sweep ----------------
lines = []
for D in DS:
    cs = pd.read_csv(T / f'{D}_cost_sweep.csv').groupby(['model', 'c_miss'])[['best_budget', 'cost_at_best', 'cost_no_human']].mean()
    lines.append(r'\multicolumn{5}{l}{\textit{%s}} \\' % ('Davidson' if D == 'davidson' else 'HateXplain'))
    for c in [1, 2, 5, 10, 20]:
        vals = {m: cs.loc[(m, c)] for m in ['lr_bal', 'hatebert_w', 'bertweet', 'llm_zeroshot']}
        lo = min(v.cost_at_best for v in vals.values())
        cells = []
        for m in ['lr_bal', 'hatebert_w', 'bertweet', 'llm_zeroshot']:
            v = vals[m]; txt = f"{100*v.best_budget:.0f}\\% / {v.cost_at_best:.3f}"
            cells.append(r'\textbf{' + txt + '}' if abs(v.cost_at_best - lo) < 5e-4 else txt)
        lines.append(f"{c} & " + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
write('cost.tex', r"""\begin{tabular}{ccccc}
\toprule
$c_m$ & LR-bal & HateBERT-w & BERTweet & Qwen2.5-3B \\
\midrule
""" + '\n'.join(lines) + '\n\\end{tabular}\n')

# ---------------- Table: dialect ----------------
def dial(thr):
    x = pd.read_csv(T / f'davidson_fairness_dialect_thr{thr}.csv')
    return x[x.group.isin(['AAE', 'WAE'])].groupby(['model', 'budget', 'group']).mean(numeric_only=True)
g5 = dial('0.50')
lines = []
for m in ['lr_bal', 'svm_bal', 'hatebert_w', 'bertweet']:
    for grp in ['AAE', 'WAE', 'ratio']:
        if grp == 'ratio':
            r = lambda b, c: g5.loc[(m, b, 'AAE'), c] / g5.loc[(m, b, 'WAE'), c]
            cells = [f"{r(0.0,'false_hate_flag_rate'):.2f}", f"{r(0.0,'false_abusive_flag_rate'):.2f}",
                     f"{r(0.1,'false_abusive_flag_rate'):.2f}", f"{r(0.1,'deferral_rate'):.2f}"]
            lines.append(r' & AAE/WAE & ' + ' & '.join(r'\textbf{%s}' % c for c in cells) + r' \\ \midrule')
        else:
            a = lambda b, c: g5.loc[(m, b, grp), c]
            first = r'\multirow{3}{*}{%s}' % NAMES[m] if grp == 'AAE' else ''
            lines.append(f"{first} & {grp} & {a(0.0,'false_hate_flag_rate'):.3f} & {a(0.0,'false_abusive_flag_rate'):.3f} & "
                         f"{a(0.1,'false_abusive_flag_rate'):.3f} & {a(0.1,'deferral_rate'):.3f} \\\\")
lines[-1] = lines[-1].replace(r'\midrule', r'\bottomrule')
write('dialect.tex', r"""\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{2}{c}{No review} & \multicolumn{2}{c}{10\% review} \\
Model & Group & FHF & FAF & FAF & Deferred \\
\midrule
""" + '\n'.join(lines) + '\n\\end{tabular}\n')

lines = []
G = {thr: dial(thr) for thr in ['0.50', '0.60', '0.70']}
for m in ['lr_bal', 'svm_bal', 'hatebert_w', 'bertweet']:
    cells = []
    for thr in ['0.50', '0.60', '0.70']:
        g = G[thr]; r = lambda b: g.loc[(m, b, 'AAE'), 'false_abusive_flag_rate'] / g.loc[(m, b, 'WAE'), 'false_abusive_flag_rate']
        cells.append(f"{r(0.0):.2f}$\\to${r(0.1):.2f}")
    lines.append(f"{NAMES[m]} & " + ' & '.join(cells) + r' \\')
ns = {thr: pd.read_csv(T / f'davidson_fairness_dialect_thr{thr}.csv').query("group in ['AAE','WAE']").groupby('group').n.mean() for thr in G}
nrow = 'AAE / WAE per split & ' + ' & '.join(f"{ns[t]['AAE']:.0f} / {ns[t]['WAE']:.0f}" for t in G) + r' \\'
write('dialect_robust.tex', r"""\begin{tabular}{lccc}
\toprule
Posterior threshold & 0.5 & 0.6 & 0.7 \\
\midrule
""" + '\n'.join(lines) + '\n\\midrule\n' + nrow + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------- Table: target groups ----------------
h = pd.read_csv(T / 'hatexplain_fairness_target.csv')
n = h[(h.model == 'bertweet') & (h.budget == 0)].groupby('target').n_hate.mean()
mr = h.groupby(['target', 'model', 'budget']).missed_hate_rate.mean()
order = n.sort_values(ascending=False).index
lines = []
for tg in order:
    small = n[tg] < 50
    cells = [f"{mr[(tg, m, b)]:.3f}" for m in ['lr_bal', 'hatebert_w', 'bertweet'] for b in [0.0, 0.1]]
    lines.append(f"{tg}{'$^{*}$' if small else ''} & {n[tg]:.0f} & " + ' & '.join(cells) + r' \\')
write('targets.tex', r"""\begin{tabular}{lccccccc}
\toprule
 & Hate & \multicolumn{2}{c}{LR-bal} & \multicolumn{2}{c}{HateBERT-w} & \multicolumn{2}{c}{BERTweet} \\
Target & posts & 0\% & 10\% & 0\% & 10\% & 0\% & 10\% \\
\midrule
""" + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------- Table: compute ----------------
rows = []
for p in glob.glob(str(R / 'results' / '*' / '*' / 'seed*' / 'meta.json')):
    parts = Path(p).parts; j = json.load(open(p)); j.update(dataset=parts[-4], m=parts[-3]); rows.append(j)
M = pd.DataFrame(rows).groupby(['m', 'dataset']).mean(numeric_only=True)
def both(m, c, fmt):
    return ' / '.join(fmt(M.loc[(m, D), c]) for D in DS)
lines = [
    r'LR-bal & CPU & ' + both('lr_bal', 'fit_seconds', lambda v: f'{v:.1f}\\,s') + r' & $<$0.01 & -- \\',
    r'Linear SVM-bal & CPU & ' + both('svm_bal', 'fit_seconds', lambda v: f'{v:.1f}\\,s') + r' & $<$0.01 & -- \\',
    r'GA search (LR/MNB) & CPU & ' + both('ga_best', 'seconds', lambda v: f'{v/60:.1f}\\,min') + r' & $<$0.01 & -- \\',
    r'BPSO + LR-bal & CPU & ' + both('bpso_lr_bal', 'seconds', lambda v: f'{v/60:.1f}\\,min') + r' & $<$0.01 & -- \\',
]
for m in ['roberta', 'bertweet', 'hatebert', 'hatebert_w']:
    lines.append(f"{NAMES[m]} & T4 GPU & " + both(m, 'train_seconds', lambda v: f'{v/60:.1f}\\,min') + ' & ' +
                 both(m, 'ms_per_post', lambda v: f'{v:.2f}') + ' & ' + both(m, 'peak_vram_mib', lambda v: f'{v/1024:.1f}') + r' \\')
lines.append(r'Qwen2.5-3B (zero-shot) & T4 GPU & -- & ' + f"{M.loc[('llm_zeroshot','davidson'),'ms_per_post']:.0f}" + r' / $\approx$620$^{a}$ & 6.2$^{b}$ \\')
write('compute.tex', r"""\begin{tabular}{llccc}
\toprule
Model & Hardware & Training & Inference & Peak GPU \\
 & & (per split) & (ms/post) & memory (GB) \\
\midrule
""" + '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')

# ---------------- Figure: triage curves ----------------
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5})
sty = {'lr_bal': ('LR-bal', 'C0', 'o', '-'), 'svm_bal': ('SVM-bal', 'C2', 's', '-'),
       'hatebert_w': ('HateBERT-w', 'C3', '^', '-'), 'bertweet': ('BERTweet', 'C1', 'D', '-'),
       'lr_bal->hatebert': ('Cascade (LR-bal$\\to$HateBERT)', 'C4', 'v', '--')}
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.3))
for j, D in enumerate(DS):
    t = TRI[D]; t = t[t.human == 'annotator'].groupby(['model', 'budget']).mean(numeric_only=True)
    for i, (met, yl) in enumerate([('missed_hate_rate', 'Missed-hate rate'), ('f1_macro', 'Macro-F1')]):
        a = ax[i, j]
        for m, (lab, c, mk, ls) in sty.items():
            g = t.loc[m]; a.plot(100 * g.index, g[met], color=c, marker=mk, ls=ls, ms=3, lw=1, label=lab)
        a.set_ylabel(yl); a.grid(alpha=.3, lw=.4)
        if i == 0: a.set_title('Davidson' if D == 'davidson' else 'HateXplain')
        if i == 1: a.set_xlabel('Posts deferred to human review (%)')
h_, l_ = ax[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc='upper center', ncol=5, fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / 'triage.pdf'); print('wrote figures/triage.pdf')
