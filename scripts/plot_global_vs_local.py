import json, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = {'own': '#2a78d6', 'glob': '#eb6834', 'central': '#1baf7a'}
LAB = {'own': "Hospital's own model", 'glob': 'Global FedAvg model', 'central': 'Pooled (not allowed in practice)'}
def load(f):
    d = json.load(open(f)); R = d['results']; hs = [h for h in ['hospital_a','hospital_b','hospital_c','hospital_d','ALL']]
    out = {}
    for h in hs:
        rs = [r for r in R if r['hospital'] == h]
        out[h] = {k: np.mean([r[k] for r in rs]) for k in ('own','glob','central') if k in rs[0]}
        out[h]['n'] = rs[0]['train']
    return d, out
def fig(panels, title, path):
    f, axes = plt.subplots(1, len(panels), figsize=(4.2*len(panels), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, fn) in zip(axes, panels):
        d, o = load(fn); hs = list(o); x = np.arange(len(hs)); w = 0.26
        for i, k in enumerate(['own','glob','central']):
            v = [o[h].get(k, np.nan) for h in hs]
            ax.bar(x + (i-1)*w, v, w*0.92, color=C[k], label=LAB[k], zorder=3)
        for j, h in enumerate(hs):
            dv = o[h]['glob'] - o[h]['own']
            ax.text(j, 1.02, f"{dv:+.2f}", ha='center', fontsize=8.5, color='#333')
        ax.set_xticks(x); ax.set_xticklabels([("All" if h=='ALL' else 'Hosp '+h[-1].upper())+f"\n{o[h]['n']} train" for h in hs], fontsize=8)
        ax.set_title(name, fontsize=11, pad=16); ax.set_ylim(0, 1.08)
        ax.grid(axis='y', color='#e6e6e6', zorder=0); ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Macro F1 on that hospital\'s own held-out patients')
    h, l = axes[0].get_legend_handles_labels(); f.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=9)
    f.suptitle(title + "   (numbers on top = global minus own)", fontsize=11.5)
    f.tight_layout(rect=(0, 0.07, 1, 0.97)); f.savefig(path, dpi=170); print('wrote', path)
W = O = 'docs/results/'
full = [('ECG (simulated)', W+'ecg.json'), ('EHR (simulated)', W+'ehr.json'), ('Diabetes (real PIMA)', W+'diabetes.json')]
import os
if os.path.exists(W+'xray.json'): full.insert(2, ('X-ray (simulated, 1 seed)', W+'xray.json'))
fig(full, 'Global model vs each hospital\'s own model - normal-sized hospitals', O+'1_global_vs_own_normal.png')
fig([('ECG, 60 records each', W+'ecg_small.json'), ('EHR, 60 records each', W+'ehr_small.json'), ('Diabetes, 40 records each', W+'diabetes_small.json')],
    'Same test when every hospital is small', O+'2_global_vs_own_small.png')
