#!/usr/bin/env python3
"""Draw Fig. 4a/b from tables/gene_expression/ into figures/.

Each model is evaluated on its own official gene panel, so global-r values
are not strictly like-for-like across models; see the paper for the panels.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from _common import ROOT, OUTPUT_DIR

B = str(ROOT)
OUT_A = str(OUTPUT_DIR / 'fig4a_global_r_bars')
OUT_B = str(OUTPUT_DIR / 'fig4b_dlpfc_per_sample')

MODELS = ['OmiCLIP', 'STPath', 'HisToGene', 'BLEEP']  # average-rank order
# 蓝色系:名次越好颜色越深(OmiCLIP > STPath > HisToGene > BLEEP)
MCOL = {'OmiCLIP': '#08519c', 'STPath': '#3182bd', 'HisToGene': '#6baed6',
        'BLEEP': '#c6dbef'}
DS_ORDER = ['DLPFC', 'breast_cancer', 'human_heart',
            'human_lymph_node', 'mouse_brain', 'mouse_kidney']
DS_LAB = {'DLPFC': 'DLPFC', 'breast_cancer': 'Breast\ncancer',
          'human_heart': 'Human\nheart', 'human_lymph_node': 'Lymph\nnode',
          'mouse_brain': 'Mouse\nbrain', 'mouse_kidney': 'Mouse\nkidney'}

summ = pd.read_csv(f'{B}/tables/gene_expression/gene_expression_summary_4x6.csv')
summ = summ.set_index('dataset')
allm = pd.read_csv(f'{B}/tables/gene_expression/gene_expression_all_metrics.csv')
# HisToGene per-sample rows live in their own table; align columns and append
ht = pd.read_csv(f'{B}/tables/gene_expression/histogene_ownpanel_per_sample.csv')
allm = pd.concat([allm, ht[allm.columns.intersection(ht.columns)]],
                 ignore_index=True)

figa, axa = plt.subplots(figsize=(8.0, 5.2), facecolor='white')   # (a) 分组横条
figb, axb = plt.subplots(figsize=(4.8, 5.2), facecolor='white')     # (b) DLPFC per-sample

# ---------------------------------------------------------------- (a) bars
n_ds, n_m = len(DS_ORDER), len(MODELS)
w = 0.19
DS_LAB_H = {'DLPFC': 'DLPFC', 'breast_cancer': 'Breast cancer',
            'human_heart': 'Human heart', 'human_lymph_node': 'Lymph node',
            'mouse_brain': 'Mouse brain', 'mouse_kidney': 'Mouse kidney'}
for j, m in enumerate(MODELS):
    for i, ds in enumerate(DS_ORDER):
        v = summ.loc[ds, m]
        y = i - (j - (n_m - 1) / 2) * w
        if isinstance(v, str) or v != v:  # 'N/A'
            axa.barh(y, 0.012, w * 0.92, facecolor='#f0f0f0',
                     edgecolor='#aaaaaa', hatch='//', linewidth=0.6, zorder=2)
            axa.text(0.028, y, 'N/A', ha='left', va='center', fontsize=9,
                     color='#555555')
            continue
        axa.barh(y, v, w * 0.92, color=MCOL[m], edgecolor='none', zorder=2)
        axa.text(v + 0.008, y, f'{v:.2f}', ha='left', va='center',
                 fontsize=9, color='#111111')
axa.set_yticks(range(n_ds))
axa.set_yticklabels([DS_LAB_H[d] for d in DS_ORDER], fontsize=11.5)
axa.invert_yaxis()  # DLPFC 在最上
axa.set_xlim(0, 0.95)
axa.set_xlabel('Global Pearson r', fontsize=12.5)
axa.set_title('H&E → expression: global r by dataset',
              fontsize=14, fontweight='bold', loc='left', pad=16)
axa.spines[['top', 'right']].set_visible(False)
axa.grid(axis='x', color='#e8eef5', lw=0.6, zorder=0)
axa.tick_params(axis='x', labelsize=11)
axa.legend([plt.Rectangle((0, 0), 1, 1, color=MCOL[m]) for m in MODELS],
           MODELS, fontsize=11.5, frameon=False, loc='center left',
           bbox_to_anchor=(1.01, 0.5), ncol=1, handlelength=1.4)

# -------------------------------------------------- (b) DLPFC per-sample
dl = allm[allm.dataset == 'DLPFC']
rng = np.random.default_rng(0)
pos = np.arange(len(MODELS))
# 散点用更深的同色系,避免浅色系(BLEEP)看不清;HisToGene 用深暖色
MCOL_PT = {'OmiCLIP': '#08306b', 'STPath': '#08519c', 'HisToGene': '#3182bd',
           'BLEEP': '#6baed6'}
for j, m in enumerate(MODELS):
    v = dl[dl.model == m].sort_values('sample')['global_r_all'].values
    bp = axb.boxplot([v], positions=[pos[j]], widths=0.5, showfliers=False,
                     patch_artist=True, medianprops=dict(color='#222222', lw=1.4),
                     whiskerprops=dict(color='#555555', lw=1.0),
                     capprops=dict(color='#555555', lw=1.0))
    bp['boxes'][0].set_facecolor(MCOL[m])
    bp['boxes'][0].set_alpha(0.8)
    bp['boxes'][0].set_edgecolor(MCOL_PT[m])
    bp['boxes'][0].set_linewidth(1.2)
    axb.scatter(pos[j] + rng.uniform(-0.13, 0.13, len(v)), v, s=26,
                color=MCOL_PT[m], edgecolor='white', linewidth=0.5,
                alpha=0.95, zorder=3)
    axb.text(pos[j], v.max() + 0.015, f'{v.mean():.3f}', ha='center',
             va='bottom', fontsize=9.5, color='#111111',
             fontweight='bold')
axb.set_xticks(pos)
axb.set_xticklabels(MODELS, fontsize=11.5)
axb.set_ylim(0, 0.70)
axb.set_ylabel('Global Pearson r', fontsize=12.5)
axb.set_title('DLPFC: per-sample global r',
              fontsize=14, fontweight='bold', loc='left', x=0.10, pad=16)
axb.spines[['top', 'right']].set_visible(False)
axb.grid(axis='y', color='#e8eef5', lw=0.6, zorder=0)
axb.tick_params(axis='y', labelsize=11)

figa.tight_layout()
figa.text(0.002, 0.995, 'a', fontsize=21, fontweight='bold', ha='left', va='top')
figa.savefig(OUT_A + '.png', dpi=300, bbox_inches='tight', facecolor='white')
figa.savefig(OUT_A + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(figa)
figb.tight_layout()
figb.text(0.002, 0.995, 'b', fontsize=21, fontweight='bold', ha='left', va='top')
figb.savefig(OUT_B + '.png', dpi=300, bbox_inches='tight', facecolor='white')
figb.savefig(OUT_B + '.pdf', bbox_inches='tight', facecolor='white')
plt.close(figb)
print(f'Saved {OUT_A} / {OUT_B}')
