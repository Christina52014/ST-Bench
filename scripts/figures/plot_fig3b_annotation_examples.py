#!/usr/bin/env python3
"""Draw Fig. 3b from the released Xenium coordinates and seven prediction tables.

Accuracy and macro-F1 are recomputed for every panel. PNG and PDF outputs are
written to figures/fig3b_annotation_examples.{png,pdf}.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
from _common import ROOT, OUTPUT_DIR

B = str(ROOT)
OUT = str(OUTPUT_DIR / 'fig3b_annotation_examples')

# model -> (prediction csv, true column, pred column)
# AIDO / HEIST per-cell predictions were not stored by their eval scripts;
# rebuilt here from the archived embeddings with the same kNN protocol
# (k=15, euclidean, distance-weighted) and verified against the summary
# table BEFORE saving (AIDO 0.2031/0.0736, HEIST 0.2622/0.0292 — exact
# match; see tables/aido|heist_xenium_breast_predictions.csv header note in
# this script's git history). HEIST was evaluated on a 20,000-cell subsample
# -> its panel shows those cells only.
MODELS = [
    ('OmniCell', 'omnicell', 'true_label', 'knn_pred'),
    ('SToFM', 'stofm', 'true', 'pred_knn'),
    ('Novae', 'novae', 'true', 'pred'),
    ('HEIST', 'heist', 'true', 'pred_knn'),
    ('scGPT-spatial', 'scgpt', 'true', 'pred_knn'),
    ('AIDO', 'aido', 'true', 'pred_knn'),
    ('Nicheformer', 'nicheformer', 'true', 'pred_knn'),
]

pos = pd.read_csv(f'{B}/tables/annotation/xenium_breast_coordinates.csv.gz')
pos['cell_id'] = pos['cell_id'].astype(str)
pos = pos.set_index('cell_id')

panels = []  # (title, labels Series indexed by cell_id)
ref = pd.read_csv(f'{B}/tables/annotation/omnicell_xenium_breast_predictions.csv')
ref['cell_id'] = ref['cell_id'].astype(str)
truth = ref.set_index('cell_id')['true_label']
panels.append(('Ground truth', truth))

for disp, fn, tc, pc in MODELS:
    p = pd.read_csv(f'{B}/tables/annotation/{fn}_xenium_breast_predictions.csv')
    p['cell_id'] = p['cell_id'].astype(str)
    p = p.set_index('cell_id')
    acc = accuracy_score(p[tc], p[pc])
    f1 = f1_score(p[tc], p[pc], average='macro', zero_division=0)
    print(f'{disp}: acc={acc:.4f} f1={f1:.4f}')
    sub = '\n(20,000-cell subsample)' if disp == 'HEIST' else ''
    panels.append((f'{disp}{sub}\nAcc={acc:.3f}  macro-F1={f1:.3f}',
                   p[pc].rename(disp)))

# shared palette: union of GT classes and any predicted-only labels
# (references contain classes absent from the GT, e.g. T_Cell_&_Tumor_Hybrid)
classes = sorted(set(truth.unique()) |
                 set(pd.concat([t for _, t in panels[1:]]).dropna().unique()))
# 按生物学谱系分组配色:同类细胞用同族色相,面板更易读
# 肿瘤系=红/粉,髓系=绿/紫,淋巴系=蓝/青,基质结构=棕/橄榄,杂交=灰/浅蓝
color_of = {
    'Invasive_Tumor':        '#c62828',  # 深红
    'Prolif_Invasive_Tumor': '#e4572e',  # 橙红
    'DCIS_1':                '#f08080',  # 浅珊瑚红
    'DCIS_2':                '#d870ad',  # 粉
    'Macrophages_1':         '#92b8a0',  # 柔和灰绿(Nicheformer 坍缩类,避免刺眼)
    'Macrophages_2':         '#3d9970',  # 深绿
    'IRF7+_DCs':             '#8e6bb5',  # 紫
    'LAMP3+_DCs':            '#c3a6e0',  # 浅紫
    'Mast_Cells':            '#f0b429',  # 金黄
    'B_Cells':               '#1f78b4',  # 蓝
    'CD4+_T_Cells':          '#17becf',  # 青
    'CD8+_T_Cells':          '#0e7c7b',  # 深青
    'Stromal':               '#c9a66b',  # 土黄
    'Endothelial':           '#8d6e63',  # 棕
    'Perivascular-Like':     '#5d4037',  # 深棕
    'Myoepi_ACTA2+':         '#a3a847',  # 橄榄
    'Myoepi_KRT15+':         '#d4d46a',  # 浅橄榄
    'Stromal_&_T_Cell_Hybrid': '#9e9e9e',  # 灰
    'T_Cell_&_Tumor_Hybrid': '#80b1d3',  # 浅蓝
}
missing = [c for c in classes if c not in color_of]
assert not missing, f'classes without color: {missing}'

labels_df = pd.concat([t for _, t in panels], axis=1)
labels_df.columns = [t for t, _ in panels]
labels_df = labels_df.reindex(pos.index)  # HEIST has only 20k rows -> NaN elsewhere
assert labels_df.iloc[:, 0].notna().mean() > 0.99

fig, axes = plt.subplots(2, 4, figsize=(25, 13.2), facecolor='white')
axes = axes.ravel()
for ax, (title, _) in zip(axes, panels):
    lab = labels_df[title]
    ok = lab.notna()
    if title.startswith('HEIST'):
        colors = [tuple(v * 0.55 for v in to_rgb(color_of[c])) for c in lab[ok]]
        point_size = 2.8
    else:
        colors = [color_of[c] for c in lab[ok]]
        point_size = 0.7
    ax.scatter(pos.loc[ok, 'x'], pos.loc[ok, 'y'], c=colors, s=point_size,
               linewidths=0, rasterized=True)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=8)

handles = [plt.Line2D([], [], marker='o', ls='', color=color_of[c],
                      markersize=11, label=c) for c in classes]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.948, 0.5),
           fontsize=17, frameon=False, ncol=1, title='Cell type',
           title_fontsize=18, handletextpad=0.5, labelspacing=0.65,
           borderaxespad=0)
fig.suptitle('Xenium breast — spatial cell-type annotation',
             fontsize=32, fontweight='bold', y=0.975)
fig.tight_layout(rect=(0, 0, 0.945, 0.935))
fig.text(0.002, 0.998, 'b', fontsize=48, fontweight='bold', ha='left', va='top')
fig.savefig(f'{OUT}.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}.pdf', bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved {OUT}.png / .pdf')
