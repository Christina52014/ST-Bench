#!/usr/bin/env python3
"""Draw the Fig. 5a deconvolution summary from tables/deconv/ into figures/."""
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _common import ROOT, OUTPUT_DIR

B = str(ROOT)
OUT = str(OUTPUT_DIR / 'fig5a_deconv_funkyheat_filled')

# ---- 模型元信息 ----
META = {
    'GraphST':       ('Specialized',),
    'Tangram':       ('Specialized',),
    'cell2location': ('Specialized',),
    'Novae':         ('ST',),
    'SToFM':         ('ST',),
    'AIDO.Tissue':   ('ST',),
    'HEIST':         ('ST',),
    'scGPT-spatial': ('ST',),
    'Nicheformer':   ('scRNA+ST',),
    'OmniCell':      ('scRNA+ST',),
    'STPath':        ('Pathology+ST',),
    'OmiCLIP':       ('Pathology+ST',),
}
SPECIALIZED = ['GraphST', 'Tangram', 'cell2location']

# ---- 数据 ----
dl = pd.read_csv(f'{B}/tables/deconv/sources/dlpfc_summary_5layer.csv')
DL_PCC = dict(zip(dl['model'], dl['mean_pcc']))
# 补测(草稿):AIDO/HEIST 的 DLPFC 5-layer pooled PCC(已有结果,口径与汇总表一致)
DL_PCC['AIDO.Tissue'] = 0.15331645663949636   # aido_dlpfc_5layer.csv (centroid)
DL_PCC['HEIST'] = 0.1753554581155674          # heist_dlpfc_5layer_centroid.csv

with open(f'{B}/tables/deconv/Table_deconvolution_summary.csv', newline='') as f:
    tbl = list(csv.DictReader(f))
# 补测(草稿):GraphST × MERFISH full 的 RMSE/JSD(已有预测 + 重建 GT 官方公式评估);
# OmniCell × Mouse Brain Anterior 的 marker PCC(同源映射后新跑,cosine_temp1)
for r in tbl:
    if r['Dataset'] == 'MERFISH full':
        r['GraphST'] = '0.0776 / 0.4469'
    if r['Dataset'] == 'Mouse Brain Anterior':
        r['OmniCell'] = '0.0176'

# STARmap(simulated, 有 GT)改用 RMSE/JSD 展示(2026-10-19 决定;PCC 全体≈0 是
# 数据集信息上限,无区分度)。取各方法 best-RMSE setting 的 (rmse, jsd):
# FM <- all_methods_detail.csv, HEIST <- heist_summary.csv, 工具 <- combined_summary.csv
# 2026-10-22:STARmap 归档 JSD 为 base=2 的 JS 距离(量程 [0,1]),
# 而 MERFISH/BaristaSeq 两列为自然对数口径(量程 [0,√ln2≈0.833]);
# 统一为自然对数口径:JSD_e = JSD_2 × √ln2(精确换算,无需重跑评估)
_sd = pd.read_csv(f'{B}/tables/deconv/sources/starmap_fm_detail.csv')
_hd = pd.read_csv(f'{B}/tables/deconv/sources/starmap_heist_summary.csv')
_cd = pd.read_csv(f'{B}/tables/deconv/sources/starmap_specialized_summary.csv')
_all = pd.concat([_sd[['model', 'rmse', 'jsd']], _hd[['model', 'rmse', 'jsd']],
                  _cd[_cd['model'].isin(['Tangram', 'GraphST', 'cell2location'])][['model', 'rmse', 'jsd']]])
_all['model'] = _all['model'].replace({'AIDO': 'AIDO.Tissue'})
_best = _all.loc[_all.groupby('model')['rmse'].idxmin()]
_LN2_SQRT = float(np.sqrt(np.log(2)))
STARM = {r['model']: (float(r.rmse), float(r.jsd) * _LN2_SQRT) for _, r in _best.iterrows()}

# (数据集标签, 指标风味, source, dataset, metric, lower_is_better, 值小数位)
# 标签一律用组织/部位全称,不用平台名与缩写,小字行已全部取消(2026-10-19 要求):
#   Mouse Brain Anterior -> Mouse brain; Simulated_seqFISH+ -> Cortex;
#   Simulated_STARmap -> Visual cortex; MERFISH full -> Whole brain;
#   BaristaSeq -> Visual cortex(与 STARmap 模拟数据同部位,第二字段放平台名小字以区分,
#   2026-10-19 要求);STARmap 列从 PCC 组移到 RMSE/JSD 组(同日决定)
COLUMNS = [
    ('DLPFC',     '', 'dlpfc', None, 'PCC', False, 2),
    ('Breast',    '', 'table', 'Breast Cancer', 'PCC', False, 2),
    ('Lymph',     '', 'table', 'Lymph Node', 'PCC', False, 2),
    ('Mouse brain', '', 'table', 'Mouse Brain Anterior', 'PCC', False, 2),
    ('Mouse cortex', '', 'table', 'Simulated_seqFISH+', 'PCC', False, 2),
    ('Mouse whole brain', '', 'table', 'MERFISH full', 'RMSE', True, 3),
    ('Mouse visual cortex', 'BaristaSeq', 'table', 'BaristaSeq', 'RMSE', True, 3),
    ('Mouse visual cortex', 'STARmap', 'starmap', None, 'RMSE', True, 3),
    ('Mouse whole brain', '', 'table', 'MERFISH full', 'JSD', True, 3),
    ('Mouse visual cortex', 'BaristaSeq', 'table', 'BaristaSeq', 'JSD', True, 3),
    ('Mouse visual cortex', 'STARmap', 'starmap', None, 'JSD', True, 3),
]
GROUPS = [('Overall', 0, 1, '#444444'),
          ('PCC \u2191', 1, 6, '#08306b'),
          ('RMSE \u2193', 6, 9, '#006d77'),
          ('JSD \u2193', 9, 12, '#6a51a3')]

methods = list(META)
GREY_NA = set()   # (method, col_idx): 评估过但 PCC 为负 -> N/A 灰点

raw = {m: [] for m in methods}
for j, (_lab, _sub, src, ds, metric, _lib, _nd) in enumerate(COLUMNS):
    for m in methods:
        v = np.nan
        if src == 'dlpfc':
            v = float(DL_PCC[m]) if m in DL_PCC else np.nan
        elif src == 'starmap':
            pair = STARM.get(m)
            if pair is not None:
                v = pair[0] if metric == 'RMSE' else pair[1]
        else:
            for r in tbl:
                if r['Dataset'] == ds:
                    s = r[m].strip()
                    if s == '' or s == '—':
                        v = np.nan
                    elif '/' in s:
                        parts = [p.strip() for p in s.split('/')]
                        v = float(parts[0]) if metric == 'RMSE' else float(parts[1])
                    else:
                        v = float(s)
                    break
        if metric == 'PCC' and v == v and v < 0:
            GREY_NA.add((m, j))
            v = np.nan
        raw[m].append(v)

# ---- 行:专用方法块 + FM 块,块内按 Overall 降序 ----
n_r = len(methods)

# 列内名次得分(只排评估过的方法;单池)
scores = {}
for j, (_lab, _sub, _src, _ds, _met, lower, _nd) in enumerate(COLUMNS):
    v = np.array([raw[m][j] for m in methods], dtype=float)
    ok = ~np.isnan(v)
    r = np.full(n_r, np.nan)
    if ok.sum():
        r[ok] = (v[ok] if lower else -v[ok]).argsort().argsort() + 1
    scores[j] = (v, (ok.sum() + 1 - r) / ok.sum())

allscores = np.nanmean(np.column_stack([s for _, s in scores.values()]), axis=1)

idx_spec = [methods.index(m) for m in SPECIALIZED]
idx_fm = [i for i in range(n_r) if i not in idx_spec]
sel = (list(np.array(idx_spec)[np.argsort(-allscores[idx_spec])]) +
       list(np.array(idx_fm)[np.argsort(-allscores[idx_fm])]))
rows = [methods[i] for i in sel]
allscores = allscores[sel]
N_SPEC = len(idx_spec)
for j in scores:
    v, s = scores[j]
    scores[j] = (v[sel], s[sel])

# ---- 布局 ----
ROW_H = 0.34
GAP_BLK = 0.45
BAR_W = 0.78
GROUP_GAP = 0.32
CHAR_COLS = [('Modality', 1.45, 0)]
x_char = 2.1
x_sum = x_char + sum(w for _, w, _ in CHAR_COLS) + 0.35
N_BAR = 1 + len(COLUMNS)
fig_w = x_sum + N_BAR * BAR_W + 3 * GROUP_GAP + 0.3
fig_h = n_r * ROW_H + 2.95 + GAP_BLK
fig, ax = plt.subplots(figsize=(fig_w * 0.62, fig_h * 0.62))
ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')

GREY_H = '#666666'

def off(i):
    return GAP_BLK * (i >= N_SPEC)

def col_x(j):
    """第 j 根条(0=Overall, 1..10=数据列)的左边缘 x。"""
    if j == 0:
        return x_sum
    g = sum(1 for _g, a, b, _c in GROUPS[1:] if j >= b) + 1
    return x_sum + j * BAR_W + g * GROUP_GAP

# ---- 行底色块(两块统一浅灰,只覆盖数据行;块标题与块间空档留白) ----
for i0, i1 in [(0, N_SPEC), (N_SPEC, n_r)]:
    y_hi = fig_h - 2.55 - i0 * ROW_H - off(i0)
    y_lo = fig_h - 2.55 - i1 * ROW_H - off(i1 - 1)
    ax.add_patch(Rectangle((0, y_lo), fig_w - 0.3, y_hi - y_lo,
                           facecolor='#f2f2f2', edgecolor='none', zorder=0))

# ---- 块标题 + 行标签 ----
for lab, i0 in [('Specialized', 0), ('Foundation models', N_SPEC)]:
    yb = fig_h - 2.55 - i0 * ROW_H - off(i0) + 0.12
    ax.text(0.15, yb, lab, ha='left', va='bottom', fontsize=8.2, fontweight='bold',
            color='#08519c')
for i, m in enumerate(rows):
    y = fig_h - 2.55 - (i + 0.5) * ROW_H - off(i)
    ax.text(0.15, y, m, ha='left', va='center', fontsize=7.6, fontweight='bold',
            color='#525252' if m in SPECIALIZED else '#111111')

# ---- Characteristics 块 ----
xx = x_char
for lab, w, mi in CHAR_COLS:
    ax.text(xx + w / 2, fig_h - 1.95, lab, ha='center', va='center', fontsize=7.6,
            fontweight='bold', color='#333333')
    for i, m in enumerate(rows):
        y = fig_h - 2.55 - (i + 0.5) * ROW_H - off(i)
        ax.text(xx + w / 2, y, META[m][mi], ha='center', va='center', fontsize=7.4,
                color='#333333')
    xx += w

# ---- Summary 头 ----
sum_w = N_BAR * BAR_W + 3 * GROUP_GAP
ax.add_patch(Rectangle((x_sum - 0.1, fig_h - 1.15), sum_w + 0.2, 0.42,
                       facecolor=GREY_H, edgecolor='none', zorder=2))
ax.text(x_sum + sum_w / 2, fig_h - 0.94, 'Summary',
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=3)

ax.text(x_sum + BAR_W / 2, fig_h - 1.95, 'Overall', ha='center', va='center',
        fontsize=7.8, fontweight='bold', color='#7f0000')
for lab, a_, b_, c_ in GROUPS[1:]:
    x0, x1 = col_x(a_), col_x(b_ - 1) + BAR_W
    ax.add_patch(Rectangle((x0, fig_h - 1.52), x1 - x0, 0.26,
                           facecolor=c_, edgecolor='none', zorder=2))
    ax.text((x0 + x1) / 2, fig_h - 1.39, lab, ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='white', zorder=3)

# 列标签(组织/部位全称;长标签按空格折两行,第二字段为平台名小字,仅用于区分同名部位;
# 数据集名 2026-10-22 起加粗)
for j, (lab, sub, *_rest) in enumerate(COLUMNS):
    xc = col_x(j + 1) + BAR_W / 2
    ax.text(xc, fig_h - 1.95, lab.replace(' ', '\n'), ha='center', va='center',
            fontsize=5.4, fontweight='bold', color='#333333', linespacing=1.1)
    if sub:
        ax.text(xc, fig_h - 2.26, sub, ha='center', va='center', fontsize=4.3,
                fontweight='bold', color='#333333')

# ---- 圆点单元格 ----
from matplotlib.colors import to_rgb

def dot_cell(x, y, frac, color, val=None, nd=2, na_style=None):
    xc = x + BAR_W / 2
    if na_style == 'grey':      # 评估过但负 PCC -> N/A 灰点
        ax.scatter([xc], [y], s=24, color='#c8c8c8', edgecolor='none', zorder=3)
        return
    if na_style == 'hollow' or np.isnan(frac):   # 未评估 -> 空心点
        ax.scatter([xc], [y], s=22, facecolor='none', edgecolor='#bbbbbb',
                   linewidth=0.9, zorder=3)
        return
    t = 0.25 + 0.75 * frac      # 颜色深浅:越好越接近本色
    base = np.array(to_rgb(color))
    c = tuple(1 - (1 - base) * t)
    ax.scatter([xc], [y], s=12 + 120 * frac, color=c, edgecolor='#555555',
               linewidth=0.4, zorder=3)
    if val is not None and not np.isnan(val):
        # 点内标原始值:统一两位小数;字色随圆底亮度切换——浅底深字、深底白字
        # y 下移 0.018:补偿数字无下行笔画导致的视觉偏高,使数字居中于圆
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        ax.text(xc, y - 0.018, f'{val:.2f}', ha='center', va='center',
                fontsize=4.2, zorder=4, color='white' if lum < 0.62 else '#333333')

# Overall 单池名次(全部 12 行)
overall_rank = np.empty(n_r, dtype=int)
overall_rank[(-allscores).argsort()] = np.arange(1, n_r + 1)

for i, m in enumerate(rows):
    y = fig_h - 2.55 - (i + 0.5) * ROW_H - off(i)
    dot_cell(x_sum, y, allscores[i], '#7f0000', val=allscores[i])
    ax.text(x_sum + BAR_W + 0.06, y, str(overall_rank[i]), ha='left', va='center',
            fontsize=6.8, fontweight='bold' if overall_rank[i] <= 3 else 'normal',
            color='#111111' if overall_rank[i] <= 3 else '#666666')
    for j, (_lab, _sub, _src, _ds, _met, _lower, nd) in enumerate(COLUMNS):
        v, s = scores[j]
        color = next(c_ for _l, a_, b_, c_ in GROUPS[1:] if a_ <= j + 1 < b_)
        na = 'grey' if (m, j) in GREY_NA else ('hollow' if np.isnan(v[i]) else None)
        dot_cell(col_x(j + 1), y, s[i], color, val=v[i], nd=nd, na_style=na)

ax.text(0.15, fig_h - 0.38, 'a',
        ha='left', va='center', fontsize=15, fontweight='bold', color='#111111')
ax.text(fig_w / 2, fig_h - 0.38,
        'Deconvolution — cross-dataset summary',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#111111')

fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '.pdf', bbox_inches='tight', facecolor='white')
print('saved', OUT)
