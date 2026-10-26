#!/usr/bin/env python3
"""Draw the Fig. 2a cross-dataset clustering summary from tables/clustering/."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _common import ROOT, OUTPUT_DIR

B = str(ROOT)
OUT = str(OUTPUT_DIR / 'fig2a_funkyheat_split')
FINAL_TABLE = f'{B}/tables/clustering/leiden_smooth_results_with_baselines.csv'
RANKING_TABLE = f'{B}/tables/clustering/clustering_overall_ranking_with_baselines.csv'

DATASETS = ['DLPFC', 'Breast', 'Mouse', 'MERFISH', 'BaristaSeq']
# 列头显示名(与 fig3a 反卷积 overview 一致:组织名 + 平台小字;
# 长名字典内不写换行,渲染时按空格逐词折行,见下方 replace)
DS_DISPLAY = {'DLPFC': ('DLPFC', ''), 'Breast': ('Breast', ''),
              'Mouse': ('Mouse brain', 'Visium'),
              'MERFISH': ('Mouse whole brain', 'MERFISH'),
              'BaristaSeq': ('Mouse visual cortex', 'BaristaSeq')}
SPECIALIZED = ['GraphST', 'STAGATE', 'SpaGCN']
DISPLAY_NAME = {'Omnicell': 'OmniCell', 'AIDO': 'AIDO.Tissue'}
METRICS = [('leiden_smooth_ari', 'ARI'), ('leiden_smooth_nmi', 'NMI')]

# ---- 模型元信息(自 Table 1;专用方法 venue 按原发表)----
META = {
    'GraphST':      ('specialized', 'Nat Commun 2023', '–', 'No'),
    'STAGATE':      ('specialized', 'Nat Commun 2022', '–', 'No'),
    'SpaGCN':       ('specialized', 'Nat Methods 2021', '–', 'No'),
    'Novae':        ('ST', 'Nat Methods 2025', 'n.r.', 'No'),
    'SToFM':        ('ST', 'ICML 2025', '~48', 'No'),
    'AIDO.Tissue':  ('ST', 'ICML 2025', '60', 'No'),
    'HEIST':        ('ST', 'ICLR 2026', '~10', 'No'),
    'OmiCLIP':      ('Path.+ST', 'Nat Methods 2025', '~638', 'Yes'),
    'STPath':       ('Path.+ST', 'npj Digit Med 2025', 'n.r.', 'Yes'),
    'Nicheformer':  ('scRNA+ST', 'Nat Methods 2025', '49.3', 'No'),
    'OmniCell':     ('scRNA+ST', 'bioRxiv 2025', '~74', 'No'),
    'scGPT-spatial': ('ST', 'bioRxiv 2025', '~55', 'No'),
}

df = pd.read_csv(FINAL_TABLE)
df['sample'] = df['sample'].astype(str)
rank = pd.read_csv(RANKING_TABLE)


def get_val(model, mode, dataset, metric):
    q = df[(df['model'] == model) & (df['dataset'] == dataset)]
    q = q[q['embedding'] == (model.lower() if model in SPECIALIZED else mode)]
    vals = pd.to_numeric(q[metric], errors='coerce').dropna().values
    return float(np.mean(vals)) if len(vals) else np.nan


sub = rank[(rank['mode'] == 'finetuned') | (rank['mode'] == 'specialized')]
sub = sub.sort_values('mean_rank')
rows = []
for m in sub['model']:
    if m in SPECIALIZED:
        rows.append((m, None, DISPLAY_NAME.get(m, m)))
    else:
        rows.append((m, 'finetuned', DISPLAY_NAME.get(m, m)))
        rows.append((m, 'frozen', DISPLAY_NAME.get(m, m)))
n_r = len(rows)

# ---- 值矩阵 + 列内归一化名次得分 ----
M = {}   # (metric, dataset) -> vector
for met, _ in METRICS:
    for ds in DATASETS:
        v = np.array([get_val(m, mode, ds, met) for m, mode, _ in rows])
        ok = ~np.isnan(v)
        r = np.full(n_r, np.nan)
        r[ok] = (-v[ok]).argsort().argsort() + 1
        M[(met, ds)] = (v, (ok.sum() + 1 - r) / ok.sum())   # 得分 1/n..1

# Overall = 全 10 列名次得分的均值
allscores = np.nanmean(np.column_stack([s for _, s in M.values()]), axis=1)

# ---- 分三块:专用方法 / ft / fz,块内按 Overall 降序 ----
idx_spec = [i for i, (m, mode, _) in enumerate(rows) if m in SPECIALIZED]
idx_ft = [i for i, (m, mode, _) in enumerate(rows) if mode == 'finetuned']
idx_fz = [i for i, (m, mode, _) in enumerate(rows) if mode == 'frozen']


# 2026-09-11: 单池排名——全部 22 行(专用+ft+fz)在每列内一起排名,
# 条长全局可比(M 在上方构建时已是全池名次得分);行序仍为 专用/ft/fz
# 三块,块内按全池 Overall 得分降序
ov = allscores
sel = (list(np.array(idx_spec)[np.argsort(-ov[idx_spec])]) +
       list(np.array(idx_ft)[np.argsort(-ov[idx_ft])]) +
       list(np.array(idx_fz)[np.argsort(-ov[idx_fz])]))
rows = [rows[i] for i in sel]
allscores = ov[sel]
N_SPEC = len(idx_spec)
N_FT = len(idx_ft)
for k in M:
    v, s = M[k]
    M[k] = (v[sel], s[sel])


def off(i):
    """第 i 行之前的块间空档数。"""
    return GAP_BLK * ((i >= N_SPEC) + (i >= N_SPEC + N_FT))

# ---- 布局 ----
ROW_H = 0.42
CHAR_COLS = [('Modality', 1.7, 0)]   # (标签, 列宽, META 下标)
N_BAR = 1 + 2 * len(DATASETS)          # Overall + ARI×5 + NMI×5
BAR_W = 1.0
GROUP_GAP = 0.45
x_char = 2.6
x_sum = x_char + sum(w for _, w, _ in CHAR_COLS) + 0.5

GAP_BLK = 0.62   # ft/fz 两块之间的空档
fig_w = x_sum + N_BAR * BAR_W + 2 * GROUP_GAP + 0.4
fig_h = n_r * ROW_H + 3.5 + 2 * GAP_BLK
fig, ax = plt.subplots(figsize=(fig_w * 0.62, fig_h * 0.62))
ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis('off')

GREY_H = '#666666'
C_OVER = '#8e2420'   # Overall:深红
C_ARI = '#4a3a85'    # ARI:深紫
C_NMI = '#084594'    # NMI:深蓝

# ---- 块标题 ----
BLOCK_LAB = [('Specialized methods', 0),
             ('Fine-tuned (FMs)', N_SPEC),
             ('Zero-shot (FMs)', N_SPEC + N_FT)]

# ---- 行底色块(三块统一浅灰,只覆盖数据行;块标题与块间空档留白) ----
for i0, i1 in [(0, N_SPEC), (N_SPEC, N_SPEC + N_FT), (N_SPEC + N_FT, n_r)]:
    y_hi = fig_h - 2.75 - i0 * ROW_H - off(i0)
    y_lo = fig_h - 2.75 - i1 * ROW_H - off(i1 - 1)
    ax.add_patch(Rectangle((0, y_lo), fig_w - 0.3, y_hi - y_lo,
                           facecolor='#f2f2f2', edgecolor='none', zorder=0))

for lab, i0 in BLOCK_LAB:
    yb = fig_h - 2.75 - i0 * ROW_H - off(i0) + 0.22
    ax.text(0.15, yb, lab, ha='left', va='bottom', fontsize=8.2, fontweight='bold',
            color='#08519c')

# ---- 行标签 ----
for i, (m, mode, disp) in enumerate(rows):
    y = fig_h - 2.75 - (i + 0.5) * ROW_H - off(i)
    if disp:
        ax.text(0.15, y, disp, ha='left', va='center', fontsize=8.6, fontweight='bold',
                color='#111111' if m not in SPECIALIZED else '#525252')


xx = x_char
for lab, w, mi in CHAR_COLS:
    ax.text(xx + w / 2, fig_h - 1.95, lab, ha='center', va='center', fontsize=7.6,
            fontweight='bold', color='#333333')
    for i, (m, mode, _) in enumerate(rows):
        key = DISPLAY_NAME.get(m, m)
        v = META.get(key, ('', '', '', ''))[mi]
        y = fig_h - 2.75 - (i + 0.5) * ROW_H - off(i)
        ax.text(xx + w / 2, y, v, ha='center', va='center', fontsize=7.4, color='#333333')
    xx += w

# ---- Summary 块 ----
sum_w = N_BAR * BAR_W + 2 * GROUP_GAP
ax.add_patch(Rectangle((x_sum - 0.1, fig_h - 1.15), sum_w + 0.2, 0.42,
                       facecolor=GREY_H, edgecolor='none', zorder=2))
ax.text(x_sum + sum_w / 2, fig_h - 0.94, 'Summary',
        ha='center', va='center', fontsize=8.5, fontweight='bold', color='white', zorder=3)

# 组标题带:Overall / ARI / NMI
ax.text(x_sum + BAR_W / 2, fig_h - 1.95, 'Overall', ha='center', va='center',
        fontsize=7.8, fontweight='bold', color=C_OVER)
ax.add_patch(Rectangle((x_sum + BAR_W + GROUP_GAP, fig_h - 1.52), 5 * BAR_W, 0.26,
                       facecolor=C_ARI, edgecolor='none', zorder=2))
ax.text(x_sum + BAR_W + GROUP_GAP + 2.5 * BAR_W, fig_h - 1.39, 'ARI', ha='center',
        va='center', fontsize=7.8, fontweight='bold', color='white', zorder=3)
x_nmi = x_sum + BAR_W + GROUP_GAP + 5 * BAR_W + GROUP_GAP
ax.add_patch(Rectangle((x_nmi, fig_h - 1.52), 5 * BAR_W, 0.26,
                       facecolor=C_NMI, edgecolor='none', zorder=2))
ax.text(x_nmi + 2.5 * BAR_W, fig_h - 1.39, 'NMI', ha='center', va='center',
        fontsize=7.8, fontweight='bold', color='white', zorder=3)

# 列标签(组织名逐词折行 + 平台小字,悬于行区上方;版式与 fig3a 一致,
# 2026-10-22:长名由两行改为逐词一行,解决 Mouse whole brain / visual cortex 拥挤;
# 同日:组织名字号 5.4→7.0、平台小字 5.0→5.8,提高可读性)
for k, ds in enumerate(DATASETS):
    main_lab, sub_lab = DS_DISPLAY[ds]
    for x0 in [x_sum + BAR_W + GROUP_GAP + k * BAR_W, x_nmi + k * BAR_W]:
        ax.text(x0 + BAR_W / 2, fig_h - 1.90, main_lab.replace(' ', '\n'),
                ha='center', va='center', fontsize=7.0, fontweight='bold',
                color='#333333', linespacing=1.1)
        if sub_lab:
            ax.text(x0 + BAR_W / 2, fig_h - 2.36, sub_lab, ha='center', va='center',
                    fontsize=5.8, fontweight='bold', color='#333333')


def bar_cell(x, y, frac, color, val=None):
    ax.add_patch(Rectangle((x + 0.08, y - 0.13), BAR_W - 0.16, 0.26,
                           facecolor='#f0f0f0', edgecolor='none', zorder=1))
    if not np.isnan(frac):
        w = (BAR_W - 0.16) * frac
        ax.add_patch(Rectangle((x + 0.08, y - 0.13), w, 0.26,
                               facecolor=color, edgecolor='none', zorder=2))
        if val is not None and not np.isnan(val):
            if frac > 0.5:   # 长条:条内右端白字
                ax.text(x + 0.08 + w - 0.03, y, f'{val:.2f}', ha='right', va='center',
                        fontsize=5.2, color='white', zorder=4)
            else:            # 短条:条端外侧深字
                ax.text(x + 0.08 + w + 0.04, y, f'{val:.2f}', ha='left', va='center',
                        fontsize=5.2, color='#444444', zorder=4)


# 单池 Overall 名次(全部 22 行按 Overall 得分排,从 1 起)
overall_rank = np.empty(n_r, dtype=int)
overall_rank[(-allscores).argsort()] = np.arange(1, n_r + 1)

for i in range(n_r):
    y = fig_h - 2.75 - (i + 0.5) * ROW_H - off(i)
    bar_cell(x_sum, y, allscores[i], C_OVER, val=allscores[i])
    ax.text(x_sum + BAR_W + 0.06, y, str(overall_rank[i]), ha='left', va='center',
            fontsize=6.8, fontweight='bold' if overall_rank[i] <= 3 else 'normal',
            color='#111111' if overall_rank[i] <= 3 else '#666666')
    for k, ds in enumerate(DATASETS):
        bar_cell(x_sum + BAR_W + GROUP_GAP + k * BAR_W, y, M[('leiden_smooth_ari', ds)][1][i], C_ARI,
                 val=M[('leiden_smooth_ari', ds)][0][i])
        bar_cell(x_nmi + k * BAR_W, y, M[('leiden_smooth_nmi', ds)][1][i], C_NMI,
                 val=M[('leiden_smooth_nmi', ds)][0][i])

ax.text(0.15, fig_h - 0.38, 'a',
        ha='left', va='center', fontsize=24, fontweight='bold', color='#111111')
ax.text(fig_w / 2, fig_h - 0.38,
        'Spatial clustering — cross-dataset summary',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#111111')

fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT + '.pdf', bbox_inches='tight', facecolor='white')
print('saved', OUT)
