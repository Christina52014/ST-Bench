#!/usr/bin/env python3
"""Draw the Fig. 1b model-characteristics table as PNG and PDF in figures/."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb, to_hex
from _common import OUTPUT_DIR

OUT = str(OUTPUT_DIR)

GROUPS = [
    ('ST', [
        ('Novae', 'Spatial transcriptomics', '~30M cells', 'Graph attention (GATv2) over\nspatial neighborhoods',
         'n.r.', 'Nat Methods 2025'),
        ('SToFM', 'Spatial transcriptomics', '88M cells', 'Multi-scale Transformer for\nST representation',
         '~48*', 'ICML 2025'),
        ('AIDO.Tissue-60M', 'Multi-tissue ST + scRNA', '~22M cells', 'Asymmetric enc–dec Transformer;\ngeneralist tissue foundation model',
         '60', 'ICML 2025'),
        ('HEIST', 'Spatial transcriptomics', '22.3M cells', 'Hierarchical graph Transformer,\ncell-type aware',
         '~10*', 'ICLR 2026'),
        ('scGPT-spatial', 'ST profiles', '30M profiles', 'scGPT Transformer backbone\n+ MoE decoder',
         '~55*', 'bioRxiv 2025'),
    ]),
    ('Pathology+ST', [
        ('OmiCLIP', 'H&E patch–\ntranscriptome pairs', '2.2M pairs', 'CoCa visual–omics contrastive',
         '~638*', 'Nat Methods 2025'),
        ('STPath', 'Whole-slide images', '983 WSIs', 'Geometry-aware spatial Transformer,\npathology-guided',
         'n.r.', 'npj Digit Med 2025'),
    ]),
    ('scRNA+ST', [
        ('Nicheformer', 'scRNA + ST', '110M cells', 'Transformer, joint scRNA–ST pretraining',
         '49.3', 'Nat Methods 2025'),
        ('OmniCell', 'scRNA + ST', '~67M cells', 'Transformer, cell-centric cross-modal\npretraining',
         '~74', 'bioRxiv 2025'),
    ]),
]

# 分组配色(与 fig1a 任务色块同色系):分组标签列用深色,普通单元格用浅色,模型/参数列用更浅色
GROUP_FILL = {'ST': '#b9d2ec', 'Pathology+ST': '#f6cfA8', 'scRNA+ST': '#bfe0b3'}


def blend(c1, c2, t):
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    return to_hex((r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t))


COLS = [('Input modality', 1.85), ('Pre-trained\nST FMs', 2.1), ('Pretraining data', 2.9),
        ('Scale', 1.6), ('Architecture & characteristics', 5.8),
        ('Params (M)', 1.2), ('Venue', 2.3)]
ROW_H = 0.62
W = sum(w for _, w in COLS)
rows_all = [(g, r) for g, rs in GROUPS for r in rs]
n_r = len(rows_all)

TITLE_BAND = 0.85   # 标题区高度(拉开标题与表格的距离)
HEAD_H = 0.5        # 表头行高
BOT_PAD = 0.40      # 底部脚注区高度
fig_h = TITLE_BAND + HEAD_H + n_r * ROW_H + BOT_PAD

fig, ax = plt.subplots(figsize=(W * 0.75, fig_h * 0.75))
ax.set_xlim(0, W); ax.set_ylim(0, fig_h); ax.axis('off')

# 顶部粗线 + 底部对应粗线
y_rule_top = fig_h - TITLE_BAND
y_data_top = y_rule_top - HEAD_H
y_bot = y_data_top - n_r * ROW_H
ax.plot([0, W], [y_rule_top, y_rule_top], color='#111111', lw=2.0)
ax.plot([0, W], [y_bot, y_bot], color='#111111', lw=2.0)

# 表头:白底黑字
x = 0
for lab, w in COLS:
    ax.text(x + w / 2, y_rule_top - HEAD_H / 2, lab, ha='center', va='center', fontsize=9.2,
            fontweight='bold', color='#111111', linespacing=1.15)
    x += w

# 数据行(首列留空,后画合并格);按分组着色
y = y_data_top
for i, (gname, r) in enumerate(rows_all):
    model, src, scale, arch_char, params, venue = r
    y -= ROW_H
    fc = blend(GROUP_FILL[gname], '#ffffff', 0.45)       # 普通单元格:浅分组色
    fc_plain = blend(GROUP_FILL[gname], '#ffffff', 0.75)  # 模型/参数列:更浅
    vals = [model, src, scale, arch_char, params, venue]
    x = COLS[0][1]
    for (lab, w), v in zip(COLS[1:], vals):
        is_model = lab.startswith('Pre-trained')
        is_num = lab == 'Params (M)'
        if is_model or is_num:
            ax.add_patch(Rectangle((x, y), w, ROW_H, facecolor=fc_plain,
                                   edgecolor='none', zorder=1))
        else:
            ax.add_patch(Rectangle((x + 0.035, y + 0.045), w - 0.07, ROW_H - 0.09,
                                   facecolor=fc, edgecolor='none', zorder=2))
        ax.text(x + (0.12 if is_model else w / 2), y + ROW_H / 2, v,
                ha='left' if is_model else 'center', va='center',
                fontsize=8.6, fontweight='bold',
                color='#111111', zorder=5)
        x += w

# 首列:模态合并单元格(分组深色、横排、粗体)
gi_start = 0
for gname, grows in GROUPS:
    n = len(grows)
    y0 = y_data_top - (gi_start + n) * ROW_H
    ax.add_patch(Rectangle((0.035, y0 + 0.045), COLS[0][1] - 0.07, n * ROW_H - 0.09,
                           facecolor=GROUP_FILL[gname], edgecolor='none', zorder=2))
    ax.text(COLS[0][1] / 2, y0 + n * ROW_H / 2, gname, ha='center', va='center',
            fontsize=9.4, fontweight='bold', color='#111111')
    gi_start += n

# 标题:居中,与表格拉开距离;"a" 更大、离表格更远
ax.text(W / 2, fig_h - 0.42, 'Model characteristics',
        fontsize=13, fontweight='bold', ha='center', va='center', color='#111111')
ax.text(0, fig_h - 0.42, 'a',
        fontsize=18, fontweight='bold', ha='left', va='center', color='#111111')

# 底部脚注:单行,左对齐,与表格拉开距离
ax.text(0, 0.20, '*  Estimated from official config/checkpoint; n.r. = not reported.',
        fontsize=9.2, ha='left', va='center', color='#333333')

fig.savefig(f'{OUT}/fig1b_model_characteristics.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/fig1b_model_characteristics.pdf', bbox_inches='tight', facecolor='white')
print('saved fig1b_model_characteristics')
