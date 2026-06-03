"""
Outputs:
  Fig 1 — Decision Tree
  Fig 2 — Small Multiples: KDE Distribution (L1 left → L5 right)
  Fig 3 — Stacked Bar: Neuron Count per Level, stacked by Super-Class
  Fig 4 — Heatmap: Pre x Post synaptic level per Super-Class (connection count, no syn weights)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'xtick.major.size':   4,
    'ytick.major.size':   4,
    'xtick.minor.size':   2,
    'ytick.minor.size':   2,
    'legend.frameon':     True,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'legend.fontsize':    9.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

# ─────────────────────────────────────────────
# Color Map & Class Order (for consistency across figures)
# ─────────────────────────────────────────────
CLASS_COLOR_MAP = {
    'optic':              '#1f77b4',
    'central':            '#2ca02c',
    'visual_centrifugal': '#9467bd',
    'visual_projection':  '#ff7f0e',
    'ascending':          '#d62728',
    'sensory':            '#17becf',
    'descending':         '#e377c2',
    'motor':              '#8c564b',
    'endocrine':          '#7f7f7f',
    'DEFAULT':            '#aaaaaa',
}

BG         = 'white'
GRID_CLR   = '#e8e8e8'
ANNOT_CLR  = '#333333'
THRESH_CLR = '#1a1a1a'

# Fixed plotting order for KDE panels and stacked bar legend
CLASS_ORDER_FIXED = [
    'sensory', 'visual_centrifugal', 'optic', 'ascending',
    'visual_projection', 'central', 'endocrine', 'descending', 'motor',
]

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("Loading data...")
df       = pd.read_csv('..//COORDINATE_XY_with_levels_tree.csv')
df_class = pd.read_csv('..//classification.csv')

if 'target' in df.columns:
    df.rename(columns={'target': 'super_class'}, inplace=True)
elif 'super_class' not in df.columns:
    df = df.merge(df_class[['root_id', 'super_class']], on='root_id', how='left')

df = df.dropna(subset=['score', 'super_class', 'y_level'])
# Normalise to lowercase so CLASS_COLOR_MAP keys always match
df['super_class'] = df['super_class'].str.lower().str.strip()

# Build palette
all_classes = sorted(df['super_class'].unique())
palette = {cls: CLASS_COLOR_MAP.get(cls, CLASS_COLOR_MAP['DEFAULT']) for cls in all_classes}

# class_order for bar/heatmap
class_order = [c for c in CLASS_ORDER_FIXED if c in all_classes]
class_order += [c for c in all_classes if c not in class_order]

# class_order for KDE panels
kde_order = [c for c in CLASS_ORDER_FIXED if c in all_classes]
kde_order += [c for c in all_classes if c not in kde_order]

# ── Number of Neurons per Super-Class ──
neuron_counts = df.groupby('super_class')['root_id'].nunique().sort_values(ascending=False)
print("\n── Number of Neurons per Super-Class ─────────────────────────────────")
for cls, n in neuron_counts.items():
    print(f"  {cls:<25} {n:>8,}")
print(f"  {'TOTAL':<25} {neuron_counts.sum():>8,}")
print("────────────────────────────────────────────────────────────")

# ─────────────────────────────────────────────
# 2. FIT DECISION TREE AND ASSIGN LEVELS
# ─────────────────────────────────────────────
print("Fitting decision tree...")
X = df[['score']].values
le = LabelEncoder()
y_enc = le.fit_transform(df['super_class'])

dt = DecisionTreeClassifier(
    criterion='entropy', max_leaf_nodes=5,
    min_samples_leaf=50, random_state=42
)
dt.fit(X, y_enc)

thresholds = sorted([t for t in dt.tree_.threshold if t != -2])

# Assign DT-derived discrete level to each neuron
df['dt_level'] = dt.apply(X)

# (high score = early sensory hierarchy = L1; low/negative score = L5 motor end).
leaf_ids = df['dt_level'].unique()
leaf_mean_score = df.groupby('dt_level')['score'].mean()
# sort descending: highest mean score gets rank 0 → L1
leaf_ids_sorted = leaf_mean_score.sort_values(ascending=False).index.tolist()
leaf_to_lv = {lid: i + 1 for i, lid in enumerate(leaf_ids_sorted)}
df['level'] = df['dt_level'].map(leaf_to_lv)
n_levels   = df['level'].nunique()

print(f"  Thresholds: {[f'{t:.1f}' for t in thresholds]}")
print(f"  Levels: {n_levels}   Classes: {len(all_classes)}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — DECISION TREE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig 1] Decision Tree...")

TREE_FONTSIZE = 28
BBOX_PAD      = 0.55
FIG_W         = 22
FIG_H         = 13

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)

annotations = plot_tree(
    dt,
    feature_names=['Score'],
    class_names=le.classes_,
    filled=True,
    rounded=False,
    ax=ax,
    impurity=False,
    proportion=True,
    node_ids=False,
    fontsize=TREE_FONTSIZE,
)

def ann_is_split(text_obj):
    return '<=' in text_obj.get_text()

def ann_is_leaf(text_obj):
    return not ann_is_split(text_obj) and text_obj.get_bbox_patch() is not None

# ── Style node boxes ──
for text_obj in annotations:
    bbox = text_obj.get_bbox_patch()
    if bbox is None:
        continue
    bbox.set_facecolor('#efefef')
    bbox.set_edgecolor('#888888')
    bbox.set_linewidth(1.8)
    bbox.set_alpha(1.0)
    bbox.set_boxstyle(f'round,pad={BBOX_PAD}')

# ── Leaf Node Styling ──
leaf_annotations = sorted(
    [(t.get_position()[0], t) for t in annotations if ann_is_leaf(t)],
    key=lambda x: x[0]
)
n_leaves = len(leaf_annotations)
leaf_level_map = {id(t): (n_leaves - rank) for rank, (_, t) in enumerate(leaf_annotations)}

for text_obj in annotations:
    lines     = text_obj.get_text().split('\n')
    new_lines = []
    for line in lines:
        if '<=' in line:
            try:
                val = float(line.split('<=')[1])
                new_lines.append(f'score ≤ {val:.1f}')
            except Exception:
                new_lines.append(line)
        elif 'class' in line.lower():
            pass
        elif 'samples' in line.lower():
            new_lines.append(line.strip())

    if ann_is_leaf(text_obj) and id(text_obj) in leaf_level_map:
        level_num = leaf_level_map[id(text_obj)]
        new_lines.append(f'L{level_num}')

    text_obj.set_text('\n'.join(new_lines))
    text_obj.set_color('#1a1a1a')
    text_obj.set_fontweight('bold' if ann_is_leaf(text_obj) else 'normal')

# ── True/False labels ──
n_nodes        = dt.tree_.node_count
children_left  = dt.tree_.children_left
children_right = dt.tree_.children_right

if len(annotations) == n_nodes:
    for i in range(n_nodes):
        child_l = children_left[i]
        child_r = children_right[i]
        if child_l != -1 and child_r != -1:
            x_p, y_p = annotations[i].get_position()
            x_l, y_l = annotations[child_l].get_position()
            x_r, y_r = annotations[child_r].get_position()

            ax.text((x_p + x_l)/2, (y_p + y_l)/2, 'True',
                    fontsize=TREE_FONTSIZE * 0.75, color='#2c3e50',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
            ax.text((x_p + x_r)/2, (y_p + y_r)/2, 'False',
                    fontsize=TREE_FONTSIZE * 0.75, color='#2c3e50',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

ax.axis('off')
fig.tight_layout()
fig.savefig('DecisionTree.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved DecisionTree.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — SMALL MULTIPLES: KDE per Super-Class (L1 left → L5 right)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 2] Small-multiple KDE distributions...")

n_cls   = len(kde_order)
n_cols  = 3
n_rows  = int(np.ceil(n_cls / n_cols))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.8 * n_cols, 3.4 * n_rows),
    facecolor=BG,
    constrained_layout=True,
)
fig.get_layout_engine().set(h_pad=0.08, hspace=0.08, w_pad=0.06, wspace=0.06)
axes_flat = axes.flatten()

all_scores = df['score']
x_lo = np.percentile(all_scores, 0.5)
x_hi = np.percentile(all_scores, 99.5)
x_pad = (x_hi - x_lo) * 0.08
x_min = x_lo - x_pad
x_max = x_hi + x_pad

for i, cls in enumerate(kde_order):
    ax     = axes_flat[i]
    subset = df[df['super_class'] == cls]['score']
    color  = palette[cls]

    sns.kdeplot(
        data=subset, ax=ax,
        fill=True, alpha=0.30,
        color=color, linewidth=0,
    )
    sns.kdeplot(
        data=subset, ax=ax,
        fill=False, alpha=0.9,
        color=color, linewidth=1.2,
    )

    for j, t in enumerate(thresholds):
        ax.axvline(t, color=THRESH_CLR, linestyle='--',
                   linewidth=0.9, alpha=0.45)

    # thresholds are sorted ascending (left to right on x-axis = low to high score).
    # L1 at the leftmost band (lowest scores), L5 at the rightmost (highest scores).
    band_edges = [x_min] + thresholds + [x_max]
    n_bands = len(band_edges) - 1  # equals n_levels = 5
    for b in range(n_bands):
        ax.axvspan(band_edges[b], band_edges[b + 1],
                   alpha=0.07, color='#666666', zorder=0)
        mid_x = (band_edges[b] + band_edges[b + 1]) / 2
        # Inverted axis: leftmost band (highest scores) -> L1; rightmost -> L5
        level_label = n_bands - b
        ax.text(
            mid_x, 1.0, f'L{level_label}',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top',
            fontsize=7, color='#888888', style='italic',
        )

    n      = len(subset)
    mu     = subset.mean()
    sigma  = subset.std()
    ax.set_title(cls, fontsize=12, fontweight='bold', color=color, pad=6)
    ax.set_xlabel('Flow Score', fontsize=9.5)
    ax.set_ylabel('Density', fontsize=9.5)
    ax.set_xlim(x_max, x_min)   # inverted: high score (L1) on left, low score (L5) on right
    ax.tick_params(labelsize=8.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    ax.annotate(
        f'n={n:,}   μ={mu:.1f}   σ={sigma:.1f}',
        xy=(0.97, 0.94), xycoords='axes fraction',
        ha='right', va='top', fontsize=7.5,
        color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#dddddd', alpha=0.85),
    )
    sns.despine(ax=ax)

for j in range(n_cls, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(None)

fig.savefig('KDE_SmallMultiples.png')
plt.close(fig)
print("  Saved KDE_SmallMultiples.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — STACKED BAR: Neuron Count per Level, stacked by Super-Class
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 3] Stacked bar chart...")

pivot = (
    df.groupby(['level', 'super_class'])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=class_order)
)

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=BG)

bottoms = np.zeros(len(pivot))
for cls in class_order:
    vals = pivot[cls].values
    bars = ax.bar(
        pivot.index, vals,
        bottom=bottoms,
        color=palette[cls],
        label=cls,
        width=0.62,
        edgecolor='white',
        linewidth=0.8,
    )
    bottoms += vals

totals = pivot.sum(axis=1)
for x, total in zip(pivot.index, totals):
    ax.text(x, total + totals.max() * 0.012, f'{total:,}',
            ha='center', va='bottom', fontsize=9.5, color=ANNOT_CLR, fontweight='bold')

ax.set_xlabel('Hierarchical Level', fontsize=11)
ax.set_ylabel('Neuron Count', fontsize=11)
ax.set_title(None)

ax.set_xticks(pivot.index)
ax.set_xticklabels([f'Level {l}' for l in pivot.index], fontsize=10)
ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(labelsize=10)

legend = ax.legend(
    title='Super-Class', title_fontsize=10,
    fontsize=9.5, loc='upper left',
    bbox_to_anchor=(1.01, 1.0),
    framealpha=0.95, edgecolor='#cccccc',
    ncol=1,
)
sns.despine(ax=ax)
fig.tight_layout(rect=[0, 0, 0.82, 1])
fig.savefig('StackedBar.png')
plt.close(fig)
print("  Saved StackedBar.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — HEATMAP: Pre × Post synaptic level per Super-Class (connection count, no syn weights)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 4] Connection heatmaps...")

try:
    conn = pd.read_csv('..//connections.csv')

    id_to_level = df.set_index('root_id')['level'].to_dict()
    id_to_class = df.set_index('root_id')['super_class'].to_dict()

    conn['pre_level']  = conn['pre_root_id'].map(id_to_level)
    conn['post_level'] = conn['post_root_id'].map(id_to_level)
    conn['pre_class']  = conn['pre_root_id'].map(id_to_class)
    conn = conn.dropna(subset=['pre_level', 'post_level', 'pre_class'])
    conn['pre_level']  = conn['pre_level'].astype(int)
    conn['post_level'] = conn['post_level'].astype(int)

    # ── Calculate feedforward / feedback / lateral (unit weight) ──────────────
    total       = len(conn)
    feedforward = (conn['pre_level'] < conn['post_level']).sum()
    feedback    = (conn['pre_level'] > conn['post_level']).sum()
    lateral     = (conn['pre_level'] == conn['post_level']).sum()

    ff_pct  = feedforward / total * 100
    fb_pct  = feedback    / total * 100
    lat_pct = lateral     / total * 100

    print(f"\n── Connection-based path analysis ──────────────────────────────")
    print(f"  Total connections          : {total:,}")
    print(f"  Feedforward (pre < post)   : {feedforward:,}  →  {ff_pct:.2f}%")
    print(f"  Feedback    (pre > post)   : {feedback:,}  →  {fb_pct:.2f}%")
    print(f"  Lateral     (pre = post)   : {lateral:,}  →  {lat_pct:.2f}%")
    print(f"────────────────────────────────────────────────────────────────")
    # ─────────────────────────────────────────────────────────────────────────

    levels   = sorted(df['level'].unique())
    n_lv     = len(levels)
    hm_order = [c for c in kde_order if c in conn['pre_class'].unique()]

    n_cols_hm = 3
    n_rows_hm = int(np.ceil(len(hm_order) / n_cols_hm))

    fig, axes = plt.subplots(
        n_rows_hm, n_cols_hm,
        figsize=(5.5 * n_cols_hm, 4.8 * n_rows_hm),
        facecolor=BG,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for idx, cls in enumerate(hm_order):
        ax    = axes_flat[idx]
        color = palette.get(cls, CLASS_COLOR_MAP['DEFAULT'])
        sub   = conn[conn['pre_class'] == cls]

        # Create a matrix of connection counts: pre_level x post_level
        mat = (
            sub.groupby(['pre_level', 'post_level'])
               .size()
               .unstack(fill_value=0)
               .reindex(index=levels, columns=levels, fill_value=0)
        )
        weight_label = 'Connection count'

        mat_vals = mat.values.astype(float)
        vmax_abs = mat_vals.max()

        non_zero = mat_vals[mat_vals > 0]
        if len(non_zero) > 0:
            vmax_clip = np.percentile(non_zero, 95)
            if vmax_clip < non_zero.min() * 1.5:
                vmax_clip = vmax_abs  # safety fallback for flat matrices
        else:
            vmax_clip = 1

        norm = PowerNorm(gamma=0.25, vmin=0, vmax=vmax_clip) if vmax_abs > 0 else None
        cmap = LinearSegmentedColormap.from_list(f'cmap_{cls}', ['#fafafa', color], N=256)

        im = ax.imshow(mat_vals, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

        # Annotate cells
        for r in range(n_lv):
            for c in range(n_lv):
                val = mat_vals[r, c]
                if val == 0:
                    continue
                intensity = (val / vmax_clip) ** 0.25 if vmax_clip > 0 else 0
                txt_color = 'white' if intensity > 0.65 else '#1a1a1a'
                display   = f'{int(val):,}' if val < 1e6 else f'{val / 1e3:.0f}k'
                ax.text(c, r, display, ha='center', va='center', fontsize=7.5, color=txt_color)

        # Diagonal markers
        for r in range(n_lv):
            ax.add_patch(plt.Rectangle(
                (r - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor='#555555', linewidth=1.5, linestyle=':',
            ))

        ax.set_xticks(range(n_lv))
        ax.set_yticks(range(n_lv))
        ax.set_xticklabels([f'L{l}' for l in levels], fontsize=9)
        ax.set_yticklabels([f'L{l}' for l in levels], fontsize=9)
        ax.set_xlabel('Post-synaptic level', fontsize=9.5)
        ax.set_ylabel('Pre-synaptic level',  fontsize=9.5)
        ax.set_title(cls, fontsize=12, fontweight='bold', color=color, pad=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, extend='max')
        cbar.set_label(weight_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7.5)

    for j in range(len(hm_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.savefig('Heatmap.png')
    plt.close(fig)
    print("  Saved Heatmap.png")

except FileNotFoundError:
    print("connections.csv not found — skipping Heatmap.")

print("\nAll figures generated successfully.")
