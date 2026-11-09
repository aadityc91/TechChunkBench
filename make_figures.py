"""Figures for the CCIS extension (vector PDF, legible in B/W: hatching + markers carry identity).
Reads results/aggregated/v2_*.csv and mustdo_D_corpus_features.csv; writes to the paper's figs/ dir."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import AGGREGATED_DIR
FIGS = "/mnt/c/Users/nivas/OneDrive/Documents/Work/chunk-paper/extension/writing/paper/figs"; os.makedirs(FIGS, exist_ok=True)
# validated categorical slots (dataviz reference palette, light): blue, orange, aqua, yellow, magenta
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
METRICS = ["hit1", "hit3", "hit5", "mrr", "ndcg5", "cprec"]; MLAB = ["Hit@1", "Hit@3", "Hit@5", "MRR", "NDCG@5", "Ctx. prec."]
T = pd.read_csv(f"{AGGREGATED_DIR}/v2_regime_table.csv")
# conference full decomposition (Table 1 of the DATA paper), as shares
CONF = pd.DataFrame({"metric": METRICS,
    "eta2_corpus": [.620, .251, .237, .425, .265, .778], "eta2_model": [.235, .419, .498, .358, .470, .111],
    "eta2_chunk_size": [.024, .041, .032, .042, .050, .004], "eta2_strategy": [.009, .014, .019, .015, .020, .017],
    "eta2_residual": [.112, .275, .214, .160, .196, .090]})

# ---- Fig 1: stacked eta^2 by regime ----
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
panels = [("R1 general QA (conference)", CONF), ("R2 technical documentation", T[T.regime == "R2_technical"]), ("R3 long legal contracts", T[T.regime == "R3_legal"])]
factors = [("eta2_strategy", "Chunking strategy", C[0], "////"), ("eta2_chunk_size", "Chunk size", C[3], ""), ("eta2_model", "Embedding model", C[1], ""), ("eta2_corpus", "Corpus / dataset", C[2], ""), ("eta2_residual", "Residual", "#c3c2b7", "")]
for ax, (title, df) in zip(axes, panels):
    df = df.set_index("metric").reindex(METRICS); left = np.zeros(len(METRICS)); y = np.arange(len(METRICS))[::-1]
    for col, lab, color, hatch in factors:
        v = df[col].fillna(0).values * 100
        ax.barh(y, v, left=left, color=color, hatch=hatch, edgecolor="white", linewidth=0.8, height=0.62, label=lab)
        if col == "eta2_strategy":
            for yi, vi, li in zip(y, v, left): ax.text(li + vi + 1.2, yi, f"{vi:.1f}%", va="center", ha="left", fontsize=6.5, color=INK)
        left += v
    ax.set_title(title, fontsize=8, color=INK, loc="left"); ax.set_xlim(0, 100); ax.set_yticks(y); ax.set_yticklabels(MLAB); ax.set_xlabel("share of variance (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.32), ncol=5, frameon=False, fontsize=7, handlelength=1.4)
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_regimes_eta2.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 2: heading density vs per-corpus strategy effect ----
F = pd.read_csv(f"{AGGREGATED_DIR}/mustdo_D_corpus_features.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
style = {"R2_technical": (C[0], "o", "R2 technical"), "R3_legal": (C[1], "s", "R3 legal")}
for reg, (color, mk, lab) in style.items():
    g = F[F.regime == reg]; ax.scatter(g["headings_per_10k"], g["eta2_strategy_mrr"] * 100, s=28, c=color, marker=mk, edgecolors="white", linewidths=0.6, label=lab, zorder=3)
for _, r in F.iterrows():
    if r["eta2_strategy_mrr"] > 0.45 or r["headings_per_10k"] > 12 or r["regime"] == "R3_legal":
        ax.annotate(r["corpus"], (r["headings_per_10k"], r["eta2_strategy_mrr"] * 100), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=INK2)
ax.set_xlabel("headings per 10k characters"); ax.set_ylabel("within-corpus $\\eta^2$ of strategy, MRR (%)")
ax.grid(color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(f"{FIGS}/fig_heading_density.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: 2x2 prefix ablation per regime (if available) ----
p = f"{AGGREGATED_DIR}/v2_prefix_ablation.csv"
if os.path.exists(p):
    A = pd.read_csv(p); fig, axes = plt.subplots(1, len(A), figsize=(3.2 * len(A), 2.4), squeeze=False)
    for ax, (_, r) in zip(axes[0], A.iterrows()):
        vals = [r["fixed_mrr"], r["struct_noprefix_mrr"], r["struct_prefix_mrr"]]; labs = ["fixed-size\n(no prefix)", "structural\nno prefix", "structural\n+ prefix"]
        bars = ax.bar(labs, vals, color=[C[3], C[0], C[0]], hatch=["", "", "////"], edgecolor="white", width=0.6)
        for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5, color=INK)
        lo = min(vals) - 0.05; ax.set_ylim(max(0, lo), max(vals) + 0.03); ax.set_ylabel("MRR"); ax.set_title(f"{r['regime'].replace('_', ' ')} (n={int(r['n_pairs'])} paired cells)", fontsize=8, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5); ax.set_axisbelow(True); ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_prefix_ablation.pdf", bbox_inches="tight"); plt.close(fig)
print("figures written to", FIGS, os.listdir(FIGS))
