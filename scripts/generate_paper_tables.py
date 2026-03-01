"""Generate publication-quality tables and figures for TechChunkBench paper."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AGGREGATED_DIR, FIGURES_DIR, CHUNKING_STRATEGIES, CORPORA, TOP_K_VALUES
from src.stats import friedman_test, friedman_test_fine, nemenyi_posthoc, compute_mean_ranks

# Plot style
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

STRATEGY_LABELS = {
    "fixed_size": "Fixed",
    "fixed_overlap": "Overlap",
    "sentence_based": "Sentence",
    "recursive": "Recursive",
    "semantic": "Semantic",
    "structure_aware": "Structure",
    "hybrid": "Hybrid",
}

METRIC_DISPLAY_NAMES = {
    "hit_rate_at_1": "Hit@1",
    "hit_rate_at_3": "Hit@3",
    "hit_rate_at_5": "Hit@5",
    "mrr": "MRR",
    "ndcg_at_5": "NDCG@5",
    "context_precision": "Ctx Prec",
    "rouge_l": "ROUGE-L",
    "rouge_1": "ROUGE-1",
    "rouge_2": "ROUGE-2",
}

RETRIEVAL_METRICS = [
    "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
    "mrr", "ndcg_at_5", "context_precision",
]


def load_results(results_path: str = None) -> pd.DataFrame:
    if results_path is None:
        results_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "results_with_generation_metrics.csv",
        )
    df = pd.read_csv(results_path)
    df = df[df["error"].isna()] if "error" in df.columns else df
    return df


def load_statistical_analysis(path: str = None) -> dict:
    """Load pre-computed statistical analysis (Cliff's Delta, variance decomposition, bootstrap CIs)."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results", "statistical_analysis.json",
        )
    with open(path) as f:
        return json.load(f)


def _save_table(df: pd.DataFrame, name: str, fmt: str = "github"):
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    path = os.path.join(AGGREGATED_DIR, f"{name}.md")
    with open(path, "w") as f:
        f.write(tabulate(df, headers="keys", tablefmt=fmt, floatfmt=".3f"))
    df.to_csv(os.path.join(AGGREGATED_DIR, f"{name}.csv"), index=True)


def _save_fig(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# --- Tables ---


def table1_overall_retrieval(df: pd.DataFrame):
    """Table 1: Overall retrieval performance by strategy."""
    metrics = [f"hit_rate_at_{k}" for k in TOP_K_VALUES] + ["mrr", "ndcg_at_5", "context_precision"]
    table = df.groupby("strategy")[metrics].mean()
    table = table.reindex(CHUNKING_STRATEGIES)
    table.columns = [f"Hit@{k}" for k in TOP_K_VALUES] + ["MRR", "NDCG@5", "Ctx Prec"]
    _save_table(table, "table1_overall_retrieval")
    return table


def table2_strategy_x_model(df: pd.DataFrame):
    """Table 2: Strategy x Embedding Model interaction (MRR)."""
    table = df.pivot_table(index="strategy", columns="embedding_model", values="mrr", aggfunc="mean")
    table = table.reindex(CHUNKING_STRATEGIES)
    _save_table(table, "table2_strategy_x_model")
    return table


def table3_strategy_x_domain(df: pd.DataFrame):
    """Table 3: Strategy x Domain breakdown (Hit@3), grouped by domain (8 columns)."""
    df = df.copy()
    df["domain"] = df["corpus_id"].map(lambda cid: CORPORA.get(cid, {}).get("domain", cid))
    table = df.pivot_table(index="strategy", columns="domain", values="hit_rate_at_3", aggfunc="mean")
    table = table.reindex(CHUNKING_STRATEGIES)
    _save_table(table, "table3_strategy_x_domain")
    return table


def table4_strategy_x_size(df: pd.DataFrame):
    """Table 4: Strategy x Chunk Size (MRR)."""
    table = df.pivot_table(index="strategy", columns="chunk_size", values="mrr", aggfunc="mean")
    table = table.reindex(CHUNKING_STRATEGIES)
    _save_table(table, "table4_strategy_x_size")
    return table


def table5_efficiency(df: pd.DataFrame):
    """Table 5: Efficiency comparison."""
    metrics = ["chunking_time_ms", "embedding_time_ms", "mean_retrieval_latency_ms", "num_chunks"]
    table = df.groupby("strategy")[metrics].mean()
    table = table.reindex(CHUNKING_STRATEGIES)
    table.columns = ["Chunk Time (ms)", "Embed Time (ms)", "Retrieval Lat (ms/q)", "Avg Chunks"]
    _save_table(table, "table5_efficiency")
    return table


def table6_generation_quality(df: pd.DataFrame):
    """Table 6: Generation quality — ROUGE-1, ROUGE-2, ROUGE-L by strategy."""
    rouge_cols = ["rouge_1", "rouge_2", "rouge_l"]
    available = [c for c in rouge_cols if c in df.columns and df[c].notna().any()]

    if not available:
        # Fallback: LLM judge metrics (backward compat)
        gen_cols = ["answer_correctness", "faithfulness", "completeness"]
        available = [c for c in gen_cols if c in df.columns and df[c].notna().any()]

    if not available:
        return None

    table = df.groupby("strategy")[available].mean()
    table = table.reindex(CHUNKING_STRATEGIES)
    col_map = {c: METRIC_DISPLAY_NAMES.get(c, c) for c in available}
    table = table.rename(columns=col_map)
    _save_table(table, "table6_generation_quality")
    return table


def table7_significance(df: pd.DataFrame):
    """Table 7: Nemenyi pairwise p-values for MRR.

    Note: Nemenyi post-hoc is conventionally run only after a significant
    Friedman test. Here we run it regardless to demonstrate that pairwise
    comparisons also show no significant differences, reinforcing the null.
    """
    friedman = friedman_test_fine(df, "mrr")
    pval_df = nemenyi_posthoc(df, "mrr")  # defaults to fine-grained blocking (N=216)
    if pval_df.empty:
        return None
    # Annotate whether the omnibus test was significant
    pval_df.attrs["friedman_p"] = friedman.get("p_value", float("nan"))
    pval_df.attrs["friedman_significant"] = friedman.get("significant", False)
    _save_table(pval_df, "table7_significance")
    return pval_df


# --- Figures ---


def fig1_critical_difference(df: pd.DataFrame):
    """Figure 1: Critical Difference Diagram for MRR with CD bars."""
    mean_ranks = compute_mean_ranks(df, "mrr")
    if mean_ranks.empty:
        return

    # Compute critical difference: CD = q_alpha * sqrt(k(k+1)/(6N))
    # q_alpha values for Nemenyi test at alpha=0.05
    # From Nemenyi/Studentized Range tables for k groups
    q_alpha_table = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
                     6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
    k = len(mean_ranks)
    # Count blocks used in the Friedman test (fine-grained: N=216)
    df_tmp = df.copy()
    df_tmp["_block"] = (df_tmp["corpus_id"].astype(str) + "|" +
                        df_tmp["chunk_size"].astype(str) + "|" +
                        df_tmp["embedding_model"].astype(str))
    pivot = df_tmp.pivot_table(index="_block", columns="strategy", values="mrr", aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    n = pivot.shape[0]
    q_alpha = q_alpha_table.get(k, 3.031)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    fig, ax = plt.subplots(figsize=(10, 4))

    strategies = mean_ranks.index.tolist()
    ranks = mean_ranks.values

    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.2, 1.2)
    ax.invert_xaxis()

    # Draw mean rank axis
    ax.axhline(y=0.5, color="gray", linestyle="-", linewidth=0.5)

    # Plot strategy markers
    for i, (strat, rank) in enumerate(zip(strategies, ranks)):
        label = STRATEGY_LABELS.get(strat, strat)
        y = 0.5
        ax.plot(rank, y, "o", markersize=10, color=f"C{i}", zorder=5)
        ax.annotate(
            f"{label} ({rank:.2f})",
            (rank, y),
            textcoords="offset points",
            xytext=(0, 15 + (i % 2) * 15),
            ha="center",
            fontsize=9,
        )

    # Draw CD bars using minimal non-redundant cliques.
    # A bar connects strategies whose max rank difference < CD.
    # We find maximal contiguous groups (since ranks are sorted, a contiguous
    # group where last - first < CD means all pairwise diffs < CD).
    sorted_ranks = sorted(zip(strategies, ranks), key=lambda x: x[1])
    bars = []
    for start in range(len(sorted_ranks)):
        # Find the rightmost end where the group is within CD
        best_end = start
        for end in range(start + 1, len(sorted_ranks)):
            if sorted_ranks[end][1] - sorted_ranks[start][1] <= cd:
                best_end = end
            else:
                break
        if best_end > start:
            bars.append((start, best_end))

    # Remove bars that are subsets of other bars
    minimal_bars = []
    for bar in bars:
        if not any(other[0] <= bar[0] and other[1] >= bar[1] and other != bar
                   for other in bars):
            minimal_bars.append(bar)

    bar_y = 0.25
    for start, end in minimal_bars:
        ax.plot(
            [sorted_ranks[start][1], sorted_ranks[end][1]],
            [bar_y, bar_y],
            color="black", linewidth=3, solid_capstyle="round"
        )
        bar_y -= 0.08

    # Annotate CD value
    ax.annotate(f"CD = {cd:.2f}", xy=(0.02, 0.95), xycoords="axes fraction",
                fontsize=10, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_yticks([])
    ax.set_title("Critical Difference Diagram — MRR")
    fig.tight_layout()
    _save_fig(fig, "fig1_critical_difference")


def fig2_heatmap(df: pd.DataFrame):
    """Figure 2: Heatmap — Strategy x Domain colored by Hit@3."""
    df = df.copy()
    df["domain"] = df["corpus_id"].map(lambda cid: CORPORA.get(cid, {}).get("domain", cid))
    pivot = df.pivot_table(index="strategy", columns="domain", values="hit_rate_at_3", aggfunc="mean")
    pivot = pivot.reindex(CHUNKING_STRATEGIES)
    pivot.index = [STRATEGY_LABELS.get(s, s) for s in pivot.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Hit@3 by Strategy and Domain")
    ax.set_ylabel("Strategy")
    ax.set_xlabel("Domain")
    fig.tight_layout()
    _save_fig(fig, "fig2_heatmap")


def fig3_boxplots(df: pd.DataFrame):
    """Figure 3: Box plots — MRR distribution per strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = CHUNKING_STRATEGIES
    labels = [STRATEGY_LABELS.get(s, s) for s in order]

    plot_df = df[["strategy", "mrr"]].copy()
    plot_df["strategy"] = plot_df["strategy"].map(STRATEGY_LABELS)

    sns.boxplot(data=plot_df, x="strategy", y="mrr", order=labels, ax=ax)
    ax.set_title("MRR Distribution by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("MRR")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    _save_fig(fig, "fig3_boxplots")


def fig4_pareto(df: pd.DataFrame):
    """Figure 4: Pareto frontier — Chunking Time vs MRR."""
    summary = df.groupby("strategy").agg(
        mrr=("mrr", "mean"),
        chunk_time=("chunking_time_ms", "mean"),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for strat, row in summary.iterrows():
        label = STRATEGY_LABELS.get(strat, strat)
        color_idx = CHUNKING_STRATEGIES.index(strat) if strat in CHUNKING_STRATEGIES else 0
        ax.scatter(row["chunk_time"], row["mrr"], s=120, zorder=5, color=f"C{color_idx}")
        ax.annotate(label, (row["chunk_time"], row["mrr"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Mean Chunking Time (ms)")
    ax.set_ylabel("MRR")
    ax.set_xscale("log")
    ax.set_title("Efficiency–Quality Trade-off")
    fig.tight_layout()
    _save_fig(fig, "fig4_pareto")


def fig5_chunk_size_dist(df: pd.DataFrame):
    """Figure 5: Chunk size distribution per strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = CHUNKING_STRATEGIES
    labels = [STRATEGY_LABELS.get(s, s) for s in order]

    plot_df = df[["strategy", "mean_chunk_tokens"]].copy()
    plot_df["strategy"] = plot_df["strategy"].map(STRATEGY_LABELS)

    sns.violinplot(data=plot_df, x="strategy", y="mean_chunk_tokens", order=labels, ax=ax, inner="box")
    ax.set_title("Chunk Size Distribution by Strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean Chunk Size (tokens)")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    _save_fig(fig, "fig5_chunk_size_dist")


def fig6_hit_vs_size(df: pd.DataFrame):
    """Figure 6: Hit@3 vs chunk size for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, strat in enumerate(CHUNKING_STRATEGIES):
        strat_df = df[df["strategy"] == strat]
        means = strat_df.groupby("chunk_size")["hit_rate_at_3"].mean()
        label = STRATEGY_LABELS.get(strat, strat)
        ax.plot(means.index, means.values, marker="o", label=label, color=f"C{i}")

    ax.set_xlabel("Chunk Size (tokens)")
    ax.set_ylabel("Hit@3")
    ax.set_title("Hit@3 vs Chunk Size by Strategy")
    ax.legend(loc="best")
    ax.set_xticks(df["chunk_size"].unique())
    fig.tight_layout()
    _save_fig(fig, "fig6_hit_vs_size")


def table8_variance_decomposition(stats: dict):
    """Table 8: ANOVA eta-squared variance decomposition by metric (retrieval only)."""
    vd = stats["variance_decomposition"]
    rows = []
    for metric, data in vd.items():
        if metric not in RETRIEVAL_METRICS:
            continue
        eta = data.get("partial_eta_squared", data.get("eta_squared", {}))
        row = {
            "Metric": METRIC_DISPLAY_NAMES.get(metric, metric),
            "Strategy": eta.get("strategy", 0),
            "Model": eta.get("model", 0),
            "Chunk Size": eta.get("size", 0),
            "Domain": eta.get("domain", 0),
            "Strat × Dom": eta.get("strategy_x_domain", 0),
            "Residual": eta.get("residual", 0),
            "R²": data["r_squared"],
        }
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Metric")
    _save_table(table, "table8_variance_decomposition")
    return table


def table9_cliffs_delta_summary(stats: dict):
    """Table 9: Cliff's Delta effect size summary per metric (retrieval only)."""
    cd = stats["cliffs_delta"]
    rows = []
    for metric, pairs in cd.items():
        if metric not in RETRIEVAL_METRICS:
            continue
        counts = {"negligible": 0, "small": 0, "medium": 0, "large": 0}
        max_abs = 0.0
        for pair_data in pairs.values():
            counts[pair_data["interpretation"]] += 1
            max_abs = max(max_abs, abs(pair_data["delta"]))
        rows.append({
            "Metric": METRIC_DISPLAY_NAMES.get(metric, metric),
            "Negligible": counts["negligible"],
            "Small": counts["small"],
            "Medium": counts["medium"],
            "Large": counts["large"],
            "Max |δ|": max_abs,
        })
    table = pd.DataFrame(rows).set_index("Metric")
    _save_table(table, "table9_cliffs_delta_summary")
    return table


def table10_within_domain_variance(stats: dict):
    """Table 10: Within-domain variance — per-domain MRR for each document."""
    wdv = stats.get("within_domain_variance", {})
    if not wdv:
        return None

    rows = []
    for domain, data in sorted(wdv.items()):
        for doc, mrr in sorted(data["documents"].items()):
            rows.append({
                "Domain": domain,
                "Document": doc,
                "MRR": mrr,
            })
        rows.append({
            "Domain": domain,
            "Document": f"  [{domain} std]",
            "MRR": data["std"],
        })

    table = pd.DataFrame(rows)
    # Save as CSV with domain grouping
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    table.to_csv(os.path.join(AGGREGATED_DIR, "table10_within_domain_variance.csv"), index=False)

    # Also save a summary version
    summary_rows = []
    for domain, data in sorted(wdv.items()):
        summary_rows.append({
            "Domain": domain,
            "Mean MRR": data["mean"],
            "Std": data["std"],
            "Range": data["range"],
            "N Docs": data["n_documents"],
        })
    summary = pd.DataFrame(summary_rows).set_index("Domain")
    _save_table(summary, "table10_within_domain_variance")
    return summary


def fig7_variance_decomposition(stats: dict):
    """Figure 7: Horizontal stacked bar chart of variance decomposition (retrieval only)."""
    vd = stats["variance_decomposition"]
    factors = ["Strategy", "Model", "Chunk Size", "Domain", "Strat × Dom", "Residual"]
    factor_keys = ["strategy", "model", "size", "domain", "strategy_x_domain", "residual"]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#dd8452", "#ccb974"]

    metrics = [m for m in vd.keys() if m in RETRIEVAL_METRICS]
    display_names = [METRIC_DISPLAY_NAMES.get(m, m) for m in metrics]
    data = {f: [vd[m].get("partial_eta_squared", vd[m].get("eta_squared", {})).get(k, 0) for m in metrics] for f, k in zip(factors, factor_keys)}

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = np.arange(len(metrics))
    left = np.zeros(len(metrics))

    for factor, color in zip(factors, colors):
        vals = np.array(data[factor])
        bars = ax.barh(y_pos, vals, left=left, color=color, label=factor, edgecolor="white", linewidth=0.5)
        # Annotate segments >= 5%
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.05:
                ax.text(l + v / 2, i, f"{v * 100:.0f}%", ha="center", va="center", fontsize=8, fontweight="bold")
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.set_xlabel("Proportion of Variance (η²)")
    ax.set_title("Variance Decomposition by Factor")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save_fig(fig, "fig7_variance_decomposition")


def fig8_bootstrap_ci(stats: dict):
    """Figure 8: 2x3 forest plot grid — mean ± 95% CI per strategy per metric (retrieval only)."""
    bci = stats["bootstrap_confidence_intervals"]
    metrics = [m for m in bci.keys() if m in RETRIEVAL_METRICS]
    # Use strategy order from STRATEGY_LABELS
    strategy_order = list(STRATEGY_LABELS.keys())
    strategy_labels = [STRATEGY_LABELS[s] for s in strategy_order]

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharey=True)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        metric_data = bci[metric]
        available_strategies = [s for s in strategy_order if s in metric_data]
        available_labels = [STRATEGY_LABELS[s] for s in available_strategies]
        means = [metric_data[s]["mean"] for s in available_strategies]
        ci_lows = [metric_data[s]["ci_lower"] for s in available_strategies]
        ci_highs = [metric_data[s]["ci_upper"] for s in available_strategies]
        errors_low = [m - lo for m, lo in zip(means, ci_lows)]
        errors_high = [hi - m for hi, m in zip(ci_highs, means)]

        y_pos = np.arange(len(available_strategies))
        ax.errorbar(means, y_pos, xerr=[errors_low, errors_high],
                     fmt="o", color="#4c72b0", capsize=4, capthick=1.5, markersize=6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(available_labels)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=11)
        ax.axvline(x=np.mean(means), color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Hide unused subplot if metrics < 9
    for idx in range(len(metrics), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Bootstrap 95% Confidence Intervals by Strategy", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fig8_bootstrap_ci")


def generate_all(results_path: str = None):
    """Generate all tables and figures."""
    df = load_results(results_path)
    print(f"Loaded {len(df)} result rows.")

    print("Generating tables...")
    table1_overall_retrieval(df)
    table2_strategy_x_model(df)
    table3_strategy_x_domain(df)
    table4_strategy_x_size(df)
    table5_efficiency(df)
    table6_generation_quality(df)
    table7_significance(df)

    # Load statistical analysis for new tables/figures
    try:
        stats = load_statistical_analysis()
        print("Generating statistical tables...")
        table8_variance_decomposition(stats)
        table9_cliffs_delta_summary(stats)
        table10_within_domain_variance(stats)
    except FileNotFoundError:
        stats = None
        print("Warning: statistical_analysis.json not found — skipping tables 8-10.")

    print("Generating figures...")
    fig1_critical_difference(df)
    fig2_heatmap(df)
    fig3_boxplots(df)
    fig4_pareto(df)
    fig5_chunk_size_dist(df)
    fig6_hit_vs_size(df)

    if stats is not None:
        print("Generating statistical figures...")
        fig7_variance_decomposition(stats)
        fig8_bootstrap_ci(stats)

    # Run Friedman test
    print("\nStatistical tests:")
    friedman = friedman_test_fine(df, "mrr")
    print(f"  Friedman test on MRR (N={friedman.get('n_blocks', '?')} blocks): "
          f"statistic={friedman.get('statistic', 'N/A'):.3f}, "
          f"p={friedman.get('p_value', 'N/A'):.4f}, significant={friedman.get('significant', 'N/A')}")

    print(f"\nAll outputs saved to {AGGREGATED_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    generate_all(path)
