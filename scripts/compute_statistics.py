"""Compute statistical analyses: Cliff's Delta, variance decomposition, bootstrap CIs.

Optionally run compute_extra_rouge.py first for ROUGE-1/2 (not required).
Uses results/results_with_generation_metrics.csv.
Output: results/statistical_analysis.json
"""

import json
import logging
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHUNKING_STRATEGIES, CORPORA, RESULTS_DIR
from src.stats import cliffs_delta_paired

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Metrics to analyze
ANALYSIS_METRICS = [
    "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
    "mrr", "ndcg_at_5", "context_precision",
    "rouge_l", "rouge_1", "rouge_2",
]


def compute_cliffs_delta_all(df):
    """Compute Cliff's Delta for all pairwise strategy comparisons."""
    strategies = sorted(df["strategy"].unique())
    pairs = list(combinations(strategies, 2))
    logger.info(f"Computing Cliff's Delta for {len(pairs)} strategy pairs")

    results = {}
    for metric in ANALYSIS_METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue

        metric_results = {}
        for s1, s2 in pairs:
            delta, interpretation = cliffs_delta_paired(df, s1, s2, metric)
            metric_results[f"{s1}_vs_{s2}"] = {
                "delta": delta,
                "interpretation": interpretation,
            }
        results[metric] = metric_results

    return results


def compute_variance_decomposition(df):
    """Variance decomposition using Type III ANOVA with strategy x domain interaction.

    Type III SS is used (with Sum coding) so that main-effect tests remain
    valid in the presence of the interaction term, even if the design becomes
    slightly unbalanced due to failed configs.
    """
    import re

    import statsmodels.api as sm
    from patsy.contrasts import Sum
    from statsmodels.formula.api import ols

    results = {}

    for metric in ANALYSIS_METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue

        # Need a clean subset for the formula
        sub = df[["strategy", "embedding_model", "chunk_size", "corpus_id", metric]].dropna()
        if len(sub) < 10:
            continue

        # Map corpus_id to domain using CORPORA dict
        sub = sub.copy()
        sub["domain"] = sub["corpus_id"].map(lambda cid: CORPORA.get(cid, {}).get("domain", cid))
        sub = sub.rename(columns={
            "embedding_model": "model",
            "chunk_size": "size",
        })

        try:
            formula = (
                f"{metric} ~ C(strategy, Sum) + C(model, Sum) + C(size, Sum)"
                f" + C(domain, Sum) + C(strategy, Sum):C(domain, Sum)"
            )
            model = ols(formula, data=sub).fit()
            anova_table = sm.stats.anova_lm(model, typ=3)

            # Partial eta-squared: SS_factor / (SS_factor + SS_residual)
            # This is the correct effect size for Type III SS, where each
            # factor's SS represents unique variance after controlling for
            # all other factors.
            ss_residual = anova_table.loc["Residual", "sum_sq"]
            partial_eta_sq = {}
            for factor in anova_table.index:
                if factor == "Intercept":
                    continue
                ss_f = anova_table.loc[factor, "sum_sq"]
                if factor == "Residual":
                    partial_eta_sq["residual"] = float(ss_f / (ss_f + ss_residual))
                elif ":" in factor:
                    partial_eta_sq["strategy_x_domain"] = float(ss_f / (ss_f + ss_residual))
                else:
                    clean_name = re.sub(r"C\((\w+)(?:,\s*\w+)?\)", r"\1", factor)
                    partial_eta_sq[clean_name] = float(ss_f / (ss_f + ss_residual))

            results[metric] = {
                "partial_eta_squared": partial_eta_sq,
                "r_squared": float(model.rsquared),
                "n_observations": len(sub),
            }
        except Exception as e:
            logger.warning(f"  ANOVA failed for {metric}: {e}")

    return results


def compute_within_domain_variance(df):
    """Compute within-domain variance: for each domain, MRR std/range across its documents."""
    results = {}

    # Map corpus_id to domain
    df = df.copy()
    df["domain"] = df["corpus_id"].map(lambda cid: CORPORA.get(cid, {}).get("domain", cid))

    for domain in sorted(df["domain"].unique()):
        domain_df = df[df["domain"] == domain]
        corpus_ids = sorted(domain_df["corpus_id"].unique())

        doc_mrrs = {}
        for cid in corpus_ids:
            cid_mrr = domain_df[domain_df["corpus_id"] == cid]["mrr"].mean()
            doc_mrrs[cid] = float(cid_mrr)

        mrr_values = list(doc_mrrs.values())
        results[domain] = {
            "documents": doc_mrrs,
            "mean": float(np.mean(mrr_values)),
            "std": float(np.std(mrr_values, ddof=1)) if len(mrr_values) > 1 else 0.0,
            "range": float(max(mrr_values) - min(mrr_values)) if len(mrr_values) > 1 else 0.0,
            "n_documents": len(corpus_ids),
        }

    return results


def compute_bootstrap_cis(df, n_bootstrap=1000, seed=42):
    """Cluster bootstrap 95% CIs per strategy per metric.

    Resamples at the document (corpus_id) level to respect the repeated-
    measures design: each document contributes multiple rows (sizes × models),
    so treating rows as independent would underestimate variance and yield
    artificially tight CIs.
    """
    rng = np.random.RandomState(seed)
    strategies = sorted(df["strategy"].unique())
    results = {}

    for metric in ANALYSIS_METRICS:
        if metric not in df.columns or df[metric].isna().all():
            continue

        metric_results = {}
        for strategy in strategies:
            strat_df = df[df["strategy"] == strategy][[metric, "corpus_id"]].dropna(subset=[metric])
            if len(strat_df) == 0:
                continue

            corpus_ids = strat_df["corpus_id"].unique()
            n_clusters = len(corpus_ids)
            if n_clusters == 0:
                continue

            # Pre-compute per-cluster metric arrays for speed
            cluster_values = {cid: strat_df[strat_df["corpus_id"] == cid][metric].values
                              for cid in corpus_ids}

            boot_means = []
            for _ in range(n_bootstrap):
                # Resample clusters (documents) with replacement
                sampled_ids = rng.choice(corpus_ids, size=n_clusters, replace=True)
                sampled_values = np.concatenate([cluster_values[cid] for cid in sampled_ids])
                boot_means.append(np.mean(sampled_values))

            ci_lower = float(np.percentile(boot_means, 2.5))
            ci_upper = float(np.percentile(boot_means, 97.5))
            metric_results[strategy] = {
                "mean": float(strat_df[metric].mean()),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(strat_df),
                "n_clusters": n_clusters,
            }
        results[metric] = metric_results

    return results


def main():
    csv_path = os.path.join(RESULTS_DIR, "results_with_generation_metrics.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Input file not found: {csv_path}")
        logger.error("Run scripts/aggregate_results.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Filter available metrics
    available = [m for m in ANALYSIS_METRICS if m in df.columns and not df[m].isna().all()]
    logger.info(f"Available metrics for analysis: {available}")

    # 1. Cliff's Delta
    logger.info("\n=== Cliff's Delta ===")
    cliffs = compute_cliffs_delta_all(df)
    for metric, pairs in cliffs.items():
        large_effects = sum(1 for p in pairs.values() if p["interpretation"] == "large")
        logger.info(f"  {metric}: {len(pairs)} pairs, {large_effects} large effects")

    # 2. Variance decomposition
    logger.info("\n=== Variance Decomposition ===")
    variance = compute_variance_decomposition(df)
    for metric, result in variance.items():
        eta = result["partial_eta_squared"]
        logger.info(f"  {metric}: " + ", ".join(f"{k}={v:.3f}" for k, v in eta.items()))

    # 3. Bootstrap CIs
    logger.info("\n=== Bootstrap CIs ===")
    cis = compute_bootstrap_cis(df)
    for metric in list(cis.keys())[:3]:
        logger.info(f"  {metric}:")
        for strategy, ci in sorted(cis[metric].items(), key=lambda x: -x[1]["mean"]):
            logger.info(f"    {strategy}: {ci['mean']:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # 4. Within-domain variance
    logger.info("\n=== Within-Domain Variance ===")
    within_domain = compute_within_domain_variance(df)
    for domain, data in within_domain.items():
        logger.info(f"  {domain}: mean={data['mean']:.4f}, std={data['std']:.4f}, range={data['range']:.4f}")
        for doc, mrr in data["documents"].items():
            logger.info(f"    {doc}: {mrr:.4f}")

    # Save results
    output = {
        "cliffs_delta": cliffs,
        "variance_decomposition": variance,
        "bootstrap_confidence_intervals": cis,
        "within_domain_variance": within_domain,
    }

    output_path = os.path.join(RESULTS_DIR, "statistical_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Validation
    logger.info("\n=== Validation ===")
    for metric, result in variance.items():
        eta = result["partial_eta_squared"]
        logger.info(f"  {metric} partial eta-sq: " + ", ".join(f"{k}={v:.3f}" for k, v in eta.items()))

    for metric, pairs in cliffs.items():
        for pair_name, pair_data in pairs.items():
            if abs(pair_data["delta"]) > 1.0:
                logger.warning(f"  Invalid Cliff's Delta: {metric}/{pair_name} = {pair_data['delta']}")


if __name__ == "__main__":
    main()
