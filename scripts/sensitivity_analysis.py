#!/usr/bin/env python3
"""Threshold sensitivity analysis for TechChunkBench retrieval metrics.

Re-evaluates retrieval metrics at multiple fuzzy-match thresholds [0.5, 0.6, 0.7, 0.8, 0.9]
to validate the default 0.7 threshold used in evaluator.py.

Retrieval is threshold-independent (FAISS search doesn't change). This script:
1. For each config: loads cached chunk embeddings, chunk texts, builds FAISS index, retrieves top-k once
2. For each threshold: recomputes hit@{1,3,5}, MRR, NDCG@5, context_precision using _fuzzy_match with that threshold
3. Aggregates across all configs and produces output files

Usage:
    python3 scripts/sensitivity_analysis.py
    python3 scripts/sensitivity_analysis.py --thresholds 0.5 0.6 0.7 0.8 0.9
    python3 scripts/sensitivity_analysis.py --max-configs 100
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHUNKING_STRATEGIES,
    CHUNK_SIZES,
    CORPORA,
    EMBEDDING_MODELS,
    TOP_K_VALUES,
    EMBEDDINGS_DIR,
    AGGREGATED_DIR,
    FIGURES_DIR,
)
from src.retriever import build_faiss_index, retrieve


# --- Metric functions with parameterized threshold ---
# These replicate src/evaluator.py but accept a threshold parameter

def _word_set(text):
    """Tokenize text into a set of lowercased words, preserving dotted/hyphenated tokens."""
    return set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()))


def _fuzzy_match(chunk, evidence, threshold=0.7):
    """Check if chunk contains >= threshold fraction of evidence words."""
    evidence_words = _word_set(evidence)
    if not evidence_words:
        return False
    chunk_words = _word_set(chunk)
    overlap = evidence_words & chunk_words
    return len(overlap) / len(evidence_words) >= threshold


def compute_hit(retrieved_texts, evidence, threshold=0.7):
    for chunk in retrieved_texts:
        if _fuzzy_match(chunk, evidence, threshold):
            return 1
    return 0


def compute_mrr(retrieved_texts, evidence, threshold=0.7):
    for rank, chunk in enumerate(retrieved_texts, start=1):
        if _fuzzy_match(chunk, evidence, threshold):
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_texts, evidence, k=5, total_relevant=1, threshold=0.7):
    relevances = []
    found = False
    for chunk in retrieved_texts[:k]:
        if not found and _fuzzy_match(chunk, evidence, threshold):
            relevances.append(1.0)
            found = True
        else:
            relevances.append(0.0)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    num_relevant_for_ideal = min(total_relevant, k)
    if num_relevant_for_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant_for_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_context_precision(retrieved_texts, evidence, threshold=0.7):
    if not retrieved_texts:
        return 0.0
    relevant = sum(1 for chunk in retrieved_texts if _fuzzy_match(chunk, evidence, threshold))
    return relevant / len(retrieved_texts)


# --- Main analysis ---

CHUNK_CACHE_DIR = PROJECT_ROOT / "data" / "chunk_cache"
QA_PAIRS_DIR = PROJECT_ROOT / "data" / "qa_pairs"
EMBEDDINGS_PATH = Path(EMBEDDINGS_DIR)


def _get_chunk_cache_key(corpus_id, strategy, chunk_size, model_name):
    """Build chunk cache filename following run_parallel.py conventions."""
    model_short = model_name.replace("/", "_")
    if strategy in ("semantic", "hybrid"):
        return f"{corpus_id}_{strategy}_{chunk_size}_{model_short}.pkl"
    return f"{corpus_id}_{strategy}_{chunk_size}.pkl"


def _get_embedding_key(corpus_id, strategy, chunk_size, model_name):
    """Build embedding filename following embedder.py conventions."""
    model_short = model_name.replace("/", "_")
    return f"{corpus_id}_{strategy}_{chunk_size}_{model_short}.npy"


def _get_query_embedding_key(corpus_id, model_name):
    """Build query embedding filename."""
    model_short = model_name.replace("/", "_")
    return f"queries_{corpus_id}_{model_short}.npy"


def load_qa_pairs(corpus_id):
    """Load QA pairs for a corpus."""
    qa_path = QA_PAIRS_DIR / f"{corpus_id}.json"
    if not qa_path.exists():
        return []
    with open(qa_path) as f:
        data = json.load(f)
    return data["qa_pairs"]


def run_sensitivity_analysis(thresholds, max_configs=None):
    """Run sensitivity analysis across all available cached configs."""
    results = []
    configs_processed = 0

    # Enumerate all possible configs
    all_configs = []
    for corpus_id in sorted(CORPORA.keys()):
        for strategy in CHUNKING_STRATEGIES:
            for chunk_size in CHUNK_SIZES:
                for model_name in EMBEDDING_MODELS:
                    all_configs.append((corpus_id, strategy, chunk_size, model_name))

    print(f"Total possible configs: {len(all_configs)}")
    print(f"Thresholds to evaluate: {thresholds}")
    print()

    for corpus_id, strategy, chunk_size, model_name in all_configs:
        if max_configs and configs_processed >= max_configs:
            break

        # Check if cached data exists
        chunk_cache_file = CHUNK_CACHE_DIR / _get_chunk_cache_key(
            corpus_id, strategy, chunk_size, model_name
        )
        emb_file = EMBEDDINGS_PATH / _get_embedding_key(
            corpus_id, strategy, chunk_size, model_name
        )
        query_emb_file = EMBEDDINGS_PATH / _get_query_embedding_key(
            corpus_id, model_name
        )

        if not chunk_cache_file.exists() or not emb_file.exists() or not query_emb_file.exists():
            continue

        # Load data
        try:
            with open(chunk_cache_file, "rb") as f:
                chunk_data = pickle.load(f)
            chunk_embeddings = np.load(emb_file)
            query_embeddings = np.load(query_emb_file)
        except Exception as e:
            print(f"  WARNING: Failed to load {corpus_id}/{strategy}/{chunk_size}/{model_name}: {e}")
            continue

        chunk_texts = [c[0] for c in chunk_data["chunks"]]
        if not chunk_texts:
            continue

        qa_pairs = load_qa_pairs(corpus_id)
        if not qa_pairs or len(qa_pairs) != len(query_embeddings):
            continue

        # Build FAISS index once
        index = build_faiss_index(chunk_embeddings)

        # Retrieve top-k once (threshold-independent)
        max_k = max(TOP_K_VALUES)
        all_retrieved = []
        for i in range(len(qa_pairs)):
            retrieved_indices = retrieve(query_embeddings[i], index, top_k=max_k)
            retrieved_texts = [
                chunk_texts[idx] for idx in retrieved_indices
                if 0 <= idx < len(chunk_texts)
            ]
            all_retrieved.append(retrieved_texts)

        # Evaluate at each threshold
        domain = CORPORA[corpus_id]["domain"]
        for threshold in thresholds:
            row = {
                "corpus_id": corpus_id,
                "domain": domain,
                "strategy": strategy,
                "chunk_size": chunk_size,
                "embedding_model": model_name,
                "threshold": threshold,
            }

            hit_rates = {k: [] for k in TOP_K_VALUES}
            mrr_scores = []
            ndcg_scores = []
            precision_scores = []

            for i, qa in enumerate(qa_pairs):
                retrieved_texts = all_retrieved[i]
                evidence = qa["evidence"]

                for k in TOP_K_VALUES:
                    hit = compute_hit(retrieved_texts[:k], evidence, threshold)
                    hit_rates[k].append(hit)

                mrr = compute_mrr(retrieved_texts, evidence, threshold)
                mrr_scores.append(mrr)

                ndcg = compute_ndcg(retrieved_texts[:5], evidence, total_relevant=1,
                                    threshold=threshold)
                ndcg_scores.append(ndcg)

                precision = compute_context_precision(retrieved_texts[:5], evidence,
                                                      threshold=threshold)
                precision_scores.append(precision)

            for k in TOP_K_VALUES:
                row[f"hit_rate_at_{k}"] = float(np.mean(hit_rates[k])) if hit_rates[k] else 0
            row["mrr"] = float(np.mean(mrr_scores)) if mrr_scores else 0
            row["ndcg_at_5"] = float(np.mean(ndcg_scores)) if ndcg_scores else 0
            row["context_precision"] = float(np.mean(precision_scores)) if precision_scores else 0

            results.append(row)

        configs_processed += 1
        if configs_processed % 10 == 0:
            print(f"  Processed {configs_processed} configs...")

    print(f"\nTotal configs evaluated: {configs_processed}")
    return results


def generate_outputs(results, thresholds):
    """Generate output CSV, summary, and plot."""
    if not results:
        print("No results to output (no cached data found).")
        print("Run the full pipeline first: python3 run_parallel.py")
        return

    df = pd.DataFrame(results)

    # Save full results CSV
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    csv_path = os.path.join(AGGREGATED_DIR, "sensitivity_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull results: {csv_path}")

    # Aggregate: mean metric per threshold
    metrics = ["hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
               "mrr", "ndcg_at_5", "context_precision"]
    agg = df.groupby("threshold")[metrics].mean()

    # Print summary table
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Configs evaluated: {len(df) // len(thresholds)}")
    print(f"Thresholds: {thresholds}")
    print()
    print(agg.to_string(float_format="%.4f"))
    print()

    # Compute relative change from 0.7 baseline
    if 0.7 in thresholds:
        baseline = agg.loc[0.7]
        print("Relative change from 0.7 baseline:")
        for t in thresholds:
            if t == 0.7:
                continue
            row = agg.loc[t]
            changes = ((row - baseline) / baseline * 100).round(2)
            print(f"  {t}: {', '.join(f'{m}={v:+.2f}%' for m, v in changes.items())}")
        print()

    # Save summary markdown
    summary_path = os.path.join(AGGREGATED_DIR, "sensitivity_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Threshold Sensitivity Analysis\n\n")
        f.write(f"Configs evaluated: {len(df) // len(thresholds)}\n\n")
        f.write("## Mean Metrics by Threshold\n\n")
        f.write(agg.to_markdown(floatfmt=".4f"))
        f.write("\n\n")
        if 0.7 in thresholds:
            f.write("## Relative Change from 0.7 Baseline\n\n")
            f.write("| Threshold |")
            for m in metrics:
                f.write(f" {m} |")
            f.write("\n|---|")
            for _ in metrics:
                f.write("---|")
            f.write("\n")
            baseline = agg.loc[0.7]
            for t in thresholds:
                if t == 0.7:
                    continue
                row = agg.loc[t]
                changes = ((row - baseline) / baseline * 100)
                f.write(f"| {t} |")
                for m in metrics:
                    f.write(f" {changes[m]:+.2f}% |")
                f.write("\n")
            f.write("\n")
    print(f"Summary: {summary_path}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(FIGURES_DIR, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Retrieval Metric Sensitivity to Fuzzy Match Threshold", fontsize=14)

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3][idx % 3]
            ax.plot(agg.index, agg[metric], "o-", linewidth=2, markersize=8)
            ax.axvline(x=0.7, color="red", linestyle="--", alpha=0.5, label="Default (0.7)")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(metric.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(thresholds)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "sensitivity_threshold.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure: {fig_path}")
    except ImportError:
        print("matplotlib not available — skipping plot generation")


def main():
    parser = argparse.ArgumentParser(
        description="Threshold sensitivity analysis for retrieval metrics."
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="Thresholds to evaluate (default: 0.5 0.6 0.7 0.8 0.9)"
    )
    parser.add_argument(
        "--max-configs", type=int, default=None,
        help="Max configs to evaluate (for quick testing)"
    )
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)
    results = run_sensitivity_analysis(thresholds, args.max_configs)
    generate_outputs(results, thresholds)


if __name__ == "__main__":
    main()
