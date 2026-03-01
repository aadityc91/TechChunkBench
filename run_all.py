"""
Master experiment runner for TechChunkBench.

Runs all combinations of:
- 7 chunking strategies
- 3 chunk sizes (256, 512, 1024 tokens)
- 3 embedding models
- 24 documents across 8 domains (3 per domain)

Total: 7 x 3 x 3 x 24 = 1,512 configurations
Each configuration evaluated on ~40 QA pairs per corpus.

Results saved to results/raw/ as individual CSV files and
results/aggregated/ as summary tables.
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CHUNKING_STRATEGIES,
    CHUNK_SIZES,
    CORPORA,
    EMBEDDING_MODELS,
    TOP_K_VALUES,
    USE_LLM_JUDGE,
    CHECKPOINT_EVERY,
    TIMING_REPEATS,
    RAW_RESULTS_DIR,
    AGGREGATED_DIR,
    FIGURES_DIR,
    MIN_CORPORA,
    MIN_EMBEDDING_MODELS,
)
from src.chunkers import get_chunker
from src.chunkers.base import BaseChunker
from src.document_loader import load_all_corpora
from src.embedder import embed_chunks, embed_queries, get_embed_fn
from src.retriever import build_faiss_index, retrieve
from src.evaluator import (
    _fuzzy_match,
    compute_hit,
    compute_mrr,
    compute_ndcg,
    compute_context_precision,
    get_most_relevant_sentence,
    compute_rouge_l,
)
from src.qa_generator import load_qa_pairs
from src.llm_judge import (
    is_ollama_available,
    generate_rag_answer,
    judge_answer_correctness,
    judge_faithfulness,
    judge_completeness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("techchunkbench_run.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_single_configuration(
    corpus_id: str,
    strategy_name: str,
    chunk_size: int,
    model_name: str,
    qa_pairs: list,
    doc_text: str,
    use_llm_judge: bool = False,
) -> dict:
    """Run a single experimental configuration and return results dict."""
    results = {
        "corpus_id": corpus_id,
        "domain": CORPORA[corpus_id]["domain"],
        "strategy": strategy_name,
        "chunk_size": chunk_size,
        "embedding_model": model_name,
    }

    # 1. Chunk the document (with timing)
    if strategy_name in ("semantic", "hybrid"):
        embed_fn = get_embed_fn(model_name)
        chunker = get_chunker(strategy_name, target_size=chunk_size, embed_fn=embed_fn)
    else:
        chunker = get_chunker(strategy_name, target_size=chunk_size)

    # Time chunking (average of TIMING_REPEATS)
    chunk_times = []
    chunks = None
    for _ in range(TIMING_REPEATS):
        start = time.perf_counter()
        chunks = chunker.chunk(doc_text, corpus_id)
        chunk_times.append((time.perf_counter() - start) * 1000)

    results["chunking_time_ms"] = np.mean(chunk_times)
    results["num_chunks"] = len(chunks)

    chunk_texts = [c.text for c in chunks]
    chunk_token_counts = [BaseChunker.count_tokens(t) for t in chunk_texts]
    results["mean_chunk_tokens"] = np.mean(chunk_token_counts) if chunk_token_counts else 0
    results["std_chunk_tokens"] = np.std(chunk_token_counts) if chunk_token_counts else 0

    if not chunks:
        results["error"] = "No chunks produced"
        return results

    # 2. Embed chunks (with timing, using cache)
    start = time.perf_counter()
    chunk_embeddings = embed_chunks(
        chunk_texts, model_name,
        corpus_id=corpus_id, strategy=strategy_name, size=chunk_size,
    )
    results["embedding_time_ms"] = (time.perf_counter() - start) * 1000

    # 3. Build FAISS index
    index = build_faiss_index(chunk_embeddings)

    # 4. Embed queries
    query_texts = [qa["question"] for qa in qa_pairs]
    query_embeddings = embed_queries(query_texts, model_name)

    # 5. Retrieve and evaluate
    max_k = max(TOP_K_VALUES)
    all_hit_rates = {k: [] for k in TOP_K_VALUES}
    all_mrr = []
    all_ndcg = []
    all_precision = []
    retrieval_latencies = []

    for i, qa in enumerate(qa_pairs):
        start = time.perf_counter()
        retrieved_indices = retrieve(query_embeddings[i], index, top_k=max_k)
        retrieval_latencies.append((time.perf_counter() - start) * 1000)

        retrieved_texts = [chunk_texts[idx] for idx in retrieved_indices if 0 <= idx < len(chunk_texts)]
        evidence = qa["evidence"]

        for k in TOP_K_VALUES:
            hit = compute_hit(retrieved_texts[:k], evidence)
            all_hit_rates[k].append(hit)

        mrr = compute_mrr(retrieved_texts, evidence)
        all_mrr.append(mrr)

        # Each QA pair has exactly one evidence passage; overlapping chunks
        # matching the same evidence are duplicates, not independent relevant
        # documents, so IDCG should always use total_relevant=1.
        ndcg = compute_ndcg(retrieved_texts[:5], evidence, total_relevant=1)
        all_ndcg.append(ndcg)

        precision = compute_context_precision(retrieved_texts[:5], evidence)
        all_precision.append(precision)

    # Aggregate retrieval metrics
    for k in TOP_K_VALUES:
        results[f"hit_rate_at_{k}"] = np.mean(all_hit_rates[k]) if all_hit_rates[k] else 0
    results["mrr"] = np.mean(all_mrr) if all_mrr else 0
    results["ndcg_at_5"] = np.mean(all_ndcg) if all_ndcg else 0
    results["context_precision"] = np.mean(all_precision) if all_precision else 0
    results["mean_retrieval_latency_ms"] = np.mean(retrieval_latencies) if retrieval_latencies else 0

    # 6. Generation metrics
    if use_llm_judge:
        correctness_scores = []
        faithfulness_scores = []
        completeness_scores = []

        for i, qa in enumerate(qa_pairs):
            retrieved_indices = retrieve(query_embeddings[i], index, top_k=3)
            context = "\n---\n".join(
                [chunk_texts[idx] for idx in retrieved_indices if 0 <= idx < len(chunk_texts)]
            )

            generated = generate_rag_answer(qa["question"], context)
            correctness = judge_answer_correctness(qa["question"], qa["answer"], generated, context)
            faith = judge_faithfulness(generated, context)
            complete = judge_completeness(qa["question"], qa["answer"], generated)

            correctness_scores.append(correctness)
            faithfulness_scores.append(faith)
            completeness_scores.append(complete)

        results["answer_correctness"] = np.mean(correctness_scores) if correctness_scores else 0
        results["faithfulness"] = np.mean(faithfulness_scores) if faithfulness_scores else 0
        results["completeness"] = np.mean(completeness_scores) if completeness_scores else 0
    else:
        # Heuristic fallback: use evidence (not answer) for generation proxy
        rouge_scores = []
        for i, qa in enumerate(qa_pairs):
            retrieved_indices = retrieve(query_embeddings[i], index, top_k=3)
            context = " ".join(
                [chunk_texts[idx] for idx in retrieved_indices if 0 <= idx < len(chunk_texts)]
            )
            ev = qa.get("evidence", "")
            extractive_answer = get_most_relevant_sentence(context, qa["question"], ev)
            rouge = compute_rouge_l(extractive_answer, ev)
            rouge_scores.append(rouge)
        results["rouge_l"] = np.mean(rouge_scores) if rouge_scores else 0

    return results


def validate_results(df: pd.DataFrame) -> list:
    """Run quality checks on results and return list of warnings."""
    warnings = []

    # Check 1: No strategy has Hit@5 = 0 across ALL corpora
    for strat in df["strategy"].unique():
        strat_df = df[df["strategy"] == strat]
        if "hit_rate_at_5" in strat_df.columns:
            if strat_df["hit_rate_at_5"].mean() == 0:
                warnings.append(f"WARNING: {strat} has Hit@5 = 0 across all corpora")

    # Check 2: Hit@1 <= Hit@3 <= Hit@5
    for _, row in df.iterrows():
        h1 = row.get("hit_rate_at_1", 0)
        h3 = row.get("hit_rate_at_3", 0)
        h5 = row.get("hit_rate_at_5", 0)
        if h1 > h3 + 0.01 or h3 > h5 + 0.01:
            warnings.append(
                f"WARNING: Hit rate ordering violated for {row.get('strategy')}/"
                f"{row.get('corpus_id')}: H@1={h1:.3f}, H@3={h3:.3f}, H@5={h5:.3f}"
            )

    # Check 3: MRR in [0, 1]
    if "mrr" in df.columns:
        bad_mrr = df[(df["mrr"] < 0) | (df["mrr"] > 1)]
        if len(bad_mrr) > 0:
            warnings.append(f"WARNING: {len(bad_mrr)} rows have MRR outside [0, 1]")

    # Check 4: fixed_size chunking time < semantic chunking time
    fixed_time = df[df["strategy"] == "fixed_size"]["chunking_time_ms"].mean()
    semantic_time = df[df["strategy"] == "semantic"]["chunking_time_ms"].mean()
    if fixed_time > semantic_time and semantic_time > 0:
        warnings.append(
            f"WARNING: fixed_size ({fixed_time:.1f}ms) slower than semantic ({semantic_time:.1f}ms)"
        )

    # Check 5: Duplicate rows
    dup_cols = ["corpus_id", "strategy", "chunk_size", "embedding_model"]
    dups = df.duplicated(subset=dup_cols)
    if dups.any():
        warnings.append(f"WARNING: {dups.sum()} duplicate rows found")

    return warnings


def main():
    start_time = time.time()

    # Ensure reproducibility — seed all RNGs used by dependencies
    np.random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass

    # Create output dirs
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Check Ollama
    use_llm_judge = USE_LLM_JUDGE and is_ollama_available()
    if use_llm_judge:
        logger.info("Ollama available. Using LLM judge for generation metrics.")
    else:
        logger.info("Ollama NOT available. Using heuristic generation metrics.")

    # Load corpora
    corpora = load_all_corpora()
    logger.info(f"Loaded {len(corpora)} corpora.")
    if len(corpora) < MIN_CORPORA:
        logger.error(f"Only {len(corpora)} corpora. Minimum {MIN_CORPORA} required.")
        logger.error("Run scripts/download_documents.py first.")
        sys.exit(1)

    # Load QA pairs
    all_qa = {}
    for corpus_id in corpora:
        try:
            qa_data = load_qa_pairs(corpus_id)
            all_qa[corpus_id] = qa_data["qa_pairs"]
            logger.info(f"  {corpus_id}: {len(qa_data['qa_pairs'])} QA pairs")
        except FileNotFoundError:
            logger.warning(f"  {corpus_id}: No QA pairs found. Skipping.")

    if not all_qa:
        logger.error("No QA pairs found. Run scripts/generate_qa_pairs.py first.")
        sys.exit(1)

    # Filter to corpora with QA pairs
    active_corpora = {k: v for k, v in corpora.items() if k in all_qa}
    logger.info(f"Active corpora (with QA pairs): {len(active_corpora)}")

    # Determine active embedding models
    active_models = []
    for model_name in EMBEDDING_MODELS:
        try:
            from src.embedder import _get_model
            _get_model(model_name)
            active_models.append(model_name)
            logger.info(f"  Model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"  Model FAILED: {model_name}: {e}")

    if len(active_models) < MIN_EMBEDDING_MODELS:
        logger.error(f"Only {len(active_models)} models. Minimum {MIN_EMBEDDING_MODELS} required.")
        sys.exit(1)

    # Calculate total configurations
    total_configs = (
        len(CHUNKING_STRATEGIES) * len(CHUNK_SIZES) * len(active_models) * len(active_corpora)
    )
    logger.info(f"\nRunning {total_configs} configurations:")
    logger.info(f"  {len(CHUNKING_STRATEGIES)} strategies x {len(CHUNK_SIZES)} sizes x "
                f"{len(active_models)} models x {len(active_corpora)} corpora")

    # NOTE: For accurate embedding_time_ms, clear cached embeddings before a
    # fresh run:  rm -f data/embeddings/*.npy data/embeddings/*.hash
    # Cached embeddings load in <1ms, which would skew latency measurements.

    # Checkpoint resume: load previously completed (non-error) results
    checkpoint_path = os.path.join(RAW_RESULTS_DIR, "all_results_checkpoint.csv")
    all_results = []
    completed_keys = set()
    if os.path.exists(checkpoint_path):
        prev_df = pd.read_csv(checkpoint_path)
        for _, row in prev_df.iterrows():
            row_dict = row.to_dict()
            key = (row_dict["corpus_id"], row_dict["strategy"],
                   int(row_dict["chunk_size"]), row_dict["embedding_model"])
            # Keep successful results; retry errored ones
            if pd.isna(row_dict.get("error")):
                all_results.append(row_dict)
                completed_keys.add(key)
        logger.info(f"Resumed {len(completed_keys)} successful configs from checkpoint "
                    f"(skipped {len(prev_df) - len(completed_keys)} errored configs for retry)")

    # Run experiment grid
    config_num = 0
    skipped = 0
    errors = 0

    for corpus_id in tqdm(list(active_corpora.keys()), desc="Corpora"):
        doc_text = active_corpora[corpus_id]
        qa_pairs = all_qa[corpus_id]

        for strategy in CHUNKING_STRATEGIES:
            for chunk_size in CHUNK_SIZES:
                for model_name in active_models:
                    config_num += 1
                    key = (corpus_id, strategy, chunk_size, model_name)
                    if key in completed_keys:
                        skipped += 1
                        continue

                    try:
                        result = run_single_configuration(
                            corpus_id, strategy, chunk_size, model_name,
                            qa_pairs, doc_text, use_llm_judge,
                        )
                        all_results.append(result)
                    except Exception as e:
                        errors += 1
                        logger.error(
                            f"ERROR [{config_num}/{total_configs}] "
                            f"{corpus_id}/{strategy}/{chunk_size}/{model_name}: {e}"
                        )
                        all_results.append({
                            "corpus_id": corpus_id,
                            "domain": CORPORA[corpus_id]["domain"],
                            "strategy": strategy,
                            "chunk_size": chunk_size,
                            "embedding_model": model_name,
                            "error": str(e),
                        })

                    # Checkpoint
                    if (config_num - skipped) % CHECKPOINT_EVERY == 0 and (config_num - skipped) > 0:
                        pd.DataFrame(all_results).to_csv(
                            checkpoint_path, index=False,
                        )
                        logger.info(
                            f"  Checkpoint at {config_num}/{total_configs} "
                            f"({skipped} skipped, {errors} errors)"
                        )

    # Save final results
    df = pd.DataFrame(all_results)
    final_path = os.path.join(RAW_RESULTS_DIR, "all_results_final.csv")
    df.to_csv(final_path, index=False)

    # Quality checks
    if "error" in df.columns:
        clean_df = df[df["error"].isna()]
    else:
        clean_df = df

    warnings = validate_results(clean_df) if len(clean_df) > 0 else []

    # Generate tables and figures
    logger.info("\nGenerating tables and figures...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
        from scripts.generate_paper_tables import generate_all
        generate_all(final_path)
    except Exception as e:
        logger.error(f"Failed to generate tables/figures: {e}")

    # Generate summary report
    elapsed = time.time() - start_time
    _generate_summary(df, clean_df, total_configs, errors, elapsed, warnings)

    logger.info(f"\nExperiment complete in {elapsed/3600:.1f} hours.")
    logger.info(f"Results: {final_path}")
    logger.info(f"Successful: {len(clean_df)}/{total_configs} ({len(clean_df)/total_configs*100:.1f}%)")


def _generate_summary(df, clean_df, total_configs, errors, elapsed, warnings):
    """Generate the SUMMARY.md report."""
    summary_path = os.path.join(os.path.dirname(__file__), "results", "SUMMARY.md")

    with open(summary_path, "w") as f:
        f.write("# TechChunkBench — Experiment Summary\n\n")
        f.write(f"## Overview\n")
        f.write(f"- Total configurations attempted: {total_configs}\n")
        f.write(f"- Successful configurations: {len(clean_df)}\n")
        f.write(f"- Failed configurations: {errors}\n")
        f.write(f"- Success rate: {len(clean_df)/max(total_configs,1)*100:.1f}%\n")
        f.write(f"- Total runtime: {elapsed/3600:.2f} hours\n\n")

        if len(clean_df) > 0:
            f.write("## Key Findings\n\n")

            # Top strategies by MRR
            if "mrr" in clean_df.columns:
                top_mrr = clean_df.groupby("strategy")["mrr"].mean().sort_values(ascending=False)
                f.write("### Top 3 Strategies by MRR\n")
                for i, (strat, val) in enumerate(top_mrr.head(3).items()):
                    f.write(f"{i+1}. **{strat}**: MRR = {val:.4f}\n")
                f.write("\n")

            # Efficiency
            if "chunking_time_ms" in clean_df.columns:
                eff = clean_df.groupby("strategy")["chunking_time_ms"].mean().sort_values()
                f.write("### Fastest Chunking Strategies\n")
                for strat, val in eff.head(3).items():
                    f.write(f"- **{strat}**: {val:.1f} ms\n")
                f.write("\n")

        if warnings:
            f.write("## Quality Check Warnings\n")
            for w in warnings:
                f.write(f"- {w}\n")
            f.write("\n")

        # Errors
        if errors > 0 and "error" in df.columns:
            error_df = df[df["error"].notna()]
            f.write("## Errors\n")
            f.write(f"Total errors: {len(error_df)}\n\n")
            for _, row in error_df.head(20).iterrows():
                f.write(f"- {row.get('corpus_id')}/{row.get('strategy')}/"
                        f"{row.get('chunk_size')}/{row.get('embedding_model')}: "
                        f"{row.get('error', 'Unknown')}\n")

    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
