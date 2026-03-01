"""
Parallel experiment runner for TechChunkBench.

3-stage pipeline:
  Stage 1: Chunk all documents (CPU strategies in parallel, GPU strategies serial)
  Stage 2: Embed all chunks + queries (serial, one model at a time)
  Stage 3: Evaluate all configs (parallel across CPU cores)

Output is identical to run_all.py: results/raw/all_results_final.csv with the same
schema and a results/SUMMARY.md report.

Does NOT modify any existing files.
"""

import gc
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
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
    EMBEDDINGS_DIR,
    RAW_RESULTS_DIR,
    AGGREGATED_DIR,
    FIGURES_DIR,
    MIN_CORPORA,
    MIN_EMBEDDING_MODELS,
)
from src.chunkers import get_chunker
from src.chunkers.base import BaseChunker
from src.document_loader import load_all_corpora
from src.embedder import embed_chunks, embed_queries, get_embed_fn, _get_model, _MODEL_CACHE
from src.retriever import build_faiss_index, retrieve
from src.evaluator import (
    compute_hit,
    compute_mrr,
    compute_ndcg,
    compute_context_precision,
    get_most_relevant_sentence,
    compute_rouge_l,
)
from src.qa_generator import load_qa_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("techchunkbench_parallel_run.log"),
    ],
)
logger = logging.getLogger(__name__)

# CPU-only chunking strategies (no embedding model needed)
CPU_STRATEGIES = ["fixed_size", "fixed_overlap", "sentence_based", "recursive", "structure_aware"]
# GPU-dependent chunking strategies (need embed_fn from a specific model)
GPU_STRATEGIES = ["semantic", "hybrid"]

# Cache directories
CHUNK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chunk_cache")


# ---------------------------------------------------------------------------
# Utility: cache key helpers
# ---------------------------------------------------------------------------

def _chunk_cache_path(corpus_id, strategy, chunk_size, model_name=None):
    """Build path for a cached chunk result pickle."""
    if model_name:
        safe_model = model_name.replace("/", "_")
        fname = f"{corpus_id}_{strategy}_{chunk_size}_{safe_model}.pkl"
    else:
        fname = f"{corpus_id}_{strategy}_{chunk_size}.pkl"
    return os.path.join(CHUNK_CACHE_DIR, fname)


def _chunk_time_path(cache_path):
    """Timing sidecar for a chunk cache file."""
    return cache_path + ".time"


def _embed_time_path(corpus_id, strategy, chunk_size, model_name):
    """Timing sidecar for an embedding cache file."""
    safe_model = model_name.replace("/", "_")
    fname = f"{corpus_id}_{strategy}_{chunk_size}_{safe_model}.npy.time"
    return os.path.join(EMBEDDINGS_DIR, fname)


def _query_embed_cache_path(corpus_id, model_name):
    """Cache path for embedded queries."""
    safe_model = model_name.replace("/", "_")
    fname = f"queries_{corpus_id}_{safe_model}.npy"
    return os.path.join(EMBEDDINGS_DIR, fname)


def _save_time(path, ms):
    """Write timing value (ms) to a sidecar file."""
    with open(path, "w") as f:
        f.write(f"{ms}")


def _load_time(path):
    """Read timing value (ms) from a sidecar file, return 0.0 if missing."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Stage manifest helpers (for checkpoint/resume)
# ---------------------------------------------------------------------------

def _stage_manifest_path(stage_num):
    if stage_num == 1:
        return os.path.join(CHUNK_CACHE_DIR, "_stage1_complete.json")
    elif stage_num == 2:
        return os.path.join(EMBEDDINGS_DIR, "_stage2_complete.json")
    else:
        return os.path.join(RAW_RESULTS_DIR, "_stage3_complete.json")


def _is_stage_complete(stage_num):
    path = _stage_manifest_path(stage_num)
    return os.path.exists(path)


def _write_stage_manifest(stage_num, completed_keys):
    path = _stage_manifest_path(stage_num)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "completed": completed_keys,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 1 worker: chunk a single (corpus, strategy, size) config
# ---------------------------------------------------------------------------

def _chunk_worker(args):
    """Stage 1 worker for CPU-only strategies.

    Args is a tuple: (corpus_id, doc_text, strategy_name, chunk_size, cache_path)

    Returns (cache_path, chunk_data_dict, time_ms) or (cache_path, None, error_str)
    on failure.
    """
    corpus_id, doc_text, strategy_name, chunk_size, cache_path = args
    try:
        chunker = get_chunker(strategy_name, target_size=chunk_size)
        start = time.perf_counter()
        chunks = chunker.chunk(doc_text, corpus_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        token_counts = [BaseChunker.count_tokens(c.text) for c in chunks]
        chunk_data = {
            "chunks": [(c.text, c.metadata) for c in chunks],
            "num_chunks": len(chunks),
            "mean_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
            "std_tokens": float(np.std(token_counts)) if token_counts else 0.0,
        }
        return (cache_path, chunk_data, elapsed_ms)
    except Exception as e:
        return (cache_path, None, str(e))


# ---------------------------------------------------------------------------
# Stage 3 worker: evaluate a single config
# ---------------------------------------------------------------------------

def _eval_worker(args):
    """Stage 3 worker: evaluate a single (corpus, strategy, size, model) config.

    Loads chunk data from cache_path (instead of receiving it in-memory) to
    avoid serializing all chunk data through the multiprocessing pipe.

    Returns a result dict with the same schema as run_single_configuration().
    """
    (corpus_id, strategy, chunk_size, model_name, domain,
     cache_path, chunk_emb_path, query_emb_path, qa_pairs,
     chunk_time_ms, embed_time_ms, use_llm_judge) = args

    results = {
        "corpus_id": corpus_id,
        "domain": domain,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "embedding_model": model_name,
        "chunking_time_ms": chunk_time_ms,
        "embedding_time_ms": embed_time_ms,
    }

    try:
        # Load chunk data from disk (S7: avoids main-process memory bloat)
        with open(cache_path, "rb") as f:
            chunk_data = pickle.load(f)

        results["num_chunks"] = chunk_data["num_chunks"]
        results["mean_chunk_tokens"] = chunk_data["mean_tokens"]
        results["std_chunk_tokens"] = chunk_data["std_tokens"]

        chunk_texts = [c[0] for c in chunk_data["chunks"]]

        if not chunk_texts:
            results["error"] = "No chunks produced"
            return results

        # Load pre-computed embeddings
        chunk_embeddings = np.load(chunk_emb_path)
        query_embeddings = np.load(query_emb_path)

        # Build FAISS index
        index = build_faiss_index(chunk_embeddings)

        # Retrieve and evaluate
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

            ndcg = compute_ndcg(retrieved_texts[:5], evidence, total_relevant=1)
            all_ndcg.append(ndcg)

            precision = compute_context_precision(retrieved_texts[:5], evidence)
            all_precision.append(precision)

        # Aggregate retrieval metrics
        for k in TOP_K_VALUES:
            results[f"hit_rate_at_{k}"] = float(np.mean(all_hit_rates[k])) if all_hit_rates[k] else 0
        results["mrr"] = float(np.mean(all_mrr)) if all_mrr else 0
        results["ndcg_at_5"] = float(np.mean(all_ndcg)) if all_ndcg else 0
        results["context_precision"] = float(np.mean(all_precision)) if all_precision else 0
        results["mean_retrieval_latency_ms"] = float(np.mean(retrieval_latencies)) if retrieval_latencies else 0

        # Generation metrics (heuristic ROUGE-L)
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
        results["rouge_l"] = float(np.mean(rouge_scores)) if rouge_scores else 0

    except Exception as e:
        results["error"] = f"Worker exception: {e}"

    return results


# ---------------------------------------------------------------------------
# Stage 1: Chunk all documents
# ---------------------------------------------------------------------------

def run_stage1(active_corpora, active_models):
    """Chunk all documents. CPU strategies in parallel, GPU strategies serial."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Chunking all documents")
    logger.info("=" * 60)

    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)

    # --- 1a. CPU-only strategies (parallel) ---
    cpu_tasks = []
    cpu_tasks_skipped = 0
    for corpus_id, doc_text in active_corpora.items():
        for strategy in CPU_STRATEGIES:
            for chunk_size in CHUNK_SIZES:
                cache_path = _chunk_cache_path(corpus_id, strategy, chunk_size)
                if os.path.exists(cache_path):
                    cpu_tasks_skipped += 1
                    continue
                cpu_tasks.append((corpus_id, doc_text, strategy, chunk_size, cache_path))

    logger.info(f"Stage 1a: {len(cpu_tasks)} CPU chunking tasks "
                f"({cpu_tasks_skipped} already cached)")

    if cpu_tasks:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"  Using {n_workers} worker processes")
        with multiprocessing.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_chunk_worker, cpu_tasks),
                total=len(cpu_tasks),
                desc="  CPU chunking",
            ))

        errors = 0
        for cache_path, chunk_data, time_or_err in results:
            if chunk_data is None:
                errors += 1
                logger.error(f"  Chunk error: {cache_path}: {time_or_err}")
                continue
            with open(cache_path, "wb") as f:
                pickle.dump(chunk_data, f)
            _save_time(_chunk_time_path(cache_path), time_or_err)

        if errors:
            logger.warning(f"  {errors} CPU chunking errors")

    # --- 1b. GPU-dependent strategies (serial, grouped by model) ---
    gpu_total = 0
    gpu_skipped = 0
    gpu_errors = 0

    for model_name in active_models:
        gpu_tasks_for_model = []
        for corpus_id in active_corpora:
            for strategy in GPU_STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    cache_path = _chunk_cache_path(corpus_id, strategy, chunk_size, model_name)
                    if os.path.exists(cache_path):
                        gpu_skipped += 1
                        continue
                    gpu_tasks_for_model.append((corpus_id, strategy, chunk_size, cache_path))

        if not gpu_tasks_for_model:
            continue

        gpu_total += len(gpu_tasks_for_model)
        logger.info(f"Stage 1b: {len(gpu_tasks_for_model)} GPU chunking tasks for {model_name}")

        # Load model once for this batch
        embed_fn = get_embed_fn(model_name)

        for corpus_id, strategy, chunk_size, cache_path in tqdm(
            gpu_tasks_for_model, desc=f"  GPU chunking ({model_name.split('/')[-1]})"
        ):
            try:
                chunker = get_chunker(strategy, target_size=chunk_size, embed_fn=embed_fn)
                doc_text = active_corpora[corpus_id]

                start = time.perf_counter()
                chunks = chunker.chunk(doc_text, corpus_id)
                elapsed_ms = (time.perf_counter() - start) * 1000

                token_counts = [BaseChunker.count_tokens(c.text) for c in chunks]
                chunk_data = {
                    "chunks": [(c.text, c.metadata) for c in chunks],
                    "num_chunks": len(chunks),
                    "mean_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
                    "std_tokens": float(np.std(token_counts)) if token_counts else 0.0,
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(chunk_data, f)
                _save_time(_chunk_time_path(cache_path), elapsed_ms)
            except Exception as e:
                gpu_errors += 1
                logger.error(f"  GPU chunk error {corpus_id}/{strategy}/{chunk_size}/{model_name}: {e}")

        # Unload model after finishing this model's GPU chunking
        _MODEL_CACHE.pop(model_name, None)
        gc.collect()

    logger.info(f"Stage 1b: {gpu_total} GPU tasks processed, {gpu_skipped} cached, "
                f"{gpu_errors} errors")
    logger.info("Stage 1 complete.")


# ---------------------------------------------------------------------------
# Stage 2: Embed all chunks + queries
# ---------------------------------------------------------------------------

def run_stage2(active_corpora, all_qa, active_models):
    """Embed all chunk sets and query sets, one model at a time."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Embedding all chunks and queries")
    logger.info("=" * 60)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    for model_name in active_models:
        logger.info(f"Loading model: {model_name}")
        _get_model(model_name)  # Pre-load into cache

        # --- Embed chunk sets ---
        embed_tasks = []

        # CPU strategies: same chunks for all models
        for corpus_id in active_corpora:
            for strategy in CPU_STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    embed_tasks.append((corpus_id, strategy, chunk_size, None))

        # GPU strategies: chunks depend on model
        for corpus_id in active_corpora:
            for strategy in GPU_STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    embed_tasks.append((corpus_id, strategy, chunk_size, model_name))

        embedded = 0
        skipped = 0
        errors = 0

        for corpus_id, strategy, chunk_size, chunk_model in tqdm(
            embed_tasks, desc=f"  Embedding ({model_name.split('/')[-1]})"
        ):
            # Load chunk data
            cache_path = _chunk_cache_path(corpus_id, strategy, chunk_size, chunk_model)
            if not os.path.exists(cache_path):
                errors += 1
                logger.warning(f"  Missing chunk cache: {cache_path}")
                continue

            with open(cache_path, "rb") as f:
                chunk_data = pickle.load(f)

            chunk_texts = [c[0] for c in chunk_data["chunks"]]
            if not chunk_texts:
                continue

            # Check if embedding is cached AND content hash matches current chunks.
            # (S1 fix: .time sidecar alone is not sufficient — chunks may have
            # changed since the embedding was computed.)
            safe_model = model_name.replace("/", "_")
            emb_path = os.path.join(
                EMBEDDINGS_DIR,
                f"{corpus_id}_{strategy}_{chunk_size}_{safe_model}.npy",
            )
            hash_path = emb_path + ".hash"
            content_hash = hashlib.md5("\0".join(chunk_texts).encode()).hexdigest()[:12]
            time_path = _embed_time_path(corpus_id, strategy, chunk_size, model_name)

            if (os.path.exists(emb_path) and os.path.exists(hash_path)
                    and os.path.exists(time_path)):
                with open(hash_path) as hf:
                    if hf.read().strip() == content_hash:
                        skipped += 1
                        continue

            start = time.perf_counter()
            embed_chunks(
                chunk_texts, model_name,
                corpus_id=corpus_id, strategy=strategy, size=chunk_size,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            _save_time(time_path, elapsed_ms)
            embedded += 1

        logger.info(f"  Embedded {embedded} chunk sets, skipped {skipped}, errors {errors}")

        # --- Embed query sets ---
        query_embedded = 0
        for corpus_id in active_corpora:
            if corpus_id not in all_qa:
                continue

            query_cache = _query_embed_cache_path(corpus_id, model_name)
            query_texts = [qa["question"] for qa in all_qa[corpus_id]]
            query_hash = hashlib.md5("\0".join(query_texts).encode()).hexdigest()[:12]
            query_hash_path = query_cache + ".hash"

            # S3 fix: validate query cache against current QA content hash
            if os.path.exists(query_cache) and os.path.exists(query_hash_path):
                with open(query_hash_path) as hf:
                    if hf.read().strip() == query_hash:
                        continue

            query_embs = embed_queries(query_texts, model_name)
            np.save(query_cache, query_embs)
            with open(query_hash_path, "w") as hf:
                hf.write(query_hash)
            query_embedded += 1

        logger.info(f"  Embedded {query_embedded} query sets for {model_name}")

        # Unload model to free memory before loading the next one
        _MODEL_CACHE.pop(model_name, None)
        gc.collect()

    logger.info("Stage 2 complete.")


# ---------------------------------------------------------------------------
# Stage 3: Evaluate all configs
# ---------------------------------------------------------------------------

def run_stage3(active_corpora, all_qa, active_models, use_llm_judge):
    """Evaluate all 1,512 configs in parallel and produce the final CSV."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Evaluating all configurations")
    logger.info("=" * 60)

    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

    # Build eval task list
    eval_tasks = []
    task_errors = []

    for corpus_id in active_corpora:
        if corpus_id not in all_qa:
            continue
        domain = CORPORA[corpus_id]["domain"]
        qa_pairs = all_qa[corpus_id]

        for strategy in CHUNKING_STRATEGIES:
            for chunk_size in CHUNK_SIZES:
                for model_name in active_models:
                    # Determine chunk cache path
                    if strategy in GPU_STRATEGIES:
                        cache_path = _chunk_cache_path(corpus_id, strategy, chunk_size, model_name)
                    else:
                        cache_path = _chunk_cache_path(corpus_id, strategy, chunk_size)

                    # Check chunk cache exists (worker loads it from disk — S7)
                    if not os.path.exists(cache_path):
                        task_errors.append({
                            "corpus_id": corpus_id,
                            "domain": domain,
                            "strategy": strategy,
                            "chunk_size": chunk_size,
                            "embedding_model": model_name,
                            "error": f"Missing chunk cache: {cache_path}",
                        })
                        continue

                    # Chunk timing
                    chunk_time_ms = _load_time(_chunk_time_path(cache_path))

                    # Embedding timing
                    embed_time_ms = _load_time(
                        _embed_time_path(corpus_id, strategy, chunk_size, model_name)
                    )

                    # Embedding file paths (for loading in worker)
                    safe_model = model_name.replace("/", "_")
                    chunk_emb_path = os.path.join(
                        EMBEDDINGS_DIR,
                        f"{corpus_id}_{strategy}_{chunk_size}_{safe_model}.npy",
                    )
                    query_emb_path = _query_embed_cache_path(corpus_id, model_name)

                    if not os.path.exists(chunk_emb_path):
                        task_errors.append({
                            "corpus_id": corpus_id,
                            "domain": domain,
                            "strategy": strategy,
                            "chunk_size": chunk_size,
                            "embedding_model": model_name,
                            "error": f"Missing chunk embeddings: {chunk_emb_path}",
                        })
                        continue

                    if not os.path.exists(query_emb_path):
                        task_errors.append({
                            "corpus_id": corpus_id,
                            "domain": domain,
                            "strategy": strategy,
                            "chunk_size": chunk_size,
                            "embedding_model": model_name,
                            "error": f"Missing query embeddings: {query_emb_path}",
                        })
                        continue

                    eval_tasks.append((
                        corpus_id, strategy, chunk_size, model_name, domain,
                        cache_path, chunk_emb_path, query_emb_path, qa_pairs,
                        chunk_time_ms, embed_time_ms, use_llm_judge,
                    ))

    logger.info(f"Stage 3: {len(eval_tasks)} evaluation tasks, {len(task_errors)} pre-errors")

    all_results = list(task_errors)  # Start with any error entries

    if eval_tasks:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"  Using {n_workers} worker processes")
        with multiprocessing.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_eval_worker, eval_tasks),
                total=len(eval_tasks),
                desc="  Evaluating",
            ))
        all_results.extend(results)

    # Count errors
    errors = sum(1 for r in all_results if "error" in r and r.get("error"))

    logger.info(f"Stage 3 complete: {len(all_results)} results, {errors} errors")
    return all_results, errors


# ---------------------------------------------------------------------------
# Validation and summary (reused from run_all.py logic)
# ---------------------------------------------------------------------------

def validate_results(df):
    """Run quality checks on results. Same logic as run_all.py."""
    warnings = []

    for strat in df["strategy"].unique():
        strat_df = df[df["strategy"] == strat]
        if "hit_rate_at_5" in strat_df.columns:
            if strat_df["hit_rate_at_5"].mean() == 0:
                warnings.append(f"WARNING: {strat} has Hit@5 = 0 across all corpora")

    for _, row in df.iterrows():
        h1 = row.get("hit_rate_at_1", 0)
        h3 = row.get("hit_rate_at_3", 0)
        h5 = row.get("hit_rate_at_5", 0)
        if h1 > h3 + 0.01 or h3 > h5 + 0.01:
            warnings.append(
                f"WARNING: Hit rate ordering violated for {row.get('strategy')}/"
                f"{row.get('corpus_id')}: H@1={h1:.3f}, H@3={h3:.3f}, H@5={h5:.3f}"
            )

    if "mrr" in df.columns:
        bad_mrr = df[(df["mrr"] < 0) | (df["mrr"] > 1)]
        if len(bad_mrr) > 0:
            warnings.append(f"WARNING: {len(bad_mrr)} rows have MRR outside [0, 1]")

    fixed_time = df[df["strategy"] == "fixed_size"]["chunking_time_ms"].mean()
    semantic_time = df[df["strategy"] == "semantic"]["chunking_time_ms"].mean()
    if fixed_time > semantic_time and semantic_time > 0:
        warnings.append(
            f"WARNING: fixed_size ({fixed_time:.1f}ms) slower than semantic ({semantic_time:.1f}ms)"
        )

    dup_cols = ["corpus_id", "strategy", "chunk_size", "embedding_model"]
    dups = df.duplicated(subset=dup_cols)
    if dups.any():
        warnings.append(f"WARNING: {dups.sum()} duplicate rows found")

    return warnings


def _generate_summary(df, clean_df, total_configs, errors, elapsed, warnings):
    """Generate the SUMMARY.md report. Same logic as run_all.py."""
    summary_path = os.path.join(os.path.dirname(__file__), "results", "SUMMARY.md")

    with open(summary_path, "w") as f:
        f.write("# TechChunkBench \u2014 Experiment Summary\n\n")
        f.write("## Overview\n")
        f.write(f"- Total configurations attempted: {total_configs}\n")
        f.write(f"- Successful configurations: {len(clean_df)}\n")
        f.write(f"- Failed configurations: {errors}\n")
        f.write(f"- Success rate: {len(clean_df)/max(total_configs,1)*100:.1f}%\n")
        f.write(f"- Total runtime: {elapsed/3600:.2f} hours\n\n")

        if len(clean_df) > 0:
            f.write("## Key Findings\n\n")

            if "mrr" in clean_df.columns:
                top_mrr = clean_df.groupby("strategy")["mrr"].mean().sort_values(ascending=False)
                f.write("### Top 3 Strategies by MRR\n")
                for i, (strat, val) in enumerate(top_mrr.head(3).items()):
                    f.write(f"{i+1}. **{strat}**: MRR = {val:.4f}\n")
                f.write("\n")

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

        if errors > 0 and "error" in df.columns:
            error_df = df[df["error"].notna()]
            f.write("## Errors\n")
            f.write(f"Total errors: {len(error_df)}\n\n")
            for _, row in error_df.head(20).iterrows():
                f.write(f"- {row.get('corpus_id')}/{row.get('strategy')}/"
                        f"{row.get('chunk_size')}/{row.get('embedding_model')}: "
                        f"{row.get('error', 'Unknown')}\n")

    logger.info(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()

    # Ensure reproducibility
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
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    # LLM judge is not supported in the parallel runner (Ollama can't safely run
    # in worker processes). Always use ROUGE-L heuristic.
    use_llm_judge = False
    if USE_LLM_JUDGE:
        logger.warning("USE_LLM_JUDGE=True in config, but parallel runner does not "
                       "support LLM judge. Falling back to ROUGE-L heuristic.")
    logger.info("Using heuristic generation metrics (ROUGE-L).")

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

    # Validate embedding models (load and immediately unload to save memory)
    active_models = []
    for model_name in EMBEDDING_MODELS:
        try:
            _get_model(model_name)
            active_models.append(model_name)
            logger.info(f"  Model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"  Model FAILED: {model_name}: {e}")
    # Unload all models after validation — Stage 1b and 2 will reload as needed
    _MODEL_CACHE.clear()
    gc.collect()

    if len(active_models) < MIN_EMBEDDING_MODELS:
        logger.error(f"Only {len(active_models)} models. Minimum {MIN_EMBEDDING_MODELS} required.")
        sys.exit(1)

    total_configs = (
        len(CHUNKING_STRATEGIES) * len(CHUNK_SIZES) * len(active_models) * len(active_corpora)
    )
    logger.info(f"\nTotal configurations: {total_configs}")
    logger.info(f"  {len(CHUNKING_STRATEGIES)} strategies x {len(CHUNK_SIZES)} sizes x "
                f"{len(active_models)} models x {len(active_corpora)} corpora")

    # -----------------------------------------------------------------------
    # Stage 1: Chunking
    # -----------------------------------------------------------------------
    if _is_stage_complete(1):
        logger.info("Stage 1 already complete (manifest found). Skipping.")
    else:
        run_stage1(active_corpora, active_models)
        # Build completed keys list for manifest
        completed = []
        expected = 0
        for corpus_id in active_corpora:
            for strategy in CPU_STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    expected += 1
                    p = _chunk_cache_path(corpus_id, strategy, chunk_size)
                    if os.path.exists(p):
                        completed.append(f"{corpus_id}_{strategy}_{chunk_size}")
            for strategy in GPU_STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    for model_name in active_models:
                        expected += 1
                        p = _chunk_cache_path(corpus_id, strategy, chunk_size, model_name)
                        if os.path.exists(p):
                            safe = model_name.replace("/", "_")
                            completed.append(f"{corpus_id}_{strategy}_{chunk_size}_{safe}")
        # S4 fix: only write manifest if all tasks completed successfully
        if len(completed) < expected:
            logger.warning(f"Stage 1 incomplete: {len(completed)}/{expected} tasks. "
                           f"Manifest NOT written — stage will re-run on next invocation.")
        else:
            _write_stage_manifest(1, completed)

    # -----------------------------------------------------------------------
    # Stage 2: Embedding
    # -----------------------------------------------------------------------
    if _is_stage_complete(2):
        logger.info("Stage 2 already complete (manifest found). Skipping.")
    else:
        run_stage2(active_corpora, all_qa, active_models)
        completed = []
        expected = 0
        for corpus_id in active_corpora:
            for strategy in CHUNKING_STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    for model_name in active_models:
                        expected += 1
                        tp = _embed_time_path(corpus_id, strategy, chunk_size, model_name)
                        if os.path.exists(tp):
                            safe = model_name.replace("/", "_")
                            completed.append(
                                f"{corpus_id}_{strategy}_{chunk_size}_{safe}"
                            )
        # S4 fix: only write manifest if all tasks completed successfully
        if len(completed) < expected:
            logger.warning(f"Stage 2 incomplete: {len(completed)}/{expected} tasks. "
                           f"Manifest NOT written — stage will re-run on next invocation.")
        else:
            _write_stage_manifest(2, completed)

    # -----------------------------------------------------------------------
    # Stage 3: Evaluation
    # -----------------------------------------------------------------------
    if _is_stage_complete(3):
        logger.info("Stage 3 already complete (manifest found). Skipping evaluation.")
        final_path = os.path.join(RAW_RESULTS_DIR, "all_results_final.csv")
        if os.path.exists(final_path):
            df = pd.read_csv(final_path)
            clean_df = df[df["error"].isna()] if "error" in df.columns else df
            elapsed = time.time() - start_time
            logger.info(f"\nExperiment already complete. Results: {final_path}")
            logger.info(f"Successful: {len(clean_df)}/{total_configs}")
            return
        # Manifest exists but CSV missing — re-run
        logger.warning("Stage 3 manifest exists but results CSV missing. Re-running.")

    all_results, errors = run_stage3(active_corpora, all_qa, active_models, use_llm_judge)

    # Save final results
    df = pd.DataFrame(all_results)
    final_path = os.path.join(RAW_RESULTS_DIR, "all_results_final.csv")
    df.to_csv(final_path, index=False)
    logger.info(f"Saved {len(df)} results to {final_path}")

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

    # Write stage 3 manifest
    _write_stage_manifest(3, [f"final_{len(all_results)}_results"])

    logger.info(f"\nExperiment complete in {elapsed/3600:.1f} hours.")
    logger.info(f"Results: {final_path}")
    logger.info(f"Successful: {len(clean_df)}/{total_configs} "
                f"({len(clean_df)/total_configs*100:.1f}%)")


if __name__ == "__main__":
    main()
