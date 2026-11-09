"""E1: Corpus-regime replication on structured technical documents.

Mirrors the conference factorial (Chauhan & Hegde, DATA 2026) on the
24-document TechChunkBench corpus:

    4 strategies (fixed_size, fixed_overlap, structure_aware, semantic)
  x 2 chunk sizes (256, 512)
  x 3 embedding models (MiniLM, BGE-large, nomic)
  x 24 corpora (8 domains)
  = 576 configurations

Every retrieval metric is computed under two relevance-labeling schemes on the
same retrieved lists:
  - fuzzy: original TechChunkBench heuristic (>=70% evidence-word overlap)
  - span:  evidence char-span coverage >= 50% in normalized document space

QA pairs whose evidence cannot be located in the source document are excluded
from both schemes (kept comparable); the exclusion count is recorded per corpus.

Resumable: appends one row per finished config to a checkpoint CSV via atomic
replace; completed configs are skipped on restart.
"""

import logging
import os
import time

import numpy as np
import pandas as pd

from config import CORPORA, RAW_RESULTS_DIR
from src.chunkers import get_chunker
from src.document_loader import load_all_corpora
from src.embedder import embed_chunks, embed_queries, get_embed_fn
from src.retriever import build_faiss_index, retrieve
from src.evaluator import (
    compute_hit,
    compute_mrr,
    compute_ndcg,
    compute_context_precision,
)
from src.qa_generator import load_qa_pairs
from src import span_eval

STRATEGIES = ["fixed_size", "fixed_overlap", "structure_aware", "semantic",
              "structure_noprefix"]
CHUNK_SIZES = [256, 512]
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
]
TOP_K_VALUES = [1, 3, 5]
CHECKPOINT = os.path.join(RAW_RESULTS_DIR, "e1_results_checkpoint.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("e1_run.log")],
)
logger = logging.getLogger(__name__)


def atomic_append(row: dict) -> None:
    """Append a result row via write-temp-then-rename so a power cut cannot
    corrupt the checkpoint."""
    df_new = pd.DataFrame([row])
    if os.path.exists(CHECKPOINT):
        df = pd.concat([pd.read_csv(CHECKPOINT), df_new], ignore_index=True)
    else:
        df = df_new
    tmp = CHECKPOINT + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, CHECKPOINT)


def completed_keys() -> set:
    if not os.path.exists(CHECKPOINT):
        return set()
    df = pd.read_csv(CHECKPOINT)
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    return {
        (r.corpus_id, r.strategy, int(r.chunk_size), r.embedding_model)
        for r in ok.itertuples()
    }


def prepare_qa(corpus_id: str, doc_text: str) -> tuple:
    """Load QA pairs and pre-locate evidence spans; drop unlocatable ones."""
    qa_pairs = load_qa_pairs(corpus_id)["qa_pairs"]
    doc_norm = span_eval.normalize(doc_text)
    kept, spans = [], []
    for qa in qa_pairs:
        ev = qa.get("evidence", "")
        span = span_eval.locate_evidence(doc_norm, ev) if ev else None
        if span is not None:
            kept.append(qa)
            spans.append(span)
    return kept, spans, doc_norm, len(qa_pairs) - len(kept)


def run_config(corpus_id, strategy, chunk_size, model_name, doc_text,
               qa_pairs, ev_spans, doc_norm) -> dict:
    row = {
        "corpus_id": corpus_id,
        "domain": CORPORA[corpus_id]["domain"],
        "strategy": strategy,
        "chunk_size": chunk_size,
        "embedding_model": model_name,
        "n_queries": len(qa_pairs),
    }

    t0 = time.perf_counter()
    if strategy == "semantic":
        chunker = get_chunker(strategy, target_size=chunk_size,
                              embed_fn=get_embed_fn(model_name))
    else:
        chunker = get_chunker(strategy, target_size=chunk_size)
    chunks = chunker.chunk(doc_text, corpus_id)
    row["chunking_time_ms"] = (time.perf_counter() - t0) * 1000
    row["num_chunks"] = len(chunks)
    if not chunks:
        row["error"] = "no chunks"
        return row

    chunk_texts = [c.text for c in chunks]
    heading_paths = [c.metadata.get("heading_path", "") for c in chunks]
    chunk_spans = span_eval.locate_chunks(doc_norm, chunk_texts, heading_paths)
    row["chunk_locate_rate"] = float(np.mean([s is not None for s in chunk_spans]))

    t0 = time.perf_counter()
    chunk_embs = embed_chunks(chunk_texts, model_name, corpus_id=corpus_id,
                              strategy=f"e1_{strategy}", size=chunk_size)
    row["embedding_time_ms"] = (time.perf_counter() - t0) * 1000

    index = build_faiss_index(chunk_embs)
    query_embs = embed_queries([qa["question"] for qa in qa_pairs], model_name)

    max_k = max(TOP_K_VALUES)
    acc = {f"{scheme}_{m}": [] for scheme in ("fuzzy", "span")
           for m in ["hit1", "hit3", "hit5", "mrr", "ndcg5", "cprec"]}

    for i, qa in enumerate(qa_pairs):
        idxs = retrieve(query_embs[i], index, top_k=max_k)
        idxs = [j for j in idxs if 0 <= j < len(chunk_texts)]
        texts = [chunk_texts[j] for j in idxs]
        spans = [chunk_spans[j] for j in idxs]
        ev = qa["evidence"]

        for k in TOP_K_VALUES:
            acc[f"fuzzy_hit{k}"].append(compute_hit(texts[:k], ev))
        acc["fuzzy_mrr"].append(compute_mrr(texts, ev))
        acc["fuzzy_ndcg5"].append(compute_ndcg(texts[:5], ev, total_relevant=1))
        acc["fuzzy_cprec"].append(compute_context_precision(texts[:5], ev))

        labels = span_eval.relevance_labels(spans, ev_spans[i])
        for k in TOP_K_VALUES:
            acc[f"span_hit{k}"].append(span_eval.span_hit(labels, k))
        acc["span_mrr"].append(span_eval.span_mrr(labels))
        acc["span_ndcg5"].append(span_eval.span_ndcg(labels))
        acc["span_cprec"].append(span_eval.span_context_precision(labels))

    for key, vals in acc.items():
        row[key] = float(np.mean(vals)) if vals else 0.0
    return row


def main():
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    corpora = load_all_corpora()
    done = completed_keys()
    logger.info(f"Resuming with {len(done)} configs already complete")

    total = len(STRATEGIES) * len(CHUNK_SIZES) * len(MODELS) * len(corpora)
    n = 0
    for model_name in MODELS:          # outer loop: load each model once
        for strategy in STRATEGIES:
            for chunk_size in CHUNK_SIZES:
                for corpus_id, doc_text in sorted(corpora.items()):
                    n += 1
                    key = (corpus_id, strategy, chunk_size, model_name)
                    if key in done:
                        continue
                    qa_pairs, ev_spans, doc_norm, dropped = prepare_qa(corpus_id, doc_text)
                    if not qa_pairs:
                        atomic_append({"corpus_id": corpus_id, "strategy": strategy,
                                       "chunk_size": chunk_size,
                                       "embedding_model": model_name,
                                       "error": "no locatable QA"})
                        continue
                    logger.info(f"[{n}/{total}] {corpus_id} | {strategy} | "
                                f"{chunk_size} | {model_name.split('/')[-1]} "
                                f"({dropped} QA dropped)")
                    try:
                        row = run_config(corpus_id, strategy, chunk_size,
                                         model_name, doc_text, qa_pairs,
                                         ev_spans, doc_norm)
                        row["qa_dropped_unlocatable"] = dropped
                    except Exception as e:
                        logger.exception(f"config failed: {key}")
                        row = {"corpus_id": corpus_id, "strategy": strategy,
                               "chunk_size": chunk_size,
                               "embedding_model": model_name, "error": str(e)}
                    atomic_append(row)

    logger.info("E1 complete.")
    df = pd.read_csv(CHECKPOINT)
    final = os.path.join(RAW_RESULTS_DIR, "e1_results_final.csv")
    df.to_csv(final, index=False)
    logger.info(f"Wrote {final} ({len(df)} rows)")


if __name__ == "__main__":
    main()
