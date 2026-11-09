"""E2: Very-long-document regime — LegalBench-RAG factorial.

Same factorial as E1 on the four LegalBench-RAG sub-benchmarks (CUAD, MAUD,
ContractNLI, PrivacyQA), which provide human-annotated exact character-span
gold labels over long legal contracts:

    4 strategies x 2 chunk sizes x 3 embedding models x 4 datasets = 96 configs

Retrieval uses one shared FAISS index per sub-benchmark (all its files),
mirroring the conference paper's multi-document setting. A retrieved chunk is
relevant iff it covers >= 50% of any gold snippet's characters for the query
(spans compared in whitespace-normalized space; snippet offsets are mapped
from original space exactly, no fuzzy matching anywhere).

Resumable via atomic per-config checkpoint rows, like run_e1.
"""

import hashlib
import logging
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from config import RAW_RESULTS_DIR, EMBEDDINGS_DIR
from src.chunkers import get_chunker
from src.embedder import embed_chunks, embed_queries, get_embed_fn
from src.retriever import build_faiss_index, retrieve
from src import span_eval

LEGAL_DATA = os.path.expanduser("~/chunk-ext/legalbench-data")
BENCHMARKS = ["privacy_qa", "contractnli", "cuad", "maud"]  # smallest first
STRATEGIES = ["fixed_size", "fixed_overlap", "structure_aware", "semantic",
              "structure_legal", "structure_legal_noprefix"]
CHUNK_SIZES = [256, 512]
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
]
TOP_K_VALUES = [1, 3, 5, 10, 20]
CHECKPOINT = os.path.join(RAW_RESULTS_DIR, "e2v2_results_checkpoint.csv")
SENT_CACHE_DIR = os.path.join(EMBEDDINGS_DIR, "sent_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("e2_run.log")],
)
logger = logging.getLogger(__name__)


def atomic_append(row: dict) -> None:
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
        (r.dataset, r.strategy, int(r.chunk_size), r.embedding_model)
        for r in ok.itertuples()
    }


def cached_embed_fn(model_name: str):
    """Disk-cached wrapper around get_embed_fn for the semantic chunker.

    The chunker embeds every sentence of a document; identical sentence lists
    recur across chunk sizes, so caching by content hash avoids recomputing
    the most expensive part of semantic chunking on this corpus.
    """
    inner = get_embed_fn(model_name)
    os.makedirs(SENT_CACHE_DIR, exist_ok=True)
    safe_model = model_name.replace("/", "_")

    def _embed(texts: List[str]) -> np.ndarray:
        key = hashlib.md5(("\0".join(texts) + safe_model).encode()).hexdigest()
        path = os.path.join(SENT_CACHE_DIR, f"{key}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape[0] == len(texts):
                return arr
        arr = inner(texts)
        tmp = path[:-4] + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
        return arr

    return _embed


def _resolve_corpus_path(fp: str) -> str:
    """Return a file_path form that exists on disk, handling Unicode
    normalization mismatches (benchmark JSON uses NFD, extracted files NFC)."""
    import unicodedata
    for cand in (fp, unicodedata.normalize("NFC", fp),
                 unicodedata.normalize("NFD", fp)):
        if os.path.exists(os.path.join(LEGAL_DATA, "corpus", cand)):
            return cand
    raise FileNotFoundError(fp)


def load_benchmark(name: str):
    import json
    with open(os.path.join(LEGAL_DATA, "benchmarks", f"{name}.json")) as f:
        tests = json.load(f)["tests"]
    files: Dict[str, str] = {}
    for t in tests:
        for s in t["snippets"]:
            fp = _resolve_corpus_path(s["file_path"])
            s["file_path"] = fp  # canonicalize so gold labels match dict keys
            if fp not in files:
                with open(os.path.join(LEGAL_DATA, "corpus", fp)) as fh:
                    files[fp] = fh.read()
    return tests, files


def run_config(dataset, tests, files, norm_cache, strategy, chunk_size, model_name):
    row = {
        "dataset": dataset,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "embedding_model": model_name,
        "n_queries": len(tests),
        "n_files": len(files),
    }

    # 1. Chunk every file; keep per-chunk (file, normalized span)
    t0 = time.perf_counter()
    if strategy == "semantic":
        chunker = get_chunker(strategy, target_size=chunk_size,
                              embed_fn=cached_embed_fn(model_name))
    else:
        chunker = get_chunker(strategy, target_size=chunk_size)

    chunk_texts: List[str] = []
    chunk_files: List[str] = []
    chunk_spans: List = []
    located = 0
    structured_files = 0
    for fp, text in sorted(files.items()):
        chunks = chunker.chunk(text, fp)
        if any(c.metadata.get("heading_path") for c in chunks):
            structured_files += 1
        norm_doc, _ = norm_cache[fp]
        texts = [c.text for c in chunks]
        paths = [c.metadata.get("heading_path", "") for c in chunks]
        spans = span_eval.locate_chunks(norm_doc, texts, paths)
        chunk_texts.extend(texts)
        chunk_files.extend([fp] * len(chunks))
        chunk_spans.extend(spans)
        located += sum(s is not None for s in spans)
    row["chunking_time_ms"] = (time.perf_counter() - t0) * 1000
    row["num_chunks"] = len(chunk_texts)
    row["chunk_locate_rate"] = located / len(chunk_texts) if chunk_texts else 0.0
    row["structured_files_frac"] = structured_files / len(files) if files else 0.0
    if not chunk_texts:
        row["error"] = "no chunks"
        return row

    # 2. Embed all chunks (disk-cached), one shared index per sub-benchmark
    t0 = time.perf_counter()
    embs = embed_chunks(chunk_texts, model_name, corpus_id=f"e2_{dataset}",
                        strategy=strategy, size=chunk_size)
    row["embedding_time_ms"] = (time.perf_counter() - t0) * 1000
    index = build_faiss_index(embs)

    query_embs = embed_queries([t["query"] for t in tests], model_name)

    # 3. Retrieve and evaluate against gold snippet spans
    max_k = max(TOP_K_VALUES)
    acc = {m: [] for m in ["hit1", "hit3", "hit5", "hit10", "hit20", "mrr",
                            "ndcg5", "cprec", "dochit5",
                            "recall5", "recall10", "recall20"]}
    for i, t in enumerate(tests):
        idxs = retrieve(query_embs[i], index, top_k=max_k)
        idxs = [j for j in idxs if 0 <= j < len(chunk_texts)]

        gold = []  # (file, normalized span) per snippet
        for s in t["snippets"]:
            _, index_map = norm_cache[s["file_path"]]
            gold.append((s["file_path"],
                         span_eval.map_span(index_map, s["span"][0], s["span"][1],
                                            files[s["file_path"]])))

        # Track WHICH gold snippet each retrieved chunk covers (-1 = none), so
        # NDCG can credit distinct snippets separately while still counting a
        # snippet covered by several overlapping chunks only once.
        gold_hits = []
        for j in idxs:
            hit = -1
            for gi, (gf, gspan) in enumerate(gold):
                if chunk_files[j] == gf and span_eval.span_overlap_fraction(
                        chunk_spans[j], gspan) >= span_eval.OVERLAP_THRESHOLD:
                    hit = gi
                    break
            gold_hits.append(hit)
        labels = [1 if g >= 0 else 0 for g in gold_hits]

        for k in TOP_K_VALUES:
            acc[f"hit{k}"].append(span_eval.span_hit(labels, k))
        acc["mrr"].append(span_eval.span_mrr(labels))
        acc["ndcg5"].append(span_eval.span_ndcg_multi(
            gold_hits, total_relevant=min(len(gold), 5)))
        acc["cprec"].append(span_eval.span_context_precision(labels))
        for k in (5, 10, 20):
            covered = {g for g in gold_hits[:k] if g >= 0}
            acc[f"recall{k}"].append(len(covered) / len(gold) if gold else 0.0)
        gold_files = {gf for gf, _ in gold}
        acc["dochit5"].append(
            1 if any(chunk_files[j] in gold_files for j in idxs[:5]) else 0)

    for key, vals in acc.items():
        row[key] = float(np.mean(vals)) if vals else 0.0
    return row


def main():
    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    done = completed_keys()
    logger.info(f"Resuming with {len(done)} configs already complete")
    total = len(BENCHMARKS) * len(STRATEGIES) * len(CHUNK_SIZES) * len(MODELS)
    n = 0
    for dataset in BENCHMARKS:
        tests, files = load_benchmark(dataset)
        logger.info(f"{dataset}: {len(tests)} queries, {len(files)} files, "
                    f"{sum(len(v) for v in files.values())/1e6:.1f}M chars")
        norm_cache = {fp: span_eval.normalize_with_map(text)
                      for fp, text in files.items()}
        for model_name in MODELS:
            for strategy in STRATEGIES:
                for chunk_size in CHUNK_SIZES:
                    n += 1
                    key = (dataset, strategy, chunk_size, model_name)
                    if key in done:
                        continue
                    logger.info(f"[{n}/{total}] {dataset} | {strategy} | "
                                f"{chunk_size} | {model_name.split('/')[-1]}")
                    try:
                        row = run_config(dataset, tests, files, norm_cache,
                                         strategy, chunk_size, model_name)
                    except Exception as e:
                        logger.exception(f"config failed: {key}")
                        row = {"dataset": dataset, "strategy": strategy,
                               "chunk_size": chunk_size,
                               "embedding_model": model_name, "error": str(e)}
                    atomic_append(row)

    logger.info("E2 complete.")
    df = pd.read_csv(CHECKPOINT)
    final = os.path.join(RAW_RESULTS_DIR, "e2v2_results_final.csv")
    df.to_csv(final, index=False)
    logger.info(f"Wrote {final} ({len(df)} rows)")


if __name__ == "__main__":
    main()
