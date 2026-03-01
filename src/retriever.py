"""FAISS-based retrieval module."""

import time
from typing import List, Tuple

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from normalized embeddings.

    Since embeddings are L2-normalized, inner product = cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def retrieve(query_embedding: np.ndarray, index: faiss.IndexFlatIP, top_k: int = 5) -> List[int]:
    """Retrieve top-k chunk indices for a single query embedding."""
    query = query_embedding.reshape(1, -1).astype(np.float32)
    _, indices = index.search(query, top_k)
    return indices[0].tolist()


def batch_retrieve(
    query_embeddings: np.ndarray, index: faiss.IndexFlatIP, top_k: int = 5
) -> Tuple[List[List[int]], float]:
    """Retrieve top-k chunk indices for multiple queries.

    Returns:
        Tuple of (list of index lists, mean latency in ms per query)
    """
    results = []
    latencies = []
    for i in range(len(query_embeddings)):
        start = time.perf_counter()
        indices = retrieve(query_embeddings[i], index, top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        results.append(indices)
    mean_latency = np.mean(latencies) if latencies else 0.0
    return results, mean_latency
