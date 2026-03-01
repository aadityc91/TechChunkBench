"""Embedding module supporting multiple sentence-transformer models."""

import hashlib
import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDINGS_DIR

# Cache loaded models to avoid reloading
_MODEL_CACHE = {}

# Models requiring special prefixes
_DOCUMENT_PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": "search_document: ",
}
_QUERY_PREFIXES = {
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# Max context window in tokens per model (used for truncation)
_MAX_TOKENS = {
    "nomic-ai/nomic-embed-text-v1.5": 8192,
}
# Chars-per-token ratio for pre-truncation.
# nomic's custom tokenizer ignores max_seq_length, so this pre-truncation
# is the effective limit.  Empirically, technical English averages ~4.6
# chars/token with nomic's tokenizer; 3.5 is a conservative lower bound
# that preserves ~90% of the 8192-token window while guarding against
# subword-heavy edge cases.
_CHARS_PER_TOKEN = 3.5


def _truncate_texts(texts: List[str], model_name: str) -> List[str]:
    """Truncate texts that exceed the model's max context window."""
    max_tokens = _MAX_TOKENS.get(model_name)
    if max_tokens is None:
        return texts
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    return [t[:max_chars] if len(t) > max_chars else t for t in texts]


def _get_model(model_name: str) -> SentenceTransformer:
    """Load a model, caching for reuse."""
    if model_name not in _MODEL_CACHE:
        trust_remote = model_name == "nomic-ai/nomic-embed-text-v1.5"
        # Force nomic to CPU — MPS hangs on large batch sets due to Metal
        # command buffer OOM that leaves the GPU in an unrecoverable state.
        device = "cpu" if model_name in _MAX_TOKENS else None
        model = SentenceTransformer(model_name, trust_remote_code=trust_remote, device=device)
        # Enforce max sequence length to prevent OOM on long inputs
        if model_name in _MAX_TOKENS:
            model.max_seq_length = _MAX_TOKENS[model_name]
        _MODEL_CACHE[model_name] = model
    return _MODEL_CACHE[model_name]


def _get_cache_path(corpus_id: str, strategy: str, size: int, model_name: str) -> str:
    """Build a file path for cached embeddings."""
    safe_model = model_name.replace("/", "_")
    filename = f"{corpus_id}_{strategy}_{size}_{safe_model}.npy"
    return os.path.join(EMBEDDINGS_DIR, filename)


def embed_chunks(
    chunks: List[str],
    model_name: str,
    corpus_id: str = "",
    strategy: str = "",
    size: int = 0,
    use_cache: bool = True,
) -> np.ndarray:
    """Embed a list of chunk texts, with optional disk caching."""
    # Try cache — validate both row count and content hash
    content_hash = hashlib.md5("\0".join(chunks).encode()).hexdigest()[:12]
    if use_cache and corpus_id and strategy:
        cache_path = _get_cache_path(corpus_id, strategy, size, model_name)
        hash_path = cache_path + ".hash"
        if os.path.exists(cache_path) and os.path.exists(hash_path):
            cached = np.load(cache_path)
            with open(hash_path) as hf:
                cached_hash = hf.read().strip()
            if cached.shape[0] == len(chunks) and cached_hash == content_hash:
                return cached

    model = _get_model(model_name)
    prefix = _DOCUMENT_PREFIXES.get(model_name, "")
    texts = [prefix + t for t in chunks] if prefix else chunks
    texts = _truncate_texts(texts, model_name)

    # nomic runs on CPU (MPS is unstable), so use a moderate batch size.
    # Texts are pre-truncated above.
    batch_size = 64 if model_name in _MAX_TOKENS else 256
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(np.array(embs, dtype=np.float32))
    embeddings = np.concatenate(all_embs, axis=0) if all_embs else np.array([], dtype=np.float32)

    # Save to cache with content hash
    if use_cache and corpus_id and strategy:
        cache_path = _get_cache_path(corpus_id, strategy, size, model_name)
        hash_path = cache_path + ".hash"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        with open(hash_path, "w") as hf:
            hf.write(content_hash)

    return embeddings


def embed_queries(queries: List[str], model_name: str) -> np.ndarray:
    """Embed a list of query strings."""
    model = _get_model(model_name)
    prefix = _QUERY_PREFIXES.get(model_name, "")
    texts = [prefix + q for q in queries] if prefix else queries
    texts = _truncate_texts(texts, model_name)

    batch_size = 256
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embs.append(np.array(embs, dtype=np.float32))
    return np.concatenate(all_embs, axis=0) if all_embs else np.array([], dtype=np.float32)


def get_embed_fn(model_name: str):
    """Return an embedding function suitable for SemanticChunker/HybridChunker.

    Batches large inputs to avoid OOM on MPS/GPU.
    """
    def _embed(texts: List[str]) -> np.ndarray:
        model = _get_model(model_name)
        prefix = _DOCUMENT_PREFIXES.get(model_name, "")
        input_texts = [prefix + t for t in texts] if prefix else texts
        input_texts = _truncate_texts(input_texts, model_name)

        # Batch to avoid OOM — use batch=1 for nomic to prevent padding amplification
        batch_size = 1 if model_name in _MAX_TOKENS else 256
        all_embs = []
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i : i + batch_size]
            embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embs.append(np.array(embs, dtype=np.float32))
        return np.concatenate(all_embs, axis=0) if all_embs else np.array([], dtype=np.float32)
    return _embed
