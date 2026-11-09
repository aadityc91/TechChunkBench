"""Span-based relevance labeling for E1 (extension experiment).

The conference paper (Chauhan & Hegde, DATA 2026) uses gold answer-span
containment on NQ/MS-MARCO and criticizes heuristic word-overlap relevance
labels. The original TechChunkBench evaluator uses exactly such a heuristic
(_fuzzy_match: >=70% of evidence words present in the chunk). This module adds
rigorous span-based labels so E1 can report every metric under both schemes.

Approach: work in a whitespace/case-normalized character space of the source
document. Locate the QA evidence passage and each chunk's core text (heading
prefixes stripped for structure_aware) in that space, then define a chunk as
relevant iff it covers >= OVERLAP_THRESHOLD of the evidence span's characters.
"""

import math
import re
from typing import List, Optional, Tuple

OVERLAP_THRESHOLD = 0.5

_PREFIX_RE = re.compile(r"^\[[^\]\n]{1,500}\] ")


def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def normalize_with_map(text: str):
    """Normalize like normalize(), also returning orig->normalized index map.

    Produces the identical string to ' '.join(text.split()).lower() and an
    array where map[i] is the index in the normalized string of the character
    at original index i (whitespace chars map to the position of the space, or
    of the next character for leading/collapsed runs).
    """
    norm_chars: List[str] = []
    index_map = [0] * len(text)
    pending_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if norm_chars:
                pending_space = True
            index_map[i] = len(norm_chars)
            continue
        if pending_space:
            norm_chars.append(" ")
            pending_space = False
        index_map[i] = len(norm_chars)
        norm_chars.append(ch.lower())
    return "".join(norm_chars), index_map


def map_span(index_map: List[int], start: int, end: int,
             text: Optional[str] = None) -> Tuple[int, int]:
    """Map an original-space [start, end) span into normalized space.

    Pass `text` (the original document) for an exact mapping: whitespace at
    either edge of the span is trimmed first, because a whitespace character's
    index_map entry points at the *next* content character, which otherwise
    shifts the boundary by one or two characters.
    """
    n = len(index_map)
    if n == 0 or start >= n:
        return (0, 0)
    end = min(end, n)
    if text is not None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
    if start >= end:
        return (0, 0)
    return (index_map[start], index_map[end - 1] + 1)


def strip_heading_prefix(chunk_text: str, heading_path: str = "") -> str:
    """Remove the '[A > B] ' prefix that structure_aware prepends."""
    if heading_path:
        prefix = f"[{heading_path}] "
        if chunk_text.startswith(prefix):
            return chunk_text[len(prefix):]
    return _PREFIX_RE.sub("", chunk_text, count=1)


def locate_evidence(doc_norm: str, evidence: str) -> Optional[Tuple[int, int]]:
    """Find the evidence span in the normalized document, or None."""
    ev_norm = normalize(evidence)
    if not ev_norm:
        return None
    idx = doc_norm.find(ev_norm)
    if idx < 0:
        return None
    return (idx, idx + len(ev_norm))


def locate_chunks(
    doc_norm: str,
    chunk_texts: List[str],
    heading_paths: Optional[List[str]] = None,
) -> List[Optional[Tuple[int, int]]]:
    """Map each chunk to a (start, end) span in the normalized document.

    Chunks arrive in document order for every strategy in this benchmark, so a
    moving search pointer keeps matching linear and disambiguates repeated
    text. Falls back to a whole-document search, then to a prefix probe (first
    60 normalized chars) for chunks whose tail was altered by tokenizer
    round-tripping.
    """
    spans: List[Optional[Tuple[int, int]]] = []
    pointer = 0
    for i, text in enumerate(chunk_texts):
        core = strip_heading_prefix(text, heading_paths[i] if heading_paths else "")
        core_norm = normalize(core)
        if not core_norm:
            spans.append(None)
            continue
        idx = doc_norm.find(core_norm, pointer)
        if idx < 0:
            idx = doc_norm.find(core_norm)
        if idx < 0:
            probe = core_norm[:60]
            j = doc_norm.find(probe, pointer)
            if j < 0:
                j = doc_norm.find(probe)
            if j >= 0:
                idx = j
        if idx < 0:
            spans.append(None)
            continue
        end = idx + len(core_norm)
        spans.append((idx, end))
        pointer = max(pointer, idx + 1)
    return spans


def span_overlap_fraction(
    chunk_span: Optional[Tuple[int, int]],
    evidence_span: Tuple[int, int],
) -> float:
    """Fraction of the evidence span covered by the chunk span."""
    if chunk_span is None:
        return 0.0
    ev_len = evidence_span[1] - evidence_span[0]
    if ev_len <= 0:
        return 0.0
    overlap = min(chunk_span[1], evidence_span[1]) - max(chunk_span[0], evidence_span[0])
    return max(0.0, overlap / ev_len)


def relevance_labels(
    retrieved_chunk_spans: List[Optional[Tuple[int, int]]],
    evidence_span: Tuple[int, int],
    threshold: float = OVERLAP_THRESHOLD,
) -> List[int]:
    return [
        1 if span_overlap_fraction(s, evidence_span) >= threshold else 0
        for s in retrieved_chunk_spans
    ]


def span_hit(labels: List[int], k: int) -> int:
    return 1 if any(labels[:k]) else 0


def span_mrr(labels: List[int]) -> float:
    for rank, rel in enumerate(labels, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def span_ndcg(labels: List[int], k: int = 5, total_relevant: int = 1) -> float:
    """NDCG@k with binary relevance; duplicates of the single evidence span
    count once, mirroring the original evaluator's total_relevant=1 IDCG."""
    dcg = 0.0
    found = False
    for i, rel in enumerate(labels[:k]):
        if rel and not found:
            dcg += 1.0 / math.log2(i + 2)
            found = True
    ideal = min(total_relevant, k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg


def span_context_precision(labels: List[int], k: int = 5) -> float:
    if not labels[:k]:
        return 0.0
    return sum(labels[:k]) / len(labels[:k])


def span_ndcg_multi(gold_hits: List[int], k: int = 5,
                    total_relevant: int = 1) -> float:
    """NDCG@k when a query has several distinct gold snippets.

    gold_hits[i] is the index of the gold snippet that retrieved chunk i
    covers, or -1 for a non-relevant chunk. Each gold snippet contributes at
    most once, so overlapping chunks covering the same snippet do not inflate
    the score, while chunks covering *different* snippets each count. Using
    span_ndcg() here would cap DCG at a single hit and systematically
    understate multi-snippet queries.
    """
    dcg = 0.0
    seen = set()
    for i, g in enumerate(gold_hits[:k]):
        if g >= 0 and g not in seen:
            dcg += 1.0 / math.log2(i + 2)
            seen.add(g)
    ideal = min(total_relevant, k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg
