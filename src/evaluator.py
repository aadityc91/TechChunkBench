"""Evaluation metrics for retrieval and generation quality."""

import math
from typing import List

import numpy as np


def _word_set(text: str) -> set:
    """Tokenize text into a set of lowercased words, preserving dotted/hyphenated tokens."""
    import re
    # Match word tokens, keeping dots and hyphens between alphanumeric parts
    # e.g. "v2.0" → "v2.0", "Section 4.1" → {"section", "4.1"}, "800-53" → "800-53"
    return set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower()))


def _fuzzy_match(chunk: str, evidence: str, threshold: float = 0.7) -> bool:
    """Check if chunk contains >= threshold fraction of evidence words."""
    evidence_words = _word_set(evidence)
    if not evidence_words:
        return False
    chunk_words = _word_set(chunk)
    overlap = evidence_words & chunk_words
    return len(overlap) / len(evidence_words) >= threshold


# --- Retrieval Metrics ---


def compute_hit(retrieved_texts: List[str], evidence: str) -> int:
    """Return 1 if any retrieved chunk matches the evidence, 0 otherwise."""
    for chunk in retrieved_texts:
        if _fuzzy_match(chunk, evidence):
            return 1
    return 0


def compute_mrr(retrieved_texts: List[str], evidence: str) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    for rank, chunk in enumerate(retrieved_texts, start=1):
        if _fuzzy_match(chunk, evidence):
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_texts: List[str], evidence: str, k: int = 5,
                  total_relevant: int = None) -> float:
    """Compute NDCG@k for a single query.

    Args:
        total_relevant: Total number of relevant documents in the full corpus.
            If None, falls back to counting relevant docs in retrieved set.
    """
    # Each QA pair has one evidence passage; duplicate chunk matches are
    # copies of the same information, so only the first match counts.
    relevances = []
    found = False
    for chunk in retrieved_texts[:k]:
        if not found and _fuzzy_match(chunk, evidence):
            relevances.append(1.0)
            found = True
        else:
            relevances.append(0.0)

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    # IDCG: best possible ranking using total relevant in corpus (capped at k)
    num_relevant_for_ideal = total_relevant if total_relevant is not None else int(sum(relevances))
    num_relevant_for_ideal = min(num_relevant_for_ideal, k)
    if num_relevant_for_ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant_for_ideal))

    return dcg / idcg if idcg > 0 else 0.0


def compute_context_precision(retrieved_texts: List[str], evidence: str) -> float:
    """Context precision: fraction of retrieved chunks that match the evidence.

    Measures what proportion of the retrieved context window contains
    relevant information.  Distinct from MRR (which measures rank of
    first hit) and from NDCG (which is rank-weighted).
    """
    if not retrieved_texts:
        return 0.0
    relevant = sum(1 for chunk in retrieved_texts if _fuzzy_match(chunk, evidence))
    return relevant / len(retrieved_texts)


# --- Heuristic Generation Metrics (no LLM needed) ---


def get_most_relevant_sentence(context: str, question: str, answer: str = "") -> str:
    """Extract the most relevant sentence from context based on word overlap with the answer.

    If answer is provided, selects the sentence with most overlap to the answer
    (measuring extractive answer quality). Falls back to question overlap if no answer.
    """
    import nltk

    sentences = nltk.sent_tokenize(context)
    if not sentences:
        return ""

    target_words = _word_set(answer) if answer else _word_set(question)
    best_sent = ""
    best_score = 0

    for sent in sentences:
        sent_words = _word_set(sent)
        overlap = len(target_words & sent_words)
        if overlap > best_score:
            best_score = overlap
            best_sent = sent

    return best_sent


def compute_rouge_all(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores in one call."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def compute_token_overlap_faithfulness(answer: str, context: str) -> float:
    """Check what fraction of answer tokens appear in the context."""
    answer_words = _word_set(answer)
    context_words = _word_set(context)
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def compute_evidence_coverage(retrieved_texts: List[str], evidence: str) -> float:
    """Check if the evidence string appears in the retrieved chunks."""
    combined = " ".join(retrieved_texts).lower()
    evidence_words = _word_set(evidence)
    if not evidence_words:
        return 0.0
    combined_words = _word_set(combined)
    overlap = evidence_words & combined_words
    return len(overlap) / len(evidence_words)
