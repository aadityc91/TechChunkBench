"""QA pair generation using Ollama or template-based fallback."""

import json
import logging
import os
import re
from typing import List

import nltk
import numpy as np
import requests

from config import OLLAMA_BASE_URL, LLM_JUDGE_MODEL, QA_DIR

logger = logging.getLogger(__name__)


def generate_qa_pairs_ollama(
    text_chunk: str, doc_id: str, n_pairs: int = 5, model: str = None
) -> list:
    """Generate QA pairs from a text chunk using Ollama."""
    model = model or LLM_JUDGE_MODEL
    prompt = f"""You are creating question-answer pairs for evaluating a document retrieval system.
Given the following excerpt from a technical document, generate exactly {n_pairs} question-answer pairs.

RULES:
1. Questions should be specific and answerable ONLY from this text
2. Questions should be diverse: include factual, procedural, definitional, and comparative types
3. Answers should be concise (1-3 sentences) and directly supported by the text
4. Include the EXACT quote or sentence(s) from the text that contain the answer as "evidence"
5. Do NOT generate questions about information not in the text

TEXT:
{text_chunk}

Respond in this EXACT JSON format (no other text):
[
  {{
    "question": "What is ...?",
    "answer": "The answer is ...",
    "evidence": "Exact text from the passage that supports the answer",
    "question_type": "factual|procedural|definitional|comparative|causal"
  }}
]"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2048},
            },
            timeout=180,
        )
        response.raise_for_status()
        text = response.json().get("response", "")

        # Extract JSON array from response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            pairs = json.loads(text[start:end])
            return pairs
    except Exception as e:
        logger.warning(f"Ollama QA generation failed for {doc_id}: {e}")

    return []


def generate_qa_pairs_template(text: str, doc_id: str) -> list:
    """Generate QA pairs using regex/heuristic templates (fallback when no LLM)."""
    sentences = nltk.sent_tokenize(text)
    qa_pairs = []

    for sent in sentences:
        sent_lower = sent.lower()
        matched = False

        # Definitional patterns
        for pattern in [r"(.+?)\s+is defined as\s+(.+)", r"(.+?)\s+refers to\s+(.+)",
                        r"(.+?)\s+means\s+(.+)", r"^(.+?)\s+is\s+a\s+(.+)"]:
            match = re.match(pattern, sent, re.IGNORECASE)
            if match:
                term = match.group(1).strip().rstrip(",.")
                definition = match.group(2).strip()
                qa_pairs.append({
                    "question": f"What is {term}?",
                    "answer": f"{term} {match.group(0).split(term, 1)[1].strip()}",
                    "evidence": sent,
                    "question_type": "definitional",
                })
                matched = True
                break

        # Procedural patterns (skip if already matched)
        if not matched:
            for keyword in ["must", "should", "follow these steps", "to configure", "to set up",
                            "to create", "to enable", "to disable"]:
                if keyword in sent_lower and len(sent.split()) > 5:
                    qa_pairs.append({
                        "question": f"What is required regarding: {sent[:60].strip()}...?",
                        "answer": sent,
                        "evidence": sent,
                        "question_type": "procedural",
                    })
                    matched = True
                    break

        # Factual patterns (skip if already matched)
        if not matched and re.search(r"\d+", sent) and len(sent.split()) > 5:
            numbers = re.findall(r"\d+[\d,.]*", sent)
            if numbers and len(qa_pairs) < 200:  # Limit
                qa_pairs.append({
                    "question": f"What numerical value is mentioned in: {sent[:60].strip()}...?",
                    "answer": sent,
                    "evidence": sent,
                    "question_type": "factual",
                })

    # Deduplicate by question
    seen = set()
    unique_pairs = []
    for pair in qa_pairs:
        q = pair["question"].lower()
        if q not in seen:
            seen.add(q)
            unique_pairs.append(pair)

    return unique_pairs


def generate_qa_for_corpus(
    doc_text: str,
    corpus_id: str,
    n_total: int = 40,
    use_ollama: bool = True,
    model: str = None,
    existing_pairs: list = None,
) -> dict:
    """Generate QA pairs for a full document corpus.

    Splits the document into segments and generates QA pairs for each.
    If existing_pairs is provided, keeps them and only generates additional
    non-overlapping pairs to reach n_total.
    Returns the QA data structure ready to save as JSON.
    """
    existing_pairs = existing_pairs or []
    existing_questions = {p["question"].lower().strip() for p in existing_pairs}
    n_needed = n_total - len(existing_pairs)

    if n_needed <= 0:
        logger.info(f"  Already have {len(existing_pairs)} pairs (target {n_total}), skipping generation.")
        return {
            "corpus_id": corpus_id,
            "qa_pairs": existing_pairs[:n_total],
        }

    logger.info(f"  Have {len(existing_pairs)} existing pairs, generating {n_needed} more.")

    # Use more segments for better document coverage at higher pair counts
    n_segments = max(8, n_needed // 5)
    segments = _split_into_segments(doc_text, n_segments=n_segments)
    # Request extra pairs per segment to account for dedup/validation losses
    pairs_per_segment = max(2, (n_needed * 2) // len(segments))

    new_pairs = []

    for seg_idx, segment in enumerate(segments):
        if not segment.strip():
            continue

        if use_ollama:
            pairs = generate_qa_pairs_ollama(segment, corpus_id, n_pairs=pairs_per_segment, model=model)
            method = f"ollama_{model or LLM_JUDGE_MODEL}"
        else:
            pairs = []
            method = "template"

        # Fallback to template if Ollama failed or returned nothing
        if not pairs:
            pairs = generate_qa_pairs_template(segment, corpus_id)
            pairs = pairs[:pairs_per_segment]
            method = "template"

        # Filter out pairs whose questions overlap with existing ones
        for pair in pairs:
            q_lower = pair.get("question", "").lower().strip()
            if q_lower and q_lower not in existing_questions:
                pair["generation_method"] = method
                new_pairs.append(pair)
                existing_questions.add(q_lower)

    # Validate new pairs and assign IDs continuing from existing count
    validated_new = _validate_qa_pairs(new_pairs, doc_text, corpus_id,
                                       id_offset=len(existing_pairs))
    combined = existing_pairs + validated_new

    return {
        "corpus_id": corpus_id,
        "qa_pairs": combined[:n_total],
    }


def _split_into_segments(text: str, n_segments: int = 8) -> list:
    """Split document into roughly equal segments at section boundaries."""
    # Try splitting at heading boundaries
    lines = text.split("\n")
    heading_indices = [i for i, line in enumerate(lines) if line.startswith("#")]

    if len(heading_indices) >= n_segments:
        # Split at heading boundaries, including preamble before first heading
        step = max(1, len(heading_indices) // n_segments)
        boundaries = [0] + heading_indices[::step][:n_segments]
        boundaries.append(len(lines))
        # Deduplicate in case heading_indices[0] == 0
        boundaries = sorted(set(boundaries))

        segments = []
        for i in range(len(boundaries) - 1):
            segment = "\n".join(lines[boundaries[i] : boundaries[i + 1]])
            if segment.strip():
                segments.append(segment)
        return segments

    # Fallback: split by sentence boundaries to avoid cutting mid-word
    import nltk
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= n_segments:
        return [text]

    sents_per_seg = len(sentences) // n_segments
    segments = []
    for i in range(n_segments):
        start = i * sents_per_seg
        end = (i + 1) * sents_per_seg if i < n_segments - 1 else len(sentences)
        segment = " ".join(sentences[start:end])
        if segment.strip():
            segments.append(segment)

    return segments


def _validate_qa_pairs(pairs: list, source_text: str, corpus_id: str,
                       id_offset: int = 0) -> list:
    """Validate QA pairs: check evidence exists, remove duplicates, assign IDs."""
    validated = []
    seen_questions = set()

    for i, pair in enumerate(pairs):
        question = pair.get("question", "").strip()
        evidence = pair.get("evidence", "").strip()

        if not question or not evidence:
            continue

        # Check evidence appears as a contiguous passage in source text.
        # Use a sliding window over source sentences to find a local match,
        # rather than checking against the entire document's word set.
        evidence_words = set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", evidence.lower()))
        if not evidence_words:
            # Evidence is purely symbolic (e.g. "...", "---") — unusable
            continue

        # Check evidence appears as a contiguous passage in source text
        source_sents = nltk.sent_tokenize(source_text)
        window_size = max(1, len(evidence.split()) // 10 + 1)  # ~sentences in evidence
        found = False
        for start in range(len(source_sents)):
            end = min(start + window_size, len(source_sents))
            window_text = " ".join(source_sents[start:end])
            window_words = set(re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", window_text.lower()))
            overlap = len(evidence_words & window_words) / len(evidence_words)
            if overlap >= 0.8:
                found = True
                break
        if not found:
            continue

        # Deduplicate
        q_lower = question.lower()
        if q_lower in seen_questions:
            continue
        seen_questions.add(q_lower)

        pair["qa_id"] = f"{corpus_id}_{id_offset + len(validated) + 1:03d}"
        validated.append(pair)

    return validated


def save_qa_pairs(qa_data: dict, output_dir: str = None):
    """Save QA pairs to JSON file."""
    output_dir = output_dir or QA_DIR
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{qa_data['corpus_id']}.json")
    with open(path, "w") as f:
        json.dump(qa_data, f, indent=2)
    logger.info(f"Saved {len(qa_data['qa_pairs'])} QA pairs to {path}")


def load_qa_pairs(corpus_id: str, qa_dir: str = None) -> dict:
    """Load QA pairs from JSON file."""
    qa_dir = qa_dir or QA_DIR
    path = os.path.join(qa_dir, f"{corpus_id}.json")
    with open(path) as f:
        return json.load(f)
