"""Generate QA pairs for all corpora."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, QA_DIR, MIN_QA_PAIRS_PER_CORPUS
from src.document_loader import load_all_corpora
from src.qa_generator import generate_qa_for_corpus, save_qa_pairs, load_qa_pairs
from src.llm_judge import is_ollama_available

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    use_ollama = is_ollama_available()
    if use_ollama:
        logger.info("Ollama is available. Using LLM-based QA generation.")
    else:
        logger.info("Ollama NOT available. Using template-based QA generation.")

    corpora = load_all_corpora()
    logger.info(f"Found {len(corpora)} corpora to process.")

    results = {}
    for corpus_id, text in corpora.items():
        logger.info(f"\nGenerating QA pairs for: {corpus_id}")

        # Load existing pairs to preserve them and avoid overlap
        existing_pairs = []
        try:
            existing_data = load_qa_pairs(corpus_id)
            existing_pairs = existing_data.get("qa_pairs", [])
            logger.info(f"  Loaded {len(existing_pairs)} existing QA pairs for {corpus_id}")
        except FileNotFoundError:
            logger.info(f"  No existing QA pairs for {corpus_id}")

        qa_data = generate_qa_for_corpus(
            text, corpus_id, n_total=100, use_ollama=use_ollama,
            existing_pairs=existing_pairs,
        )
        n_pairs = len(qa_data["qa_pairs"])
        logger.info(f"  -> Generated {n_pairs} QA pairs for {corpus_id}")

        if n_pairs < MIN_QA_PAIRS_PER_CORPUS:
            logger.warning(
                f"  -> WARNING: Only {n_pairs} pairs (minimum {MIN_QA_PAIRS_PER_CORPUS})"
            )

        save_qa_pairs(qa_data)
        results[corpus_id] = n_pairs

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("QA Generation Summary:")
    total = 0
    for cid, n in results.items():
        status = "OK" if n >= MIN_QA_PAIRS_PER_CORPUS else "LOW"
        logger.info(f"  {cid}: {n} pairs [{status}]")
        total += n
    logger.info(f"Total QA pairs: {total}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
