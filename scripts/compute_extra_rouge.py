"""Compute ROUGE-1/2/L for all 504 configs.

Checkpoints one JSON per config in results/generation_metrics/.
Cross-validates ROUGE-L against all_results_final.csv.
Runtime: ~5-10 min.
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GENERATION_METRICS_DIR, RAW_RESULTS_DIR
from src.evaluator import compute_rouge_all
from src.replay import config_id, iter_all_configs, replay_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    os.makedirs(GENERATION_METRICS_DIR, exist_ok=True)

    # Load existing results for ROUGE-L cross-validation
    final_csv = os.path.join(RAW_RESULTS_DIR, "all_results_final.csv")
    df_original = pd.read_csv(final_csv)
    original_rouge_l = {}
    for _, row in df_original.iterrows():
        key = (row["corpus_id"], row["strategy"], int(row["chunk_size"]), row["embedding_model"])
        original_rouge_l[key] = row["rouge_l"]

    configs = list(iter_all_configs())
    total = len(configs)
    start_time = time.time()
    completed = 0
    rouge_l_diffs = []

    for idx, (corpus_id, strategy, chunk_size, model_name) in enumerate(configs):
        cid = config_id(corpus_id, strategy, chunk_size, model_name)
        checkpoint_path = os.path.join(GENERATION_METRICS_DIR, f"{cid}.json")

        # Skip completed checkpoints
        if os.path.exists(checkpoint_path):
            completed += 1
            continue

        logger.info(f"[{idx + 1}/{total}] {cid}")

        try:
            qa_results = replay_config(corpus_id, strategy, chunk_size, model_name)
        except Exception as e:
            logger.error(f"  Replay failed: {e}")
            continue

        if not qa_results:
            logger.warning(f"  No QA results, skipping")
            continue

        # ROUGE-1/2/L per query
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        for qr in qa_results:
            scores = compute_rouge_all(qr["extractive_answer"], qr["evidence"])
            rouge_1_scores.append(scores["rouge_1"])
            rouge_2_scores.append(scores["rouge_2"])
            rouge_l_scores.append(scores["rouge_l"])

        result = {
            "corpus_id": corpus_id,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "embedding_model": model_name,
            "rouge_1": float(np.mean(rouge_1_scores)),
            "rouge_2": float(np.mean(rouge_2_scores)),
            "rouge_l": float(np.mean(rouge_l_scores)),
            "num_qa_pairs": len(qa_results),
        }

        # Cross-validate ROUGE-L
        key = (corpus_id, strategy, chunk_size, model_name)
        if key in original_rouge_l:
            diff = abs(result["rouge_l"] - original_rouge_l[key])
            rouge_l_diffs.append(diff)
            if diff > 0.001:
                logger.warning(
                    f"  ROUGE-L mismatch: replay={result['rouge_l']:.6f} "
                    f"vs original={original_rouge_l[key]:.6f} (diff={diff:.6f})"
                )

        with open(checkpoint_path, "w") as f:
            json.dump(result, f, indent=2)

        completed += 1
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = (total - idx - 1) / rate if rate > 0 else 0
        logger.info(
            f"  Completed {completed}/{total}. "
            f"ETA: {remaining / 60:.1f} min"
        )

    # Summary
    if rouge_l_diffs:
        logger.info(f"\nROUGE-L cross-validation: max diff = {max(rouge_l_diffs):.6f}, "
                     f"mean diff = {np.mean(rouge_l_diffs):.6f}")
        if max(rouge_l_diffs) > 0.001:
            logger.warning("ROUGE-L diffs exceed 0.001 threshold!")
        else:
            logger.info("ROUGE-L cross-validation PASSED (all diffs < 0.001)")

    logger.info(f"Done. {completed} checkpoints in {GENERATION_METRICS_DIR}")


if __name__ == "__main__":
    main()
