"""Merge all results into final CSV.

Loads results/raw/all_results_final.csv, then merges in checkpoint JSONs
from results/generation_metrics/.
Output: results/results_with_generation_metrics.csv (1512 rows)
"""

import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GENERATION_METRICS_DIR, RAW_RESULTS_DIR, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MERGE_KEYS = ["corpus_id", "strategy", "chunk_size", "embedding_model"]


def _load_checkpoints(directory, label):
    """Load all JSON checkpoint files from a directory into a DataFrame."""
    records = []
    if not os.path.exists(directory):
        logger.warning(f"No {label} directory: {directory}")
        return pd.DataFrame()

    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    logger.info(f"Loading {len(files)} {label} checkpoints from {directory}")

    for filename in files:
        path = os.path.join(directory, filename)
        try:
            with open(path) as f:
                data = json.load(f)
            records.append(data)
        except Exception as e:
            logger.warning(f"  Failed to load {filename}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def main():
    # 1. Load original results
    original_csv = os.path.join(RAW_RESULTS_DIR, "all_results_final.csv")
    df = pd.read_csv(original_csv)
    logger.info(f"Loaded original results: {len(df)} rows, {list(df.columns)}")

    # 2. Load ROUGE checkpoints
    df_gen = _load_checkpoints(GENERATION_METRICS_DIR, "generation_metrics")

    if len(df_gen) > 0:
        # Drop columns that duplicate originals (except merge keys and new metrics)
        new_gen_cols = ["rouge_1", "rouge_2"]
        gen_keep = MERGE_KEYS + [c for c in new_gen_cols if c in df_gen.columns]
        # Also keep the new rouge_l from replay for replacement
        if "rouge_l" in df_gen.columns:
            gen_keep.append("rouge_l")
            df_gen = df_gen[gen_keep].rename(columns={"rouge_l": "rouge_l_new"})
        else:
            df_gen = df_gen[gen_keep]

        df_gen["chunk_size"] = df_gen["chunk_size"].astype(int)
        df = df.merge(df_gen, on=MERGE_KEYS, how="left")

        # Replace old rouge_l with new computation if available
        if "rouge_l_new" in df.columns:
            mask = df["rouge_l_new"].notna()
            df.loc[mask, "rouge_l"] = df.loc[mask, "rouge_l_new"]
            df = df.drop(columns=["rouge_l_new"])

        logger.info(f"Merged {len(df_gen)} generation metric checkpoints")
    else:
        logger.warning("No generation metric checkpoints found")

    # 3. Save
    output_path = os.path.join(RESULTS_DIR, "results_with_generation_metrics.csv")
    df.to_csv(output_path, index=False)

    # 5. Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Output: {output_path}")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")

    logger.info(f"\nNon-null counts per new metric:")
    new_metrics = ["rouge_1", "rouge_2"]
    for metric in new_metrics:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            logger.info(f"  {metric}: {non_null}/{len(df)}")
        else:
            logger.info(f"  {metric}: not present")

    # Validate
    if len(df) != 1512:
        logger.warning(f"Expected 1512 rows, got {len(df)}")

    # Check for unexpected NaN in ROUGE (should be complete)
    for metric in ["rouge_1", "rouge_2"]:
        if metric in df.columns:
            nan_count = df[metric].isna().sum()
            if nan_count > 0:
                logger.warning(f"Unexpected NaN in {metric}: {nan_count}/{len(df)}")


if __name__ == "__main__":
    main()
