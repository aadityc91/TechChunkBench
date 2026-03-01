"""Statistical significance tests for TechChunkBench."""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


def friedman_test(df: pd.DataFrame, metric: str, strategy_col: str = "strategy",
                  block_col: str = "corpus_id") -> dict:
    """Run Friedman test comparing strategies across corpora (blocks).

    Args:
        df: DataFrame with columns for strategy, corpus_id, and the metric.
        metric: Name of the metric column to test.
        strategy_col: Column name for strategies.
        block_col: Column name for blocking variable (corpora).

    Returns:
        dict with 'statistic', 'p_value', 'significant' keys.
    """
    # Pivot: rows = blocks (corpora), columns = strategies, values = metric
    pivot = df.pivot_table(index=block_col, columns=strategy_col, values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan, "significant": False,
                "error": "Not enough data for Friedman test"}

    # Friedman test expects each column as a separate array
    groups = [pivot[col].values for col in pivot.columns]
    stat, p_value = friedmanchisquare(*groups)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_blocks": pivot.shape[0],
        "n_strategies": pivot.shape[1],
    }


def friedman_test_by_domain(df: pd.DataFrame, metric: str, domain_map: dict,
                            strategy_col: str = "strategy",
                            corpus_col: str = "corpus_id") -> dict:
    """Friedman test that averages within domain first (8 blocks instead of 24).

    This variant aggregates documents within each domain to produce one value
    per domain per strategy, enabling direct comparison with the original n=1
    results.

    Args:
        df: DataFrame with columns for strategy, corpus_id, and the metric.
        metric: Name of the metric column to test.
        domain_map: Dict mapping corpus_id -> domain string.
        strategy_col: Column name for strategies.
        corpus_col: Column name for corpus identifier.

    Returns:
        dict with 'statistic', 'p_value', 'significant' keys.
    """
    df = df.copy()
    df["domain"] = df[corpus_col].map(domain_map)

    # Average within domain for each strategy
    agg = df.groupby(["domain", strategy_col])[metric].mean().reset_index()

    # Pivot: rows = domains, columns = strategies
    pivot = agg.pivot_table(index="domain", columns=strategy_col, values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan, "significant": False,
                "error": "Not enough data for Friedman test"}

    groups = [pivot[col].values for col in pivot.columns]
    stat, p_value = friedmanchisquare(*groups)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_blocks": pivot.shape[0],
        "n_strategies": pivot.shape[1],
    }


def friedman_test_fine(df: pd.DataFrame, metric: str,
                      strategy_col: str = "strategy") -> dict:
    """Friedman test using (corpus_id, chunk_size, embedding_model) as blocks.

    This gives N=216 blocks (24×3×3) instead of N=24, providing higher
    statistical power by preserving per-condition observations rather than
    averaging them away.
    """
    block_cols = ["corpus_id", "chunk_size", "embedding_model"]
    df = df.copy()
    df["_block"] = df[block_cols].astype(str).agg("|".join, axis=1)

    pivot = df.pivot_table(index="_block", columns=strategy_col, values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return {"statistic": np.nan, "p_value": np.nan, "significant": False,
                "error": "Not enough data for Friedman test"}

    groups = [pivot[col].values for col in pivot.columns]
    stat, p_value = friedmanchisquare(*groups)

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_blocks": pivot.shape[0],
        "n_strategies": pivot.shape[1],
    }


def _make_block_col(df: pd.DataFrame, block_col) -> tuple:
    """Create a composite block column if block_col is a list, or use as-is.

    Returns (df_copy, block_col_name).
    """
    if isinstance(block_col, list):
        df = df.copy()
        df["_block"] = df[block_col].astype(str).agg("|".join, axis=1)
        return df, "_block"
    return df, block_col


def nemenyi_posthoc(df: pd.DataFrame, metric: str, strategy_col: str = "strategy",
                    block_col = None) -> pd.DataFrame:
    """Run Nemenyi post-hoc test after Friedman test.

    Args:
        block_col: Blocking variable.  Default uses fine-grained blocks
            (corpus_id × chunk_size × embedding_model, N=216).  Pass
            "corpus_id" for the coarse N=24 variant.

    Returns a DataFrame of p-values for pairwise comparisons.
    """
    if block_col is None:
        block_col = ["corpus_id", "chunk_size", "embedding_model"]
    df, bcol = _make_block_col(df, block_col)

    pivot = df.pivot_table(index=bcol, columns=strategy_col, values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 3:
        return pd.DataFrame()

    result = sp.posthoc_nemenyi_friedman(pivot.values)
    result.index = pivot.columns
    result.columns = pivot.columns
    return result


def cliffs_delta(x, y) -> tuple:
    """Compute Cliff's Delta effect size.

    Returns:
        Tuple of (delta, interpretation) where interpretation is one of
        'negligible', 'small', 'medium', 'large'.
    """
    x = np.array(x)
    y = np.array(y)
    n_x = len(x)
    n_y = len(y)

    if n_x == 0 or n_y == 0:
        return 0.0, "negligible"

    # Count dominance
    more = 0
    less = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                more += 1
            elif xi < yi:
                less += 1

    delta = (more - less) / (n_x * n_y)

    # Interpret
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"

    return float(delta), interpretation


def cliffs_delta_paired(df: pd.DataFrame, s1: str, s2: str, metric: str,
                        strategy_col: str = "strategy",
                        block_cols: list = None) -> tuple:
    """Cliff's Delta for matched-observation designs.

    Matches observations by block (corpus_id, chunk_size, embedding_model)
    to ensure equal sample sizes, then computes standard all-pairs Cliff's
    Delta (dominance statistic) with thresholds from Romano et al. (2006).

    Returns:
        Tuple of (delta, interpretation).
    """
    if block_cols is None:
        block_cols = ["corpus_id", "chunk_size", "embedding_model"]

    df1 = df[df[strategy_col] == s1].set_index(block_cols)[metric].dropna()
    df2 = df[df[strategy_col] == s2].set_index(block_cols)[metric].dropna()

    common = df1.index.intersection(df2.index)
    if len(common) == 0:
        return 0.0, "negligible"

    x = df1.loc[common].values
    y = df2.loc[common].values

    return cliffs_delta(x, y)


def compute_mean_ranks(df: pd.DataFrame, metric: str, strategy_col: str = "strategy",
                       block_col = None) -> pd.Series:
    """Compute mean ranks of strategies across blocks (for CD diagram).

    Args:
        block_col: Blocking variable.  Default uses fine-grained blocks
            (corpus_id × chunk_size × embedding_model, N=216).
    """
    if block_col is None:
        block_col = ["corpus_id", "chunk_size", "embedding_model"]
    df, bcol = _make_block_col(df, block_col)

    pivot = df.pivot_table(index=bcol, columns=strategy_col, values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")

    # Rank within each row (block), higher metric = lower rank (rank 1 = best)
    ranks = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0).sort_values()
    return mean_ranks
