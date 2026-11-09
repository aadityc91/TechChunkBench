"""Variance decomposition for the extension experiments (E1, E2).

Replicates the conference paper's analysis (Type III ANOVA, eta-squared as
share of total sum of squares) on each new regime, so results are directly
comparable with the published Table 1:

  E1 (technical docs):  metric ~ strategy + chunk_size + model + corpus
                        under BOTH labeling schemes (fuzzy vs span)
  E2 (legal long docs): metric ~ strategy + chunk_size + model + dataset

Outputs one tidy CSV per experiment in results/aggregated/ plus a console
summary of strategy's share and significance per metric.
"""

import os

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import RAW_RESULTS_DIR, AGGREGATED_DIR

E1_CSV = os.path.join(RAW_RESULTS_DIR, "e1_results_final.csv")
E2_CSV = os.path.join(RAW_RESULTS_DIR, "e2_results_final.csv")

E1_METRICS_SPAN = ["span_hit1", "span_hit3", "span_hit5", "span_mrr",
                   "span_ndcg5", "span_cprec"]
E1_METRICS_FUZZY = ["fuzzy_hit1", "fuzzy_hit3", "fuzzy_hit5", "fuzzy_mrr",
                    "fuzzy_ndcg5", "fuzzy_cprec"]
E2_METRICS = ["hit1", "hit3", "hit5", "mrr", "ndcg5", "cprec", "dochit5"]


def eta_squared_table(df: pd.DataFrame, metric: str, unit_col: str) -> dict:
    """Type III ANOVA; returns eta^2 (share of total SS) per factor + strategy p."""
    data = df.dropna(subset=[metric]).copy()
    data["model_short"] = data["embedding_model"].str.split("/").str[-1]
    formula = (f"Q('{metric}') ~ C(strategy) + C(chunk_size) "
               f"+ C(model_short) + C({unit_col})")
    fit = smf.ols(formula, data=data).fit()
    anova = sm.stats.anova_lm(fit, typ=3)
    anova = anova.drop(index=[i for i in ("Intercept",) if i in anova.index])
    total_ss = anova["sum_sq"].sum()
    out = {"metric": metric, "n": len(data)}
    rename = {"C(strategy)": "strategy", "C(chunk_size)": "chunk_size",
              "C(model_short)": "emb_model", f"C({unit_col})": unit_col,
              "Residual": "residual"}
    for idx, r in anova.iterrows():
        name = rename.get(idx, idx)
        out[f"eta2_{name}"] = r["sum_sq"] / total_ss
        if name == "strategy":
            out["strategy_F"] = r["F"]
            out["strategy_p"] = r["PR(>F)"]
    return out


def analyze(csv_path: str, metrics: list, unit_col: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    rows = [eta_squared_table(df, m, unit_col) for m in metrics]
    out = pd.DataFrame(rows)
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    path = os.path.join(AGGREGATED_DIR, f"{label}_variance_decomposition.csv")
    out.to_csv(path, index=False)
    print(f"\n=== {label} (unit: {unit_col}, n={out['n'].iloc[0]}) -> {path}")
    cols = ["metric", "eta2_strategy", "eta2_chunk_size", "eta2_emb_model",
            f"eta2_{unit_col}", "eta2_residual", "strategy_p"]
    print(out[cols].round(4).to_string(index=False))
    return out


def main():
    if os.path.exists(E1_CSV):
        analyze(E1_CSV, E1_METRICS_SPAN, "corpus_id", "e1_span")
        analyze(E1_CSV, E1_METRICS_FUZZY, "corpus_id", "e1_fuzzy")
        # domain-level view (8 domains instead of 24 corpora)
        analyze(E1_CSV, E1_METRICS_SPAN, "domain", "e1_span_domain")
    else:
        print(f"missing {E1_CSV}")
    if os.path.exists(E2_CSV):
        analyze(E2_CSV, E2_METRICS, "dataset", "e2_span")
    else:
        print(f"missing {E2_CSV} (E2 not finished yet)")


if __name__ == "__main__":
    main()
