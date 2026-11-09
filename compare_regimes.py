"""Cross-regime comparison: the extension's headline table.

Places three corpus regimes side by side on the same variance-decomposition
analysis, to answer "when does chunking matter?":

  R1 general QA        NQ / MS-MARCO       from the conference paper
                                           (Chauhan & Hegde, DATA 2026, Table 1)
  R2 technical docs    24 corpora, 8 domains          E1 (this work)
  R3 long legal docs   LegalBench-RAG, 4 sub-benchmarks  E2 (this work)

Conference values are transcribed from the published Table 1 and are cited as
such in the paper; they are not recomputed here.

Outputs results/aggregated/regime_comparison.csv and a console table.
"""

import os

import pandas as pd

from config import RAW_RESULTS_DIR, AGGREGATED_DIR
from analyze_extension import eta_squared_table

# Published conference results (Chauhan & Hegde, DATA 2026, Table 1):
# eta^2 as share of total SS; strategy was not significant (p > 0.21).
CONFERENCE = {
    "hit1":  dict(strategy=0.009, chunk_size=0.024, emb_model=0.235, unit=0.620),
    "hit3":  dict(strategy=0.014, chunk_size=0.041, emb_model=0.419, unit=0.251),
    "hit5":  dict(strategy=0.019, chunk_size=0.032, emb_model=0.498, unit=0.237),
    "mrr":   dict(strategy=0.015, chunk_size=0.042, emb_model=0.358, unit=0.425),
    "ndcg5": dict(strategy=0.020, chunk_size=0.050, emb_model=0.470, unit=0.265),
    "cprec": dict(strategy=0.017, chunk_size=0.004, emb_model=0.111, unit=0.778),
}
CONFERENCE_P = "> 0.21 (n.s.)"

METRICS = ["hit1", "hit3", "hit5", "mrr", "ndcg5", "cprec"]


def rows_for(csv_path: str, prefix: str, unit_col: str, regime: str):
    """Run the decomposition on one experiment's results."""
    if not os.path.exists(csv_path):
        print(f"  (skipping {regime}: {csv_path} not found)")
        return []
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    keys = [c for c in ("corpus_id", "dataset", "strategy", "chunk_size",
                        "embedding_model") if c in df.columns]
    df = df.drop_duplicates(keys, keep="last")
    out = []
    for m in METRICS:
        col = f"{prefix}{m}"
        if col not in df.columns:
            continue
        r = eta_squared_table(df, col, unit_col)
        out.append({
            "regime": regime,
            "metric": m,
            "n_configs": r["n"],
            "eta2_strategy": round(r["eta2_strategy"], 4),
            "eta2_chunk_size": round(r["eta2_chunk_size"], 4),
            "eta2_emb_model": round(r["eta2_emb_model"], 4),
            "eta2_corpus_unit": round(r[f"eta2_{unit_col}"], 4),
            "strategy_p": f"{r['strategy_p']:.2e}",
        })
    return out


def main():
    rows = []
    for m in METRICS:
        c = CONFERENCE[m]
        rows.append({
            "regime": "R1 general QA (conference)",
            "metric": m,
            "n_configs": 32,
            "eta2_strategy": c["strategy"],
            "eta2_chunk_size": c["chunk_size"],
            "eta2_emb_model": c["emb_model"],
            "eta2_corpus_unit": c["unit"],
            "strategy_p": CONFERENCE_P,
        })

    rows += rows_for(os.path.join(RAW_RESULTS_DIR, "e1_results_final.csv"),
                     "span_", "corpus_id", "R2 technical docs (E1)")
    rows += rows_for(os.path.join(RAW_RESULTS_DIR, "e2_results_final.csv"),
                     "", "dataset", "R3 long legal docs (E2)")

    out = pd.DataFrame(rows)
    os.makedirs(AGGREGATED_DIR, exist_ok=True)
    path = os.path.join(AGGREGATED_DIR, "regime_comparison.csv")
    out.to_csv(path, index=False)

    print("\n=== Chunking strategy's share of variance across corpus regimes ===\n")
    piv = out.pivot(index="metric", columns="regime", values="eta2_strategy")
    piv = piv.reindex(METRICS)
    print(piv.to_string())
    print(f"\nfull table -> {path}")


if __name__ == "__main__":
    main()
