"""Post-run analysis for the CCIS extension (tables only; figures in make_figures.py).

Inputs : results/raw/e1_results_final.csv (or checkpoint), results/raw/e2v2_results_final.csv (or checkpoint)
Outputs: results/aggregated/v2_*.csv + console summary.
Core comparisons use the four conference strategies only, so eta^2 is comparable
with the conference paper; ablation/legal strategies are analysed separately.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from config import RAW_RESULTS_DIR, AGGREGATED_DIR
warnings.filterwarnings("ignore")
OUT = AGGREGATED_DIR; os.makedirs(OUT, exist_ok=True)
CORE = ["fixed_size", "fixed_overlap", "structure_aware", "semantic"]
METRICS = ["hit1", "hit3", "hit5", "mrr", "ndcg5", "cprec"]
CONF = {"hit1": 0.009, "hit3": 0.014, "hit5": 0.019, "mrr": 0.015, "ndcg5": 0.020, "cprec": 0.017}

def pick(final, ckpt):
    p = os.path.join(RAW_RESULTS_DIR, final)
    if not os.path.exists(p): p = os.path.join(RAW_RESULTS_DIR, ckpt); print("using checkpoint", p)
    return p

def load(path, prefix, unit):
    df = pd.read_csv(path)
    if "error" in df.columns: df = df[df["error"].isna()]
    df = df.drop_duplicates([unit, "strategy", "chunk_size", "embedding_model"], keep="last").copy()
    df["model"] = df["embedding_model"].str.split("/").str[-1]; df["corpus"] = df[unit]
    for m in METRICS:
        if f"{prefix}{m}" in df.columns: df[m] = df[f"{prefix}{m}"]
    return df

def eta(df, metric, extra=("C(corpus)",), typ=3):
    f = f"Q('{metric}') ~ C(strategy) + C(chunk_size) + C(model)" + "".join(f" + {t}" for t in extra)
    an = sm.stats.anova_lm(smf.ols(f, data=df).fit(), typ=typ).drop(index="Intercept", errors="ignore")
    tot = an["sum_sq"].sum(); r = {}
    for idx, row in an.iterrows():
        n = idx.replace("C(", "").replace(")", ""); r[f"eta2_{n}"] = row["sum_sq"] / tot
        if not np.isnan(row["F"]): r[f"p_{n}"] = row["PR(>F)"]
    return r

def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for rank, i in enumerate(o): run = max(run, p[i] * (len(p) - rank)); adj[i] = min(run, 1)
    return adj

e1 = load(pick("e1_results_final.csv", "e1_results_checkpoint.csv"), "span_", "corpus_id")
e2 = load(pick("e2v2_results_final.csv", "e2v2_results_checkpoint.csv"), "", "dataset")
e1c, e2c = e1[e1.strategy.isin(CORE)], e2[e2.strategy.isin(CORE)]
print(f"E1 rows {len(e1)} (core {len(e1c)}) | E2v2 rows {len(e2)} (core {len(e2c)}) strategies={sorted(e2.strategy.unique())}")

# 1. three-regime table (core strategies), with Holm p
rows = []
for reg, df in [("R2_technical", e1c), ("R3_legal", e2c)]:
    res = [eta(df, m) for m in METRICS]; ph = holm([r["p_strategy"] for r in res])
    for m, r, pa in zip(METRICS, res, ph):
        rows.append(dict(regime=reg, metric=m, n=len(df), eta2_strategy=r["eta2_strategy"], p_raw=r["p_strategy"], p_holm=pa,
                         eta2_chunk_size=r["eta2_chunk_size"], eta2_model=r["eta2_model"], eta2_corpus=r["eta2_corpus"], eta2_residual=r["eta2_Residual"]))
for m in METRICS: rows.append(dict(regime="R1_general_QA_conference", metric=m, n=32, eta2_strategy=CONF[m]))
T1 = pd.DataFrame(rows); T1.to_csv(f"{OUT}/v2_regime_table.csv", index=False)
print("\n== eta2_strategy by regime ==\n", T1.pivot(index="metric", columns="regime", values="eta2_strategy").reindex(METRICS).round(4))

# 2. per-strategy means per regime (all strategies incl. legal/noprefix)
rows = []
for reg, df in [("R2_technical", e1), ("R3_legal", e2)]:
    g = df.groupby("strategy")[METRICS + [c for c in ["hit10", "hit20", "recall5", "recall10", "recall20", "structured_files_frac"] if c in df.columns]].mean()
    g["regime"] = reg; g["n_configs"] = df.groupby("strategy").size(); rows.append(g.reset_index())
T2 = pd.concat(rows); T2.to_csv(f"{OUT}/v2_strategy_means.csv", index=False)
print("\n== strategy means ==\n", T2.set_index(["regime", "strategy"])[[c for c in ["mrr", "hit5", "hit20", "recall20", "structured_files_frac"] if c in T2.columns]].round(3))

# 3. 2x2 ablations: boundary (fixed vs structural) x prefix, paired on (corpus,size,model)
def ablation(df, struct_pref, struct_nopref, label):
    piv = df.pivot_table(index=["corpus", "chunk_size", "model"], columns="strategy", values="mrr").dropna(subset=[struct_pref, struct_nopref, "fixed_size"])
    out = dict(regime=label, n_pairs=len(piv), fixed_mrr=piv["fixed_size"].mean(), struct_noprefix_mrr=piv[struct_nopref].mean(), struct_prefix_mrr=piv[struct_pref].mean())
    out["boundary_effect"] = out["struct_noprefix_mrr"] - out["fixed_mrr"]
    out["prefix_effect"] = out["struct_prefix_mrr"] - out["struct_noprefix_mrr"]
    out["p_boundary"] = stats.wilcoxon(piv[struct_nopref], piv["fixed_size"]).pvalue if len(piv) > 5 else np.nan
    out["p_prefix"] = stats.wilcoxon(piv[struct_pref], piv[struct_nopref]).pvalue if len(piv) > 5 else np.nan
    return out
ab = []
if "structure_noprefix" in set(e1.strategy): ab.append(ablation(e1, "structure_aware", "structure_noprefix", "R2_technical"))
if "structure_legal_noprefix" in set(e2.strategy): ab.append(ablation(e2, "structure_legal", "structure_legal_noprefix", "R3_legal_legalparser"))
if ab:
    T3 = pd.DataFrame(ab); T3.to_csv(f"{OUT}/v2_prefix_ablation.csv", index=False); print("\n== prefix ablation (MRR) ==\n", T3.round(4).to_string(index=False))

# 4. legal parser vs generic structure-aware vs fixed, per dataset (paired)
if "structure_legal" in set(e2.strategy):
    piv = e2.pivot_table(index=["corpus", "chunk_size", "model"], columns="strategy", values="mrr")
    rows = []
    for ds, g in piv.groupby(level="corpus"):
        rows.append(dict(dataset=ds, n=len(g), fixed=g["fixed_size"].mean(), fixed_overlap=g["fixed_overlap"].mean(), structure_generic=g["structure_aware"].mean(),
                         structure_legal=g.get("structure_legal", pd.Series(dtype=float)).mean(), semantic=g["semantic"].mean(),
                         structured_files_frac=e2[(e2.corpus == ds) & (e2.strategy == "structure_legal")]["structured_files_frac"].mean() if "structured_files_frac" in e2.columns else np.nan))
    T4 = pd.DataFrame(rows); T4.to_csv(f"{OUT}/v2_legal_parser_by_dataset.csv", index=False); print("\n== legal parser by dataset (MRR) ==\n", T4.round(3).to_string(index=False))
    # eta2 with structure_legal replacing generic structure_aware
    alt = e2[e2.strategy.isin(["fixed_size", "fixed_overlap", "structure_legal", "semantic"])]
    if alt.strategy.nunique() == 4:
        res = [eta(alt, m) for m in METRICS]
        T5 = pd.DataFrame([dict(metric=m, eta2_strategy=r["eta2_strategy"], p=r["p_strategy"]) for m, r in zip(METRICS, res)])
        T5.to_csv(f"{OUT}/v2_regime_R3_with_legal_parser.csv", index=False); print("\n== R3 eta2_strategy with legal parser in place of generic ==\n", T5.round(4).to_string(index=False))

# 5. floor check at depth (R3): eta2 for hit10/hit20/recall@k
deep = [c for c in ["hit10", "hit20", "recall5", "recall10", "recall20"] if c in e2c.columns]
if deep:
    rows = [dict(metric=m, mean_level=e2c[m].mean(), **{k: v for k, v in eta(e2c, m).items() if k in ("eta2_strategy", "p_strategy", "eta2_model", "eta2_corpus")}) for m in deep]
    T6 = pd.DataFrame(rows); T6.to_csv(f"{OUT}/v2_R3_depth_metrics.csv", index=False); print("\n== R3 at depth ==\n", T6.round(4).to_string(index=False))

# 6. efficiency
rows = []
for reg, df in [("R2_technical", e1), ("R3_legal", e2)]:
    g = df.groupby("strategy").agg(chunking_ms=("chunking_time_ms", "median"), embedding_ms=("embedding_time_ms", "median"), num_chunks=("num_chunks", "median"), mrr=("mrr", "mean")).reset_index(); g["regime"] = reg; rows.append(g)
T7 = pd.concat(rows); T7.to_csv(f"{OUT}/v2_efficiency.csv", index=False); print("\n== efficiency (medians) ==\n", T7.round(1).to_string(index=False))
print("\nwrote v2_* tables to", OUT)
