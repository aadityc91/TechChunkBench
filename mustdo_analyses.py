"""Must-do analyses for the CCIS extension (CPU only).

A. Floor-effect check on R3 (E2 legal): per-dataset ANOVA, MAUD-excluded pooled
   ANOVA, difficulty-vs-strategy-effect across all 28 corpora.
B. Pooled interaction ANOVA (E1 + E2): strategy x regime and strategy x corpus,
   Type II SS; Holm correction across metrics; bootstrap CIs on eta^2_strategy.
C. Paired Cliff's delta + Wilcoxon for pairwise strategy contrasts per regime.
D. Corpus-feature regression: per-corpus strategy eta^2 vs heading density,
   document length, evidence-span length.
E. Label rigor (E1): fuzzy vs span labels on identical retrieval runs.

Outputs results/aggregated/mustdo_*.csv and prints a summary.
NOTE: E2 rows are the pre-NDCG-fix final CSV until the rerun completes; only
ndcg5 is affected.
"""
import os, re, json, glob, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
from config import RAW_RESULTS_DIR, AGGREGATED_DIR
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
OUT = AGGREGATED_DIR; os.makedirs(OUT, exist_ok=True)
METRICS = ["hit1", "hit3", "hit5", "mrr", "ndcg5", "cprec"]
LB = os.path.expanduser("~/chunk-ext/legalbench-data")

def load(path, prefix, unit):
    df = pd.read_csv(path)
    if "error" in df.columns: df = df[df["error"].isna()]
    keys = [unit, "strategy", "chunk_size", "embedding_model"]
    df = df.drop_duplicates(keys, keep="last").copy()
    df["model"] = df["embedding_model"].str.split("/").str[-1]
    df["corpus"] = df[unit]
    for m in METRICS: df[m] = df[f"{prefix}{m}"]
    return df

e1 = load(os.path.join(RAW_RESULTS_DIR, "e1_results_final.csv"), "span_", "corpus_id")
e1f = load(os.path.join(RAW_RESULTS_DIR, "e1_results_final.csv"), "fuzzy_", "corpus_id")
e2 = load(os.path.join(RAW_RESULTS_DIR, "e2_results_final.csv"), "", "dataset")
e1["regime"] = "R2_technical"; e2["regime"] = "R3_legal"
print(f"E1 rows={len(e1)} corpora={e1.corpus.nunique()} | E2 rows={len(e2)} datasets={e2.corpus.nunique()}")

def anova_eta(df, metric, extra_terms, typ=3):
    f = f"Q('{metric}') ~ C(strategy) + C(chunk_size) + C(model)" + ("".join(f" + {t}" for t in extra_terms))
    fit = smf.ols(f, data=df).fit()
    an = sm.stats.anova_lm(fit, typ=typ)
    an = an.drop(index=[i for i in ["Intercept"] if i in an.index])
    tot = an["sum_sq"].sum()
    res = {}
    for idx, r in an.iterrows():
        name = idx.replace("C(", "").replace(")", "").replace("Q('", "").replace("')", "")
        res[f"eta2_{name}"] = r["sum_sq"] / tot
        if not np.isnan(r["F"]): res[f"p_{name}"] = r["PR(>F)"]
    return res

def holm(pvals):
    p = np.asarray(pvals); n = len(p); order = np.argsort(p); adj = np.empty(n)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, p[i] * (n - rank)); adj[i] = min(running, 1.0)
    return adj

# ---------------- A. floor effect ----------------
print("\n=== A. Floor-effect check (R3 legal) ===")
rowsA = []
for ds, g in e2.groupby("corpus"):
    for m in METRICS:
        r = anova_eta(g, m, [])
        sm_ = g.groupby("strategy")[m].mean()
        rowsA.append(dict(dataset=ds, metric=m, n=len(g), mean_level=g[m].mean(),
                          mean_dochit5=g["dochit5"].mean(), eta2_strategy=r["eta2_strategy"],
                          p_strategy=r["p_strategy"], strategy_spread=sm_.max()-sm_.min(),
                          best=sm_.idxmax(), worst=sm_.idxmin()))
A = pd.DataFrame(rowsA); A.to_csv(f"{OUT}/mustdo_A_floor_per_dataset.csv", index=False)
print(A[A.metric.isin(["hit5","mrr"])].round(4).to_string(index=False))
rowsA2 = []
for label, sub in [("all_4", e2), ("excl_MAUD", e2[e2.corpus != "maud"]), ("excl_MAUD_ContractNLI", e2[~e2.corpus.isin(["maud","contractnli"])])]:
    for m in METRICS:
        r = anova_eta(sub, m, ["C(corpus)"])
        rowsA2.append(dict(subset=label, metric=m, n=len(sub), mean_level=sub[m].mean(),
                           eta2_strategy=r["eta2_strategy"], p_strategy=r["p_strategy"],
                           eta2_model=r["eta2_model"], eta2_corpus=r["eta2_corpus"]))
A2 = pd.DataFrame(rowsA2); A2.to_csv(f"{OUT}/mustdo_A_floor_pooled_subsets.csv", index=False)
print("\nPooled R3 with/without hardest datasets:"); print(A2.round(4).to_string(index=False))

# per-corpus strategy effect for all 28 corpora (used in A3 and D)
rowsPC = []
for df_, reg in [(e1, "R2_technical"), (e2, "R3_legal")]:
    for c, g in df_.groupby("corpus"):
        d = dict(corpus=c, regime=reg, n=len(g))
        for m in ["hit5", "mrr", "ndcg5", "cprec"]:
            r = anova_eta(g, m, [])
            d[f"eta2_strategy_{m}"] = r["eta2_strategy"]; d[f"p_strategy_{m}"] = r["p_strategy"]
            d[f"mean_{m}"] = g[m].mean()
            sm_ = g.groupby("strategy")[m].mean()
            d[f"spread_{m}"] = sm_.max() - sm_.min()
            d[f"sa_minus_fixed_{m}"] = sm_.get("structure_aware", np.nan) - sm_.get("fixed_size", np.nan)
        rowsPC.append(d)
PC = pd.DataFrame(rowsPC)
rho, p = stats.spearmanr(PC["mean_hit5"], PC["eta2_strategy_mrr"])
print(f"\nA3. Across 28 corpora: Spearman(mean hit5 difficulty, eta2_strategy[mrr]) rho={rho:.3f} p={p:.3f}")
rho2, p2 = stats.spearmanr(PC["mean_hit5"], PC["spread_mrr"])
print(f"    Spearman(mean hit5, strategy spread[mrr]) rho={rho2:.3f} p={p2:.3f}")

# ---------------- B. pooled interaction ANOVA ----------------
print("\n=== B. Pooled interaction ANOVA (E1+E2, Type II) ===")
pooled = pd.concat([e1, e2], ignore_index=True)
rowsB = []
for m in METRICS:
    f1 = f"Q('{m}') ~ C(strategy) + C(chunk_size) + C(model) + C(corpus) + C(strategy):C(regime)"
    fit1 = smf.ols(f1, data=pooled).fit(); an1 = sm.stats.anova_lm(fit1, typ=2)
    f2 = f"Q('{m}') ~ C(strategy) + C(chunk_size) + C(model) + C(corpus) + C(strategy):C(corpus)"
    fit2 = smf.ols(f2, data=pooled).fit(); an2 = sm.stats.anova_lm(fit2, typ=2)
    tot1 = an1["sum_sq"].sum(); tot2 = an2["sum_sq"].sum()
    rowsB.append(dict(metric=m, n=len(pooled),
        eta2_strategy=an1.loc["C(strategy)","sum_sq"]/tot1, p_strategy=an1.loc["C(strategy)","PR(>F)"],
        eta2_strategy_x_regime=an1.loc["C(strategy):C(regime)","sum_sq"]/tot1, p_strategy_x_regime=an1.loc["C(strategy):C(regime)","PR(>F)"],
        eta2_strategy_x_corpus=an2.loc["C(strategy):C(corpus)","sum_sq"]/tot2, p_strategy_x_corpus=an2.loc["C(strategy):C(corpus)","PR(>F)"],
        eta2_model=an1.loc["C(model)","sum_sq"]/tot1, eta2_corpus=an1.loc["C(corpus)","sum_sq"]/tot1,
        eta2_chunk_size=an1.loc["C(chunk_size)","sum_sq"]/tot1))
B = pd.DataFrame(rowsB); B.to_csv(f"{OUT}/mustdo_B_interaction_anova.csv", index=False)
print(B.round(4).to_string(index=False))

# Holm across 6 metrics within each regime (Type III as in the paper)
rowsH = []
for df_, reg in [(e1, "R2_technical"), (e2, "R3_legal")]:
    ps = []; etas = []
    for m in METRICS:
        r = anova_eta(df_, m, ["C(corpus)"]); ps.append(r["p_strategy"]); etas.append(r["eta2_strategy"])
    adj = holm(ps)
    for m, pv, pa, et in zip(METRICS, ps, adj, etas):
        rowsH.append(dict(regime=reg, metric=m, eta2_strategy=et, p_raw=pv, p_holm=pa, sig_holm_05=pa < 0.05))
H = pd.DataFrame(rowsH); H.to_csv(f"{OUT}/mustdo_B_holm.csv", index=False)
print("\nHolm-corrected strategy p per regime:"); print(H.round(4).to_string(index=False))

# bootstrap CIs on eta2_strategy per regime (resample configs within corpus strata)
print("\nBootstrapping eta2_strategy CIs (1000 reps)...")
def boot_eta(df_, m, reps=1000):
    vals = []
    groups = [g.index.values for _, g in df_.groupby("corpus")]
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        s = df_.loc[idx]
        try: vals.append(anova_eta(s, m, ["C(corpus)"])["eta2_strategy"])
        except Exception: pass
    return np.percentile(vals, [2.5, 97.5])
rowsCI = []
for df_, reg in [(e1, "R2_technical"), (e2, "R3_legal")]:
    for m in METRICS:
        pt = anova_eta(df_, m, ["C(corpus)"])["eta2_strategy"]; lo, hi = boot_eta(df_, m)
        rowsCI.append(dict(regime=reg, metric=m, eta2_strategy=pt, ci_lo=lo, ci_hi=hi))
CI = pd.DataFrame(rowsCI); CI.to_csv(f"{OUT}/mustdo_B_bootstrap_ci.csv", index=False)
print(CI.round(4).to_string(index=False))

# ---------------- C. Cliff's delta, paired ----------------
print("\n=== C. Paired strategy contrasts (Cliff's delta on paired differences, Wilcoxon) ===")
def cliffs(x, y):
    x = np.asarray(x); y = np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))
rowsC = []
strategies = ["fixed_size", "fixed_overlap", "structure_aware", "semantic"]
for df_, reg in [(e1, "R2_technical"), (e2, "R3_legal")]:
    for m in ["hit5", "mrr", "ndcg5"]:
        piv = df_.pivot_table(index=["corpus", "chunk_size", "model"], columns="strategy", values=m)
        for i, a in enumerate(strategies):
            for b in strategies[i+1:]:
                d = cliffs(piv[a], piv[b]); w = stats.wilcoxon(piv[a], piv[b], zero_method="wilcox").pvalue if (piv[a]-piv[b]).abs().sum() > 0 else 1.0
                rowsC.append(dict(regime=reg, metric=m, A=a, B=b, mean_A=piv[a].mean(), mean_B=piv[b].mean(),
                                  mean_diff=(piv[a]-piv[b]).mean(), cliffs_delta=d, wilcoxon_p=w, n_pairs=len(piv)))
CD = pd.DataFrame(rowsC); CD.to_csv(f"{OUT}/mustdo_C_cliffs_delta.csv", index=False)
print(CD[CD.metric=="mrr"].round(4).to_string(index=False))

# ---------------- D. corpus features ----------------
print("\n=== D. Corpus features vs per-corpus strategy effect ===")
HEAD = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
feat = []
for c in e1.corpus.unique():
    txt = open(f"data/processed/{c}.txt", encoding="utf-8", errors="ignore").read()
    qa = json.load(open(f"data/qa_pairs/{c}.json"))["qa_pairs"]
    nh = len(HEAD.findall(txt))
    secs = [len(s) for s in HEAD.split(txt)[::3] if s.strip()]
    feat.append(dict(corpus=c, regime="R2_technical", n_docs=1, total_chars=len(txt), mean_doc_chars=len(txt),
                     n_headings=nh, headings_per_10k=nh/len(txt)*1e4, mean_section_chars=np.mean(secs) if secs else len(txt),
                     mean_evidence_chars=np.mean([len(q["evidence"]) for q in qa]), n_queries=len(qa),
                     domain=e1[e1.corpus==c].domain.iloc[0]))
for ds in e2.corpus.unique():
    files = glob.glob(f"{LB}/corpus/{ds}/*"); lens = []; nh = 0
    for fp in files:
        t = open(fp, encoding="utf-8", errors="ignore").read(); lens.append(len(t)); nh += len(HEAD.findall(t))
    tests = json.load(open(f"{LB}/benchmarks/{ds}.json"))["tests"]
    snip = [s["span"][1]-s["span"][0] for t in tests for s in t["snippets"]]
    feat.append(dict(corpus=ds, regime="R3_legal", n_docs=len(files), total_chars=sum(lens), mean_doc_chars=np.mean(lens),
                     n_headings=nh, headings_per_10k=nh/sum(lens)*1e4, mean_section_chars=np.nan,
                     mean_evidence_chars=np.mean(snip), n_queries=len(tests), domain="legal"))
F = pd.DataFrame(feat).merge(PC, on=["corpus","regime"])
F["log_doc_chars"] = np.log10(F["mean_doc_chars"]); F["log_evidence_chars"] = np.log10(F["mean_evidence_chars"])
F.to_csv(f"{OUT}/mustdo_D_corpus_features.csv", index=False)
print(F[["corpus","regime","mean_doc_chars","headings_per_10k","mean_evidence_chars","mean_hit5","eta2_strategy_mrr","sa_minus_fixed_mrr"]].round(3).to_string(index=False))
print("\nSpearman correlations with per-corpus eta2_strategy (MRR), n=%d:" % len(F))
for col in ["headings_per_10k", "log_doc_chars", "log_evidence_chars", "mean_hit5", "n_queries"]:
    r, p = stats.spearmanr(F[col], F["eta2_strategy_mrr"]); r2, p2 = stats.spearmanr(F[col], F["sa_minus_fixed_mrr"])
    print(f"  {col:20s} vs eta2: rho={r:+.3f} p={p:.3f} | vs (structure_aware - fixed) MRR: rho={r2:+.3f} p={p2:.3f}")
ols = smf.ols("eta2_strategy_mrr ~ headings_per_10k + log_doc_chars + log_evidence_chars", data=F).fit()
print("\nOLS eta2_strategy_mrr ~ heading density + log doc length + log evidence length:")
print(ols.summary().tables[1]); print(f"R^2={ols.rsquared:.3f} adjR^2={ols.rsquared_adj:.3f} n={int(ols.nobs)}")
# E1-only version (same regime, cleaner)
F1 = F[F.regime=="R2_technical"]
ols1 = smf.ols("eta2_strategy_mrr ~ headings_per_10k + log_doc_chars + log_evidence_chars", data=F1).fit()
print(f"E1-only OLS: R^2={ols1.rsquared:.3f} n={int(ols1.nobs)}; coef headings_per_10k p={ols1.pvalues['headings_per_10k']:.3f}, log_doc p={ols1.pvalues['log_doc_chars']:.3f}, log_evid p={ols1.pvalues['log_evidence_chars']:.3f}")

# ---------------- E. label rigor ----------------
print("\n=== E. Label rigor on E1 (fuzzy vs span, identical retrieval runs) ===")
rowsE = []
for m in METRICS:
    rs = anova_eta(e1, m, ["C(corpus)"]); rf = anova_eta(e1f, m, ["C(corpus)"])
    ms = e1.groupby("strategy")[m].mean(); mf = e1f.groupby("strategy")[m].mean()
    rowsE.append(dict(metric=m, eta2_strategy_span=rs["eta2_strategy"], eta2_strategy_fuzzy=rf["eta2_strategy"],
                      ratio_fuzzy_over_span=rf["eta2_strategy"]/rs["eta2_strategy"],
                      mean_span=e1[m].mean(), mean_fuzzy=e1f[m].mean(),
                      sa_minus_fixed_span=ms["structure_aware"]-ms["fixed_size"], sa_minus_fixed_fuzzy=mf["structure_aware"]-mf["fixed_size"],
                      corr_span_fuzzy=np.corrcoef(e1[m], e1f[m])[0,1]))
E = pd.DataFrame(rowsE); E.to_csv(f"{OUT}/mustdo_E_label_rigor.csv", index=False)
print(E.round(4).to_string(index=False))
inflation = (e1f[METRICS].mean() - e1[METRICS].mean())
print("\nPer-strategy fuzzy-minus-span inflation (MRR):")
print((e1f.groupby("strategy")["mrr"].mean() - e1.groupby("strategy")["mrr"].mean()).round(4).to_string())
print("\nDone. Outputs in", OUT)
