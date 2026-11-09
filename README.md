# Chunking Is Not the Bottleneck — experiment code

Code and results for

* **Chauhan, A. and Hegde, N. (2026).** *Chunking Is Not the Bottleneck: Why RAG Evaluation Must Look Beyond Segmentation Strategy.* Proceedings of the 15th International Conference on Data Science, Technology and Applications (DATA 2026). SCITEPRESS.
* the revised and extended version submitted to *Communications in Computer and Information Science* (Springer, DATA 2026 selected papers), which adds two further corpus regimes (branch `ccis-extension`).

The repository name (TechChunkBench) is historical: the 24-document technical corpus it was built for is reused as one of the three regimes studied in the extended paper.

## Research question

How much of the variance in RAG retrieval quality is attributable to the chunking strategy, relative to the embedding model, the chunk size, and the corpus itself? The conference paper answered this for general-purpose QA (Natural Questions, MS-MARCO): chunking strategy explains 1-2 % of variance and is not significant. The extension asks *when* chunking matters by repeating the balanced factorial on two further regimes.

| Regime | Corpora | Gold labels | Configs | Runner |
|---|---|---|---|---|
| R1 general QA (conference) | NQ dev, MS-MARCO dev | answer-span / qrels | 32 | conference pipeline |
| R2 structured technical docs | 24 manuals/standards/filings, 8 domains | evidence char-span >= 50 % coverage (plus legacy fuzzy labels) | 576 + 144 ablation | `run_e1.py` |
| R3 long legal contracts | LegalBench-RAG (PrivacyQA, ContractNLI, CUAD, MAUD) | expert-annotated char spans | 144 | `run_e2.py` |

Factors: chunking strategy (fixed-size, fixed-size with overlap, structure-aware, semantic; plus `structure_legal` and no-prefix ablation variants), chunk size (256, 512 tokens), embedding model (all-MiniLM-L6-v2, BGE-large-en-v1.5, nomic-embed-text-v1.5), corpus. Metrics: Hit@k, MRR, NDCG@5, context precision, recall@k (R3). Analysis: Type III ANOVA with eta^2 as share of total sum of squares, strategy x corpus interactions, bootstrap CIs, Holm correction, paired Cliff's delta.

## Layout

```
src/chunkers/          chunking strategies (structure_legal.py adds the legal-aware + no-prefix variants)
src/span_eval.py       exact character-span relevance labelling, multi-gold NDCG
src/embedder.py        sentence-transformers embedding with on-disk caching
src/retriever.py       FAISS cosine retrieval, one shared index per corpus
run_e1.py              R2 technical-corpus factorial (resumable checkpoints)
run_e2.py              R3 LegalBench-RAG factorial (resumable checkpoints)
analyze_extension.py   per-regime variance decomposition
compare_regimes.py     three-regime headline table (R1 values from the conference paper)
mustdo_analyses.py     floor-effect checks, interaction ANOVA, bootstrap CIs, Cliff's delta,
                       corpus-feature regression, label-rigor comparison
data/processed/        24 technical source documents (plain text)
data/qa_pairs/         2,406 QA pairs with evidence passages
results/raw/           one row per configuration
results/aggregated/    tables used in the papers
```

LegalBench-RAG data is not redistributed; download it from the authors' release (Pipitone and Houir Alami, 2024) into `~/chunk-ext/legalbench-data/` (`benchmarks/*.json`, `corpus/<dataset>/`).

## Reproduce

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # torch with CUDA, sentence-transformers, faiss, statsmodels
python run_e1.py                           # ~5 GPU-hours on an RTX 3080 capped at 200 W
python run_e2.py                           # ~25 GPU-hours (semantic chunking on MAUD dominates)
python analyze_extension.py && python compare_regimes.py && python mustdo_analyses.py
```

Both runners checkpoint one row per finished configuration with atomic writes and skip completed work on restart.

## Notes

* Earlier versions of this README described a preliminary 7-strategy study on the technical corpus that used fuzzy word-overlap relevance labels. That analysis was superseded by the span-labelled design above; the fuzzy labels are retained only for the label-rigor comparison in the extended paper.
* The authors used Claude (Anthropic) to assist with code implementation, experiment execution and analysis. Experimental design, interpretation and the integrity of reported results are the authors' responsibility.
