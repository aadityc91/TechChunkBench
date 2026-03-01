# TechChunkBench

A comprehensive benchmark for evaluating text chunking strategies in Retrieval-Augmented Generation (RAG) pipelines, applied to technical documentation.

## Overview

TechChunkBench is a fully-crossed experiment evaluating **7 chunking strategies** across **3 chunk sizes**, **3 embedding models**, and **24 technical documents** spanning **8 domains** (3 per domain), yielding **1,512 configurations**. Each configuration is evaluated on retrieval quality (Hit@k, MRR, NDCG@5, context precision) and computational efficiency.

## Key Findings

- **Structure-aware** and **hybrid** chunking form a significantly superior top tier (Friedman p < 10^-76; Nemenyi p < 0.0001 for all inter-tier comparisons)
- Three distinct performance tiers: structure-aware/hybrid > fixed-overlap/recursive > fixed-size/sentence-based/semantic
- Structure-aware achieves the highest MRR (0.704) with low overhead (157.7 ms chunking time)
- Semantic chunking is the worst choice: 401x slower than fixed-size while achieving the lowest retrieval quality
- Document domain explains 32-57% of performance variance, embedding model 21-38%, and chunking strategy 9-27%

## Repository Structure

```
TechChunkBench/
├── src/                          # Core modules
│   ├── chunkers/                 # 7 chunking strategy implementations
│   ├── embedder.py               # Embedding with caching
│   ├── retriever.py              # FAISS-based retrieval
│   ├── evaluator.py              # Hit@k, MRR, NDCG, context precision
│   └── stats.py                  # Friedman, Nemenyi, Cliff's Delta, variance decomposition
├── scripts/                      # Pipeline scripts
│   ├── download_documents.py     # Fetch 24 source documents
│   ├── generate_qa_pairs.py      # Generate QA pairs
│   ├── aggregate_results.py      # Merge raw results into summary tables
│   ├── compute_statistics.py     # Statistical analysis
│   └── generate_paper_tables.py  # Generate tables and figures
├── data/
│   ├── processed/                # 24 source documents (plain text)
│   └── qa_pairs/                 # 2,406 QA pairs across 24 corpora
├── results/
│   ├── raw/                      # Full experiment results (1,512 rows)
│   └── aggregated/               # Summary tables (CSV)
├── config.py                     # Experiment configuration
├── run_all.py                    # Sequential experiment runner
├── run_parallel.py               # Parallel experiment runner (recommended)
├── test_*.py                     # Unit tests
└── requirements.txt
```

## Chunking Strategies

| Strategy | Description |
|----------|-------------|
| Fixed-Size Token | Baseline; splits at exact token boundaries |
| Fixed-Size with Overlap | 20% token overlap between consecutive chunks |
| Sentence-Based | Groups complete sentences up to target size |
| Recursive Character | LangChain-style recursive splitting on hierarchical delimiters |
| Semantic | Sentence embeddings + cosine similarity breakpoints |
| Structure-Aware | Heading/section-based splitting with hierarchical path prefixes |
| Hybrid | Structure-aware with semantic fallback for oversized chunks |

## Embedding Models

| Model | Dimensions |
|-------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `BAAI/bge-large-en-v1.5` | 1024 |
| `nomic-ai/nomic-embed-text-v1.5` | 768 |

## Document Corpora

24 documents across 8 domains (3 per domain):

| Domain | Documents |
|--------|-----------|
| Cloud API | AWS S3, Azure Blob Storage, Google Cloud Storage |
| Database | PostgreSQL, MySQL, SQLite |
| Hardware | Arduino, Raspberry Pi, ESP32 |
| Legal | GDPR, CCPA, HIPAA |
| Corporate Finance | Apple 10-K, Microsoft 10-K, Alphabet 10-K |
| Cybersecurity | NIST 800-53, OWASP Top 10, NIST CSF |
| Medical | WHO Malaria, WHO COVID-19, CDC Immunization |
| Aerospace | NASA SW Engineering, FAA Advisory Circulars, NASA SE Handbook |

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
python3 -m nltk.downloader punkt punkt_tab
python3 -m spacy download en_core_web_sm
```

### Reproduce from Existing Results

The repository includes pre-computed results. To regenerate tables and statistical analysis:

```bash
python3 scripts/compute_statistics.py
python3 scripts/generate_paper_tables.py
```

### Run Full Experiment

```bash
# Step 1: Download source documents
python3 scripts/download_documents.py

# Step 2: Generate QA pairs
python3 scripts/generate_qa_pairs.py

# Step 3: Run experiment (parallel runner recommended)
python3 run_parallel.py

# Step 4: Post-processing
python3 scripts/compute_extra_rouge.py
python3 scripts/aggregate_results.py
python3 scripts/compute_statistics.py
python3 scripts/generate_paper_tables.py
```

## Results

Pre-computed results are included in this repository:

| File | Description |
|------|-------------|
| `results/raw/all_results_final.csv` | Full experiment output (1,512 configurations) |
| `results/aggregated/table1_overall_retrieval.csv` | Overall retrieval metrics by strategy |
| `results/aggregated/table2_strategy_x_model.csv` | Strategy x embedding model interaction (MRR) |
| `results/aggregated/table3_strategy_x_domain.csv` | Strategy x domain interaction (Hit@3) |
| `results/aggregated/table4_strategy_x_size.csv` | Strategy x chunk size interaction (MRR) |
| `results/aggregated/table5_efficiency.csv` | Computational efficiency metrics |
| `results/aggregated/table7_significance.csv` | Nemenyi post-hoc pairwise p-values |
| `results/aggregated/table8_variance_decomposition.csv` | Partial eta-squared variance decomposition |
| `results/aggregated/table9_cliffs_delta_summary.csv` | Cliff's Delta effect size summary |
| `results/statistical_analysis.json` | Full statistical analysis output |

## Configuration

Edit `config.py` to modify:
- `CHUNKING_STRATEGIES` — which strategies to evaluate
- `CHUNK_SIZES` — target chunk sizes in tokens (default: 256, 512, 1024)
- `EMBEDDING_MODELS` — which embedding models to use
- `CORPORA` — document corpus definitions and domain mappings

## Tests

```bash
python3 test_friedman.py
python3 test_cliffs_paired.py
python3 test_ndcg_part4.py
python3 test_fixed_overlap.py
python3 test_qa_validation.py
```

## Citation

```bibtex
@article{chauhan2025techchunkbench,
  title={TechChunkBench: A Comprehensive Benchmark for Evaluating Text Chunking Strategies in Retrieval-Augmented Generation},
  author={Chauhan, Aaditya and Hegde, Nivas},
  journal={IEEE Access},
  year={2025}
}
```

## License

This project is released for academic research purposes.
