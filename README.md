# TechChunkBench

A cross-domain benchmark for evaluating text chunking strategies in technical document RAG systems.

## Overview

TechChunkBench evaluates **7 chunking strategies** x **3 chunk sizes** x **3 embedding models** across **24 technical documents in 8 domains** (3 per domain), yielding **1,512 fully-crossed configurations**. Measures retrieval quality, generation faithfulness, and computational efficiency.

### Chunking Strategies
1. **Fixed-Size Token** — Baseline, splits at exact token boundaries
2. **Fixed-Size with Overlap** — 20% overlap between chunks
3. **Sentence-Based** — Groups sentences up to target size
4. **Recursive Character** — LangChain-style recursive splitting
5. **Semantic** — Sentence embeddings + similarity breakpoints
6. **Structure-Aware** — Heading/section-based splitting with path prefixes
7. **Hybrid** — Structure-aware + semantic fallback for oversized chunks

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` — Lightweight baseline
- `BAAI/bge-large-en-v1.5` — Strong general-purpose
- `nomic-ai/nomic-embed-text-v1.5` — High MTEB score

### Document Corpora (24 documents, 3 per domain)
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

```bash
# Install dependencies
pip install -r requirements.txt
python3 -m nltk.downloader punkt punkt_tab
python3 -m spacy download en_core_web_sm

# (Optional) Install Ollama for LLM-based QA generation and judging
# https://ollama.com
# ollama pull llama3.1:8b

# Run the full experiment
python3 run_all.py
```

### Step-by-Step Execution

```bash
# Phase 1: Download documents
python3 scripts/download_documents.py

# Phase 3: Generate QA pairs
python3 scripts/generate_qa_pairs.py

# Phase 4-6: Run experiment, stats, and generate outputs
python3 run_all.py

# (Or just generate tables/figures from existing results)
python3 scripts/generate_paper_tables.py
```

## Results

After running, find outputs in:
- `results/raw/all_results_final.csv` — Full experiment results
- `results/aggregated/` — Summary tables (Markdown + CSV)
- `results/figures/` — Publication-quality plots (PNG + PDF)
- `results/SUMMARY.md` — Key findings

## Configuration

Edit `config.py` to adjust:
- Chunk sizes, embedding models, strategies
- Ollama model and availability flag
- Timing repeats, checkpoint frequency

## Citation

```bibtex
@article{techchunkbench2025,
  title={TechChunkBench: A Cross-Domain Benchmark for Text Chunking Strategies in Technical Document RAG},
  year={2025},
  journal={IEEE Access}
}
```
