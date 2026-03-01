"""Configuration for TechChunkBench experiment."""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
QA_DIR = os.path.join(DATA_DIR, "qa_pairs")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RAW_RESULTS_DIR = os.path.join(RESULTS_DIR, "raw")
AGGREGATED_DIR = os.path.join(RESULTS_DIR, "aggregated")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Chunking strategies
CHUNKING_STRATEGIES = [
    "fixed_size",
    "fixed_overlap",
    "sentence_based",
    "recursive",
    "semantic",
    "structure_aware",
    "hybrid",
]

# Chunk sizes (in tokens)
CHUNK_SIZES = [256, 512, 1024]

# Embedding models (all free, run locally via sentence-transformers)
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",       # Lightweight baseline
    "BAAI/bge-large-en-v1.5",                        # Strong general-purpose
    "nomic-ai/nomic-embed-text-v1.5",                # Open-source, high MTEB score
]

# Retrieval settings
TOP_K_VALUES = [1, 3, 5]
VECTOR_STORE = "faiss"

# LLM Judge settings (Ollama)
LLM_JUDGE_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
USE_LLM_JUDGE = False  # Disabled: Ollama too slow for 504 configs. Using ROUGE-L heuristic instead.

# Number of repeated runs for timing measurements
TIMING_REPEATS = 1

# Checkpoint frequency
CHECKPOINT_EVERY = 50

# Document corpora (3 documents per domain, 24 total)
CORPORA = {
    # Cloud API domain
    "aws_s3": {
        "domain": "cloud_api",
        "description": "AWS S3 User Guide",
    },
    "azure_blob": {
        "domain": "cloud_api",
        "description": "Azure Blob Storage Documentation",
    },
    "gcs": {
        "domain": "cloud_api",
        "description": "Google Cloud Storage Documentation",
    },
    # Software Manual domain
    "postgresql": {
        "domain": "software_manual",
        "description": "PostgreSQL 16 Documentation",
    },
    "mysql": {
        "domain": "software_manual",
        "description": "MySQL 8.0 Reference Manual",
    },
    "sqlite": {
        "domain": "software_manual",
        "description": "SQLite Documentation",
    },
    # Hardware Manual domain
    "arduino": {
        "domain": "hardware_manual",
        "description": "Arduino Language Reference",
    },
    "raspberry_pi": {
        "domain": "hardware_manual",
        "description": "Raspberry Pi Documentation",
    },
    "esp32": {
        "domain": "hardware_manual",
        "description": "ESP-IDF Programming Guide",
    },
    # Legal/Regulatory domain
    "gdpr": {
        "domain": "legal_regulatory",
        "description": "GDPR Full Text",
    },
    "ccpa": {
        "domain": "legal_regulatory",
        "description": "California Consumer Privacy Act (CCPA)",
    },
    "hipaa": {
        "domain": "legal_regulatory",
        "description": "HIPAA Privacy Rule",
    },
    # Corporate Finance domain
    "basel_iii": {
        "domain": "corporate_finance",
        "description": "Apple Inc. 10-K Annual Report (legacy key name)",
    },
    "msft_10k": {
        "domain": "corporate_finance",
        "description": "Microsoft 10-K Annual Report (SEC EDGAR)",
    },
    "goog_10k": {
        "domain": "corporate_finance",
        "description": "Alphabet Inc. 10-K Annual Report (SEC EDGAR)",
    },
    # Technical Standard / Cybersecurity domain
    "nist_800_53": {
        "domain": "technical_standard",
        "description": "NIST SP 800-53 Rev 5",
    },
    "owasp_top10": {
        "domain": "technical_standard",
        "description": "OWASP Top 10 (2021)",
    },
    "nist_csf": {
        "domain": "technical_standard",
        "description": "NIST Cybersecurity Framework",
    },
    # Medical Guideline domain
    "who_malaria": {
        "domain": "medical_guideline",
        "description": "WHO Malaria Treatment Guidelines",
    },
    "who_covid": {
        "domain": "medical_guideline",
        "description": "WHO COVID-19 Clinical Guidelines",
    },
    "cdc_immunization": {
        "domain": "medical_guideline",
        "description": "CDC Immunization Guidelines",
    },
    # Engineering Spec / Aerospace domain
    "nasa_std": {
        "domain": "engineering_spec",
        "description": "NASA Software Engineering Requirements",
    },
    "faa_ac": {
        "domain": "engineering_spec",
        "description": "FAA Advisory Circulars",
    },
    "nasa_std_2": {
        "domain": "engineering_spec",
        "description": "NASA Systems Engineering Handbook",
    },
}

# Domain helper
DOMAINS = sorted(set(v["domain"] for v in CORPORA.values()))

# Generation metrics paths
GENERATION_METRICS_DIR = os.path.join(RESULTS_DIR, "generation_metrics")

# Minimum requirements
MIN_CORPORA = 18
MIN_EMBEDDING_MODELS = 2
MIN_QA_PAIRS_PER_CORPUS = 80
MIN_SUCCESS_RATE = 0.80
