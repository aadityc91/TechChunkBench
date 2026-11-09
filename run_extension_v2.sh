#!/bin/bash
# CCIS extension chain (resumable; every runner checkpoints per config).
#   1. E2 v2: LegalBench-RAG, 6 strategies incl. structure_legal(+noprefix), k<=20, recall  (144 configs)
#   2. E1 ablation: structure_noprefix on 24 technical corpora                                (144 new configs)
#   3. sync raw results to OneDrive extension/results/
cd ~/chunk-ext/pipeline || exit 1
source ~/chunk-ext/venv/bin/activate
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
echo "=== chain start $(date) ===" 
python run_e2.py && python run_e1.py && touch RUNS_DONE_v2 
bash sync_results.sh
echo "=== chain end $(date) ==="
