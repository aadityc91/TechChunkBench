#!/bin/bash
D=/mnt/c/Users/nivas/OneDrive/Documents/Work/chunk-paper/extension/results
mkdir -p "$D/raw_v2"
cp -f ~/chunk-ext/pipeline/results/raw/e2v2_results_checkpoint.csv "$D/raw_v2/" 2>/dev/null
cp -f ~/chunk-ext/pipeline/results/raw/e2v2_results_final.csv "$D/raw_v2/" 2>/dev/null
cp -f ~/chunk-ext/pipeline/results/raw/e1_results_checkpoint.csv "$D/raw_v2/" 2>/dev/null
cp -f ~/chunk-ext/pipeline/results/raw/e1_results_final.csv "$D/raw_v2/" 2>/dev/null
echo "synced $(date)"
