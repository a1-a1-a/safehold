#!/usr/bin/env bash
# Lambda (anchor-only) and lambda_c (joint) sweeps on the fixed eight probes.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
PYTHON="${PYTHON:-python}"
TRAIN_DATA="${TRAIN_DATA:-data/training_corpora/llm_lat_300.json}"
PROBE_FILE="${PROBE_FILE:-data/fixed_safety_supervision/advbench_probes8.jsonl}"
OUT_DIR="${OUT_DIR:-outputs}"
TAG="${TAG:-model}"
LAMBDAS="${LAMBDAS:-0.6 1.5 3.0}"
LCS="${LCS:-0.5 1.0 2.0}"
mkdir -p "$OUT_DIR"

for lam in $LAMBDAS; do
  dest="$OUT_DIR/${TAG}_lam_${lam}"
  if [ -e "$dest" ]; then echo "already exists: $dest"; continue; fi
  "$PYTHON" -u src/safehold/def_fix_pool.py --model "$MODEL_PATH" \
    --data "$TRAIN_DATA" --pool "$PROBE_FILE" --poolsize 8 --k 8 \
    --sched round --maxsteps 600 --epochsteps 300 --seed 0 --out "$dest" \
    --mode anchor --anchor_dir m2r --lam "$lam" --lc 0
done

for lc in $LCS; do
  dest="$OUT_DIR/${TAG}_lc_${lc}"
  if [ -e "$dest" ]; then echo "already exists: $dest"; continue; fi
  "$PYTHON" -u src/safehold/def_fix_pool.py --model "$MODEL_PATH" \
    --data "$TRAIN_DATA" --pool "$PROBE_FILE" --poolsize 8 --k 8 \
    --sched round --maxsteps 600 --epochsteps 300 --seed 0 --out "$dest" \
    --mode anchor_path --anchor_dir m2r --lam 0.6 --lc "$lc"
done
