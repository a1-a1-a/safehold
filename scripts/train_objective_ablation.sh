#!/usr/bin/env bash
# A / R / F / J on the same 300 submitted pairs and fixed eight probes.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
PYTHON="${PYTHON:-python}"
TRAIN_DATA="${TRAIN_DATA:-data/training_corpora/llm_lat_300.json}"
PROBE_FILE="${PROBE_FILE:-data/fixed_safety_supervision/advbench_probes8.jsonl}"
OUT_DIR="${OUT_DIR:-outputs}"
TAG="${TAG:-model}"
mkdir -p "$OUT_DIR"
for variant in A R F J; do
  case "$variant" in
    A) mode=anchor;;
    R) mode=path;;
    F) mode=anchor_first;;
    J) mode=anchor_path;;
  esac
  dest="$OUT_DIR/${TAG}_ab_${variant}"
  if [ -e "$dest" ]; then echo "already exists: $dest"; continue; fi
  "$PYTHON" -u src/safehold/def_fix_pool.py --model "$MODEL_PATH" \
    --data "$TRAIN_DATA" --pool "$PROBE_FILE" --poolsize 8 --k 8 \
    --sched round --maxsteps 600 --epochsteps 300 --seed 0 --out "$dest" \
    --mode "$mode" --anchor_dir m2r --lam 0.6 --lc 1.0
done
