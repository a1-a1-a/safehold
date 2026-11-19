#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
DATA_DIR="${DATA_DIR:-data/training_corpora/mixed_fractions}"
OUT_DIR="${OUT_DIR:-outputs}"
PROBE_FILE="${PROBE_FILE:-data/fixed_safety_supervision/advbench_probes8.jsonl}"
TAG="${TAG:-model}"
PYTHON="${PYTHON:-python}"
FRACTIONS="${FRACTIONS:-000 025 050 075 100}"
MODES="${MODES:-undef safe}"
MAXSTEPS="${MAXSTEPS:-600}"
EPOCHSTEPS="${EPOCHSTEPS:-300}"
mkdir -p "$OUT_DIR"

for frac in $FRACTIONS; do
  data="$DATA_DIR/mixed_${frac#0}.json"
  if [ "$frac" = 000 ]; then data="$DATA_DIR/mixed_0.json"; fi
  [ -f "$data" ] || { echo "missing $data"; exit 1; }
  for mode in $MODES; do
    name="${TAG}_f${frac}_${mode}"
    if [ -f "$OUT_DIR/$name/adapter_config.json" ]; then
      echo "skip $name"; continue
    fi
    common=(--model "$MODEL_PATH" --data "$data" --pool "$PROBE_FILE"
            --poolsize 8 --k 8 --sched round --maxsteps "$MAXSTEPS" --epochsteps "$EPOCHSTEPS"
            --seed 0 --out "$OUT_DIR/$name")
    if [ "$mode" = safe ]; then
      "$PYTHON" -u src/safehold/def_fix_pool.py "${common[@]}" \
        --mode anchor_path --anchor_dir m2r --lam 0.6 --lc 1.0
    else
      "$PYTHON" -u src/safehold/def_fix_pool.py "${common[@]}" --mode none
    fi
  done
done
