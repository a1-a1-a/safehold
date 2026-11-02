#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
DATA_DIR="${DATA_DIR:-data}"
OUT_DIR="${OUT_DIR:-outputs}"
RESULT_DIR="${RESULT_DIR:-results}"
TAG="${TAG:-model}"
PYTHON="${PYTHON:-python}"
mkdir -p "$RESULT_DIR"

EVAL_DIR="${EVAL_DIR:-data/eval}"
RMC="$EVAL_DIR/rmc178.jsonl"
HB="$EVAL_DIR/hb_cyber.jsonl"
MAL="$EVAL_DIR/mal215.jsonl"
XST="${XSTEST_FILE:-$EVAL_DIR/xstest.jsonl}"

for frac in 000 025 050 075 100; do
  for mode in undef safe; do
    name="${TAG}_f${frac}_${mode}"
    ad="$OUT_DIR/$name"
    [ -d "$ad" ] || { echo "skip missing $ad"; continue; }
    for spec in "rmc $RMC 178" "hb $HB 198" "mal $MAL 215"; do
      set -- $spec; key=$1; harm=$2; n=$3
      "$PYTHON" src/safehold/eval_asr.py --lora "$ad" --tag "${name}_${key}" \
        --harm "$harm" --n "$n" --out "$RESULT_DIR/asr.json"
    done
    "$PYTHON" src/safehold/eval_xstest.py --lora "$ad" --tag "${name}_xst" \
      --n 249 --out "$RESULT_DIR/xstest.json"
    HUMANEVAL_ARROW="${HUMANEVAL_ARROW:-}" "$PYTHON" src/safehold/eval_humaneval.py \
      --lora "$ad" --tag "${name}_he" --n 164 --out "$RESULT_DIR/humaneval.json"
  done
done
