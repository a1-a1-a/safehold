#!/usr/bin/env bash
# Fixed, cyclic, probe-selection, and refusal-template groups.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
PYTHON="${PYTHON:-python}"
TAG="${TAG:-model}"
DATA_PATH="${DATA_PATH:-data/training_corpora/llm_lat_300.json}"
POOL_DIR="${POOL_DIR:-data/probe_pool}"
FIXED_POOL="${FIXED_POOL:-data/fixed_safety_supervision/advbench_probes8.jsonl}"
OUT_DIR="${OUT_DIR:-outputs}"
STAGE="${1:-all}"
mkdir -p "$OUT_DIR"

train() {
  local name="$1" pool="$2" poolsize="$3" sched="$4" maxsteps="$5" epochsteps="$6" safe="${7:-}"
  local dest="$OUT_DIR/$name"
  if [ -f "$dest/adapter_config.json" ]; then
    echo "skip $name"
    return
  fi
  local args=(--model "$MODEL_PATH" --data "$DATA_PATH" --pool "$pool" --poolsize "$poolsize"
              --k 8 --sched "$sched" --maxsteps "$maxsteps" --epochsteps "$epochsteps"
              --seed 0 --out "$dest" --mode anchor_path --anchor_dir m2r --lam 0.6 --lc 1.0)
  if [ -n "$safe" ]; then
    args+=(--safe_file "$safe")
  fi
  echo "train $name"
  "$PYTHON" -u src/safehold/def_fix_pool.py "${args[@]}"
}

if [ "$STAGE" = "fixed" ] || [ "$STAGE" = "all" ]; then
  train "${TAG}_pool_fixed" "$FIXED_POOL" 8 round 600 300
  train "${TAG}_pool_cyclic" "$POOL_DIR/probe_pool_188.jsonl" 188 cyclic 600 300
fi

if [ "$STAGE" = "selection" ] || [ "$STAGE" = "all" ]; then
  train "${TAG}_probe_draw1" "$POOL_DIR/draw_seed1.jsonl" 8 round 600 0
  train "${TAG}_probe_draw2" "$POOL_DIR/draw_seed11.jsonl" 8 round 600 0
  train "${TAG}_probe_draw3" "$POOL_DIR/draw_seed22.jsonl" 8 round 600 0
  train "${TAG}_probe_template" "$POOL_DIR/template_replacement.jsonl" 8 round 600 0 \
    "$POOL_DIR/refusal_target_alternate.txt"
fi
