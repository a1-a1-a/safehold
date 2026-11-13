#!/usr/bin/env bash
# Score the fixed/cyclic and probe-selection adapters with the paper protocol.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
PYTHON="${PYTHON:-python}"
TAG="${TAG:-model}"
OUT_DIR="${OUT_DIR:-outputs}"
RESULT_DIR="${RESULT_DIR:-results}"
export PYTHON RESULT_DIR
mkdir -p "$RESULT_DIR"

for name in \
  "${TAG}_pool_fixed_ep1" "${TAG}_pool_fixed" \
  "${TAG}_pool_cyclic_ep1" "${TAG}_pool_cyclic" \
  "${TAG}_probe_draw1" "${TAG}_probe_draw2" "${TAG}_probe_draw3" "${TAG}_probe_template"; do
  adapter="$OUT_DIR/$name"
  if [ ! -f "$adapter/adapter_config.json" ]; then
    echo "skip $name"
    continue
  fi
  ADAPTER_PATH="$adapter" RUN_ID="$name" bash scripts/eval_one.sh
done
