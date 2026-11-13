#!/usr/bin/env bash
# Extract gated-MLP features one layer at a time, then aggregate the selections.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
PYTHON="${PYTHON:-python}"
TAG="${TAG:-model}"
LAYERS="${LAYERS:-28}"
OUT_DIR="${OUT_DIR:-results}"
mkdir -p "$OUT_DIR"
for ((layer = 0; layer < LAYERS; layer++)); do
  out="$OUT_DIR/${TAG}_L${layer}_2400.npz"
  if [ -e "$out" ] || [ -e "${out%.npz}.json" ]; then
    echo "skip ${TAG} layer $layer"; continue
  fi
  "$PYTHON" scripts/internal_diagnostics.py neurons --base "$MODEL_PATH" \
    --layer "$layer" --out "$out"
done
"$PYTHON" scripts/aggregate_neurons.py "$OUT_DIR/${TAG}_L"*_2400.npz \
  --out "$OUT_DIR/${TAG}_neurons_2400"
