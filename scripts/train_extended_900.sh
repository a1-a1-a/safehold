#!/usr/bin/env bash
# SAFEHOLD third-epoch training: 900 updates on the 300-pair corpus.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FRACTIONS="${FRACTIONS:-100}"
export MODES="${MODES:-safe}"
export MAXSTEPS="${MAXSTEPS:-900}"
export EPOCHSTEPS="${EPOCHSTEPS:-300}"
export OUT_DIR="${OUT_DIR:-outputs_900}"
exec bash "$SCRIPT_DIR/train_fraction_sweep.sh"
