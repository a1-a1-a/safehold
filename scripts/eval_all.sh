#!/usr/bin/env bash
# Paper evaluation; separate result files preserve all earlier metrics.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
OUT_DIR="${OUT_DIR:-outputs}"
RESULT_DIR="${RESULT_DIR:-results}"
TAG="${TAG:-model}"
PYTHON="${PYTHON:-python}"
EVAL_DIR="${EVAL_DIR:-data/evaluation_benchmarks}"
mkdir -p "$RESULT_DIR"

run_eval() {
  local name="$1" ad="$2" base_args=()
  if [ -n "$ad" ]; then base_args=(--lora "$ad"); fi
  "$PYTHON" src/safehold/eval_asr.py --base "$MODEL_PATH" "${base_args[@]}" \
    --tag "${name}_rmc" --harm "$EVAL_DIR/rmcbench_178.jsonl" --n 178 --out "$RESULT_DIR/asr_paper.json"
  "$PYTHON" src/safehold/eval_asr.py --base "$MODEL_PATH" "${base_args[@]}" \
    --tag "${name}_hb" --harm "$EVAL_DIR/harmbench_cy_198.jsonl" --n 198 --out "$RESULT_DIR/asr_paper.json"
  "$PYTHON" src/safehold/eval_asr.py --base "$MODEL_PATH" "${base_args[@]}" \
    --tag "${name}_mal" --harm "$EVAL_DIR/malwarebench_215.jsonl" --n 215 --out "$RESULT_DIR/asr_paper.json"
  "$PYTHON" src/safehold/eval_xstest_judge.py --base "$MODEL_PATH" "${base_args[@]}" \
    --tag "${name}_xst" --xst "$EVAL_DIR/xstest_249.jsonl" --n 249 --out "$RESULT_DIR/xstest_paper.json"
  "$PYTHON" src/safehold/eval_humaneval.py --base "$MODEL_PATH" "${base_args[@]}" \
    --tag "${name}_he" --tasks "$EVAL_DIR/humaneval_164.jsonl" --n 164 --out "$RESULT_DIR/humaneval_paper.json"
}

run_eval "${TAG}_base" ""
for frac in 000 025 050 075 100; do
  for mode in undef safe; do
    name="${TAG}_f${frac}_${mode}"
    ad="$OUT_DIR/$name"
    if [ ! -f "$ad/adapter_config.json" ]; then echo "skip missing $ad"; continue; fi
    run_eval "$name" "$ad"
    for ep in 1 2 3; do
      ck="${ad}_ep${ep}"
      if [ -f "$ck/adapter_config.json" ]; then run_eval "${name}_ep${ep}" "$ck"; fi
    done
  done
done
