#!/usr/bin/env bash
# Score one already-trained checkpoint with the paper protocol.
set -euo pipefail
: "${MODEL_PATH:?set MODEL_PATH}"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY}"
: "${RUN_ID:?set a new RUN_ID for the scored checkpoint}"
PYTHON="${PYTHON:-python}"
EVAL_DIR="${EVAL_DIR:-data/evaluation_benchmarks}"
RESULT_DIR="${RESULT_DIR:-results}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
mkdir -p "$RESULT_DIR"
args=(--base "$MODEL_PATH")
if [ -n "$ADAPTER_PATH" ]; then
  [ -f "$ADAPTER_PATH/adapter_config.json" ] || { echo "adapter missing: $ADAPTER_PATH" >&2; exit 1; }
  args+=(--lora "$ADAPTER_PATH")
fi
"$PYTHON" src/safehold/eval_asr.py "${args[@]}" --harm "$EVAL_DIR/rmcbench_178.jsonl" \
  --n 178 --tag "${RUN_ID}_rmc" --out "$RESULT_DIR/asr_paper.json"
"$PYTHON" src/safehold/eval_asr.py "${args[@]}" --harm "$EVAL_DIR/harmbench_cy_198.jsonl" \
  --n 198 --tag "${RUN_ID}_hb" --out "$RESULT_DIR/asr_paper.json"
"$PYTHON" src/safehold/eval_asr.py "${args[@]}" --harm "$EVAL_DIR/malwarebench_215.jsonl" \
  --n 215 --tag "${RUN_ID}_mal" --out "$RESULT_DIR/asr_paper.json"
"$PYTHON" src/safehold/eval_xstest_judge.py "${args[@]}" --xst "$EVAL_DIR/xstest_249.jsonl" \
  --n 249 --tag "${RUN_ID}_xst" --out "$RESULT_DIR/xstest_paper.json"
"$PYTHON" src/safehold/eval_humaneval.py "${args[@]}" --tasks "$EVAL_DIR/humaneval_164.jsonl" \
  --n 164 --tag "${RUN_ID}_he" --out "$RESULT_DIR/humaneval_paper.json"
