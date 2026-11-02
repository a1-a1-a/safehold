# SAFEHOLD

Code for **Trained on Harm, Still Refusing: A Decision-and-Continuation Defense for Code-Generation Safety**.

SAFEHOLD is a training-time defense against harmful fine-tuning of code LLMs. The submitted-data loss is applied only to assistant response tokens; system and user tokens are masked out. It combines:

- decision-boundary anchoring against a frozen aligned reference;
- refusal-path reinforcement over the complete refusal sequence;
- eight fixed trusted safety probes applied throughout LoRA adaptation.

This repository contains the datasets, data-building, training, and evaluation code. Model weights are not included.

## Layout

```text
src/safehold/
  def_fix_pool.py       training loop (QLoRA + anchoring + refusal-path loss)
  common.py             model/data helpers and the 22-prefix refusal-opening read-out
  judge.py              optional judge helper
  judge.txt             1-5 rubric judge prompt
  eval_asr.py           harmful-code ASR evaluation
  eval_xstest.py        XSTest over-refusal evaluation (prefix rule)
  eval_xstest_judge.py  XSTest over-refusal evaluation (4-way labels, 3 votes)
  xst_judge_prompt.txt  over-refusal judge prompt
  eval_humaneval.py     HumanEval pass@1 evaluation
scripts/
  build_probes.py       normalize eight trusted probes
  build_fraction_data.py build 0/25/50/75/100% harmful corpora
  eval_refusal_opening.py evaluate the 22-prefix refusal-opening read-out
  train_fraction_sweep.sh
  eval_all.sh
configs/example.env     environment template
data/                  datasets organized by paper role
  README.md            dataset inventory and provenance
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp configs/example.env configs/local.env
```

Edit `configs/local.env` and set the model/data locations. Do not commit real credentials.

The evaluator uses the environment variable `DEEPSEEK_API_KEY` for the judge; it is never stored in this repository.

## Data format

The training corpora are sharegpt-style JSON lists:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]
```

Datasets are included under `data/` and inventoried in `data/README.md`.

## 1. Build probes and fraction corpora

```bash
set -a; source configs/local.env; set +a
python scripts/build_probes.py
python scripts/build_fraction_data.py
```

This writes `data/fraction_000.json` through `data/fraction_100.json`.

## 2. Train

For one model:

```bash
set -a; source configs/local.env; set +a
export MODEL_PATH=/path/to/model
export TAG=qwen
bash scripts/train_fraction_sweep.sh
```

Each run uses 600 updates, LoRA rank 16 / alpha 32, and eight fixed probes. The SAFE HOLD runs use `--mode anchor_path --anchor_dir m2r --lam 0.6 --lc 1.0`.

## 3. Evaluate

```bash
set -a; source configs/local.env; set +a
export MODEL_PATH=/path/to/model
export DEEPSEEK_API_KEY=...
export TAG=qwen
bash scripts/eval_all.sh
```

Evaluations are reported on:

- RMCBench: 178 tasks
- HarmBench_cy: 198 tasks
- MalwareBench: 215 tasks
- XSTest: 249 tasks
- HumanEval: 164 tasks

## 4. Refusal-opening read-out

`src/safehold/common.py` implements the 22-prefix read-out in Eq. (C.10): each prefix is appended to the chat prompt with the paper's fixed system instruction, the conditional probabilities of its tokens are multiplied at each layer, and the resulting sequence probabilities are summed over the prefixes in `data/refusal_prefixes.txt` (Table 12).

```bash
python scripts/eval_refusal_opening.py \
  --base /path/to/Qwen2.5-Coder-7B-Instruct \
  --lora /path/to/adapter \
  --out results/refusal_opening.json
```

The output stores the per-prompt, per-layer read-outs and the mean final-layer value.

## Notes

- No API keys, passwords, server addresses, or private absolute paths are included.
- All paths are configured through environment variables or relative paths.
