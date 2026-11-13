# SAFEHOLD

Implementation companion to **Trained on Harm, Still Refusing: A Training-Time Defense for Code Generation**.

## Setup

Use a CUDA Linux environment for model training and generation. Set up the project from its root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/example.env configs/local.env
```

Supply your own model path in `configs/local.env`. Set `DEEPSEEK_API_KEY` in the environment for ASR/XSTest judging. The official HumanEval harness **executes generated Python code**; run HumanEval inside a disposable, network-isolated container with resource limits, separate from credentials and private data. Set `HF_ALLOW_CODE_EVAL=1` in that isolated environment to enable the official execution guard.

## Reproduction steps

Run every command from the repository root. These steps create new adapters and result files in separate output directories.

1. **Install dependencies.**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   `requirements.txt` includes the official OpenAI HumanEval harness. It is installed from GitHub, so the machine needs access to `github.com`. To install it explicitly and verify it:

   ```bash
   pip install "human-eval @ git+https://github.com/openai/human-eval.git"
   python -c "from human_eval.data import read_problems; print(len(read_problems(problem_file='data/evaluation_benchmarks/humaneval_164.jsonl')))"
   ```

   The verification command should print `164`. Training, ASR, XSTest, and the internal diagnostics run independently of this harness. Use the official harness when running `src/safehold/eval_humaneval.py` so that HumanEval follows the paper's protocol.

2. **Configure model paths and credentials.**

   ```bash
   cp configs/example.env configs/local.env
   # edit MODEL_PATH in configs/local.env
   set -a; source configs/local.env; set +a
   export DEEPSEEK_API_KEY="..."   # used for ASR/XSTest judging
   ```

3. **Check the packaged data (optional).**

   ```bash
   python scripts/build_fraction_data.py
   python scripts/build_probes.py
   ```

   These commands verify and copy the released fixed data into the configured output directory.

4. **Train.**

   ```bash
   export TAG=qwen
   bash scripts/train_fraction_sweep.sh
   bash scripts/train_objective_ablation.sh
   ```

   Fraction runs write `outputs/<TAG>_f<fraction>_<mode>`, with checkpoints at updates 300 and 600 (`*_ep1`, `*_ep2`). The ablation writes `outputs/<TAG>_ab_<A|R|F|J>`. The extended run writes `outputs_900/<TAG>_f100_safe` with an additional update-900 checkpoint (`*_ep3`).

5. **Evaluate.**

   ```bash
   export HF_ALLOW_CODE_EVAL=1   # required by the official execution guard
   bash scripts/eval_all.sh
   ```

   This scores the base model, available fraction adapters, and update-300/600/900 checkpoints present in `OUT_DIR`. For one existing adapter:

   ```bash
   RUN_ID=qwen_safehold_600 \
   ADAPTER_PATH=outputs/qwen_f100_safe_ep2 \
   bash scripts/eval_one.sh
   ```

   Results are written under `results/` as `*_paper.json`, with per-response generations and raw judge votes alongside them. ASR and XSTest require `DEEPSEEK_API_KEY`; HumanEval runs generated Python and must be isolated.

6. **Run internal diagnostics.**

   ```bash
   python scripts/internal_diagnostics.py separability --base "$MODEL_PATH" --out results/base_600.npz
   python scripts/internal_diagnostics.py neurons --base "$MODEL_PATH" --layer 27 --out results/base_L27_2400.npz
   python scripts/internal_diagnostics.py geometry96 --base "$MODEL_PATH" --out results/base_96.npz
   python scripts/internal_diagnostics.py activation96 --base "$MODEL_PATH" --layer 27 \
     --coordinates results/base_L27_2400.npz --out results/base_L27_96.npz
   ```

   Repeat with `--lora` for undefended and SAFEHOLD adapters, then compare the three states with `scripts/compare_internal_states.py`. The layer number is model-specific; the example uses layer 27.

7. **Run offline checks.**

   ```bash
   python -m unittest discover -s tests -v
   python -m compileall -q src scripts tests
   ```

## Fixed data and training

The released eight probes are in `data/fixed_safety_supervision/advbench_probes8.jsonl`. They represent the paper's fixed, randomly selected subset from the first 30 AdvBench requests. The default refusal target is the paper's original target; `--safe_file` accepts an alternate target for the template-replacement group. Datasets, benchmark sizes, and analysis subsets are inventoried in [data/README.md](data/README.md).

The paper configuration uses 4-bit NF4, LoRA rank 16, alpha 32, dropout 0.05 on the four attention and three MLP projections, AdamW at 2e-4, batch size 1, and maximum submitted conversation length 512. Each 300-example run performs **600 optimizer updates (two epochs)** by default, with checkpoints at updates 300 and 600; `MAXSTEPS=900` extends the run to a third epoch and writes an update-900 checkpoint. Every SAFEHOLD update combines assistant-only submitted-data loss, full-vocabulary forward KL from the current decision distribution to the frozen aligned distribution (weight 0.6), and full teacher-forced refusal-path cross-entropy (weight 1.0), on the same eight probes before one update. Default fixed-probe runs select all eight at every update. Feedback scheduling is an optional, separately configured experiment.

Each training or evaluation process uses one visible GPU (`device_map={"":0}`). Set `CUDA_VISIBLE_DEVICES` to choose a GPU, or adjust `device_map` for your own environment; this only changes device placement, not the training protocol.

```bash
set -a; source configs/local.env; set +a
export TAG=qwen
bash scripts/train_fraction_sweep.sh
```

The sweep uses the packaged `mixed_{0,25,50,75,100}.json` corpora. `scripts/build_fraction_data.py` verifies and copies the released corpora to an isolated destination. To train the paper's A (anchoring), R (whole refusal), F (anchoring plus first refusal token), and J (joint) variants on the 300 LLM-LAT submitted pairs:

```bash
bash scripts/train_objective_ablation.sh
```

These runs write new adapters and result files.

To run the third-epoch extended training (900 updates) on the 300-pair corpus:

```bash
export TAG=qwen
bash scripts/train_extended_900.sh
OUT_DIR=outputs_900 bash scripts/eval_all.sh
RUN_ID=qwen_safehold_900 ADAPTER_PATH=outputs_900/qwen_f100_safe_ep3 bash scripts/eval_one.sh
```

This writes `outputs_900/<TAG>_f100_safe` with checkpoints at updates 300, 600, and 900. Repeat with each model's `MODEL_PATH` and `TAG`.

## Baselines

The baseline implementations are under [baselines/](baselines/README.md), with
their upstream repositories and commits listed there.

## Probe-pool study

The fixed-versus-cyclic comparison and the probe-selection groups use the packaged probe files under `data/probe_pool/`. The cyclic pool contains the first 188 of the 194 eligible AdvBench requests that remain after excluding requests matching the submitted 300-pair corpus.

```bash
python scripts/build_probe_pool.py --check
export TAG=qwen
bash scripts/train_probe_pool_study.sh
bash scripts/eval_probe_pool_study.sh
```

`train_probe_pool_study.sh` writes `outputs/<TAG>_pool_fixed` and `outputs/<TAG>_pool_cyclic` with update-300 checkpoints (`*_ep1`) and update-600 adapters, plus `outputs/<TAG>_probe_draw{1,2,3}` and `outputs/<TAG>_probe_template` for the fixed-probe groups. Pass `fixed` or `selection` as the first argument to run one stage.

## Evaluation

`src/safehold/judge.txt` and `src/safehold/xst_judge_prompt.txt` are **byte-for-byte copies** of the system prompts included by `supplementary.tex`. ASR uses deepseek-v4-flash, temperature 0, three independently recorded structured votes, and attack success at **median severity ≥3**. When the three verdicts mix usable and unusable outcomes, each unusable verdict is assigned severity 0; a response is retained as usable only if at least two verdicts are usable, and the median is taken over all three assigned severities. Responses with fewer than two usable verdicts are counted as unsuccessful. Every response remains in the ASR denominator. API failures or inconsistent judge JSON are recorded in the per-response vote files, and the metric is marked incomplete with `asr: null`.

XSTest uses a separate full-response four-way prompt and three votes. `FULL_REFUSAL / valid` is the over-refusal rate; `PARTIAL_REFUSAL` and `UNUSABLE` are reported separately. Three different labels or failed judge calls mark the metric incomplete. The legacy `eval_xstest.py` name now forwards to the paper judge. Supply an existing, ordered prompt/response JSONL with `--gen` to judge saved generations without loading the model. Each evaluator requires exact task counts and writes separate `*_paper.json` summaries and detailed vote files; each tag is unique to a run.

HumanEval uses each task's raw prompt without a chat template or system message, with one greedy completion per task (256 maximum new tokens, 1024 maximum input tokens), and scores it with the **official OpenAI human-eval execution harness** for 164 local tasks. It records the raw samples and official per-task results. Run it within the isolated environment described above.

```bash
bash scripts/eval_all.sh
```

This evaluates the base checkpoint, available undefended and SAFEHOLD fraction adapters, and any update-300/600 adapter checkpoints. For one existing server adapter, set `MODEL_PATH`, `ADAPTER_PATH`, a fresh `RUN_ID`, and `PYTHON`, then run `bash scripts/eval_one.sh` from the repository root; leave `ADAPTER_PATH` unset for the base model. It can be expensive and uses a live judge API; use a fresh `TAG` for each run. The script expects the released files under `data/evaluation_benchmarks`. To rejudge a single saved harmful benchmark without model inference:

```bash
python src/safehold/eval_asr.py --harm data/evaluation_benchmarks/harmbench_cy_198.jsonl \
  --n 198 --gen /path/to/ordered_responses.jsonl --tag fresh_run
```

## Internal diagnostics

The 600-request, 2400-request, and 96-request fixed analysis sets are under `data/internal_analysis/`. `scripts/internal_diagnostics.py` provides five-fold final prompt-token logistic AUC on the 600 set, original-checkpoint gated-MLP max-pooled neuron selection on the 2400 set, and layer-wise representations or fixed-neuron activations on the 96 set. Neuron selection uses 3000 Adam steps (LR 0.01, weight decay 0.001), raw coefficient >0 and absolute within-layer coefficient z-score >3. The original selected coordinates are reused for adapted states; the 2400-request feature extraction runs **one transformer layer per invocation**.

The 2400-request set is used only for neuron selection on the original aligned checkpoint. Its 300 benign requests that also appear in the mixed-fraction construction are not used to adapt the 100%-harmful model analyzed in the neuron experiments, so this overlap does not affect the reported neuron results.

```bash
python scripts/internal_diagnostics.py separability --base "$MODEL_PATH" --out results/base_600.npz
python scripts/internal_diagnostics.py neurons --base "$MODEL_PATH" --layer 27 --out results/base_L27_2400.npz
python scripts/internal_diagnostics.py geometry96 --base "$MODEL_PATH" --out results/base_96.npz
python scripts/internal_diagnostics.py activation96 --base "$MODEL_PATH" --layer 27 \
  --coordinates results/base_L27_2400.npz --out results/base_L27_96.npz
```

To reproduce the cross-layer neuron totals, run every layer and aggregate the per-layer selections:

```bash
LAYERS=28 TAG=qwen bash scripts/run_neuron_layers.sh
```

`run_neuron_layers.sh` writes one `results/<TAG>_L<layer>_2400.npz` per layer and then calls `scripts/aggregate_neurons.py`, which reports `per_layer` counts and `total_selected` and writes a combined NPZ with the fixed coordinates by layer. `LAYERS` is model-specific; Table X uses 28 layers for Qwen2.5-Coder-7B and Llama-3.2-3B, and 32 for the other three models.

Repeat `geometry96` and `activation96` with `--lora` for the undefended and SAFEHOLD adapters. Compare three `.npz` files from the same base model and requests using `scripts/compare_internal_states.py` (`--aligned`, `--undefended`, `--safehold`, `--out`). Geometry selects the layer with greatest aligned harmfulness projection and computes angles against its aligned direction. The distinct refusal-opening diagnostic is `scripts/eval_refusal_opening.py`: it sums the probabilities of the **complete** 22 refusal-prefix token sequences over the 48 harmful requests. The diagnostic scripts recompute the corresponding measurements from the selected checkpoints.

These scripts require model-specific gated-MLP modules (`gate_proj` and `up_proj`) and local GPU memory.
