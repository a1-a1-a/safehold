# Datasets

Datasets are organized by their role in the paper.

## Fixed safety supervision

| File | Size | Role |
|---|---:|---|
| `fixed_safety_supervision/advbench_probes8.jsonl` | 8 probes | Baseline fixed trusted AdvBench probes used for safety supervision |
| `probe_groups/draw_seed{1,11,22}.jsonl` | 8 probes each | Alternative fixed-probe selections |
| `probe_groups/template_replacement.jsonl` | 8 probes | Draw 1 probes with the alternate refusal target |
| `probe_groups/refusal_target_{original,alternate}.txt` | 1 target each | Shared refusal targets |

Only these probe files are redistributed. The full AdvBench collection and the 194/188-entry derived pools are not included.

## Refusal-opening read-out

| File | Size | Role |
|---|---:|---|
| `refusal_prefixes.txt` | 22 prefixes | Refusal prefixes used by the 22-prefix read-out in Table 12 and Eq. (C.10) |

## Training corpora

| File | Size | Role |
|---|---:|---|
| `training_corpora/llm_lat_300.json` | 300 pairs | Main harmful fine-tuning corpus (LLM-LAT) |
| `training_corpora/cyberstrike_300.json` | 300 pairs | Code-domain harmful fine-tuning corpus (CyberStrike) |
| `training_corpora/cysecbench_300.json` | 300 pairs | Code-domain harmful fine-tuning corpus (CySecBench) |
| `training_corpora/benign_300.json` | 300 pairs | Benign corpus used to construct mixed submissions |
| `training_corpora/mixed_fractions/mixed_{0,25,50,75,100}.json` | 300 pairs each | Mixed harmful/benign corpora used in the fraction analysis |

The mixed corpora contain 0, 75, 150, 225, and 300 harmful examples, respectively. Adjacent fractions share 225 prompts and replace 75 prompts; they are not nested subsets. `mixed_0` is exactly `benign_300.json`, and `mixed_100` is exactly `llm_lat_300.json`.

## Evaluation benchmarks

| File | Size | Metric |
|---|---:|---|
| `evaluation_benchmarks/rmcbench_178.jsonl` | 178 tasks | Harmful-code ASR |
| `evaluation_benchmarks/harmbench_cy_198.jsonl` | 198 tasks | Harmful-code ASR |
| `evaluation_benchmarks/malwarebench_215.jsonl` | 215 tasks | Harmful-code ASR |
| `evaluation_benchmarks/xstest_249.jsonl` | 249 tasks | Benign over-refusal |
| `evaluation_benchmarks/humaneval_164.jsonl` | 164 tasks | Code-generation pass@1 |

## Internal analysis

| File | Size | Role |
|---|---:|---|
| `internal_analysis/cysecbench_600.jsonl` | 300 harmful + 300 benign | Harmfulness separability |
| `internal_analysis/cysecbench_2400.jsonl` | 1,200 harmful + 1,200 benign | Neuron identification |
| `internal_analysis/cysecbench_96.jsonl` | 48 harmful + 48 benign | Activation, refusal, and geometry analyses |

The 600- and 96-request sets are disjoint from the training corpora. The
2,400-request set is used only for neuron selection on the original aligned
checkpoint. Its first 300 benign requests also appear in
`training_corpora/benign_300.json`, which is used only to construct the
mixed-fraction corpora (`mixed_0`--`mixed_75`). The neuron analysis uses the
100%-harmful LLM-LAT adaptation, which contains no benign examples; no 2,400
request is used to adapt that model or as safety supervision. The overlap
therefore does not affect the reported neuron analysis.

## Provenance and licensing

- AdvBench: eight fixed trusted probes only.
- LLM-LAT: general-domain harmful fine-tuning corpus.
- CyberStrike and CySecBench: code-domain harmful fine-tuning corpora.
- RMCBench, HarmBench_cy, MalwareBench, XSTest, and HumanEval: evaluation benchmarks.

Please follow the licenses and terms of the original dataset releases.
