# SBR

`paper_protocol.py` is the included implementation of the SBR objective.

```bash
python baselines/SBR/paper_protocol.py \
  --model "$MODEL_PATH" \
  --data data/training_corpora/mixed_fractions/mixed_100.json \
  --probes data/fixed_safety_supervision/advbench_probes8.jsonl \
  --out outputs_baseline/sbr
```
