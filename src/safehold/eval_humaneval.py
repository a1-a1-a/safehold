#!/usr/bin/env python3
"""HumanEval greedy pass@1 with the official OpenAI human_eval execution harness.

The official harness executes generated code. Run this evaluator in a disposable,
network-isolated environment with appropriate resource limits.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("MODEL_PATH", os.environ.get("SC_BASE", "")))
    ap.add_argument("--lora", default="")
    ap.add_argument("--tasks", default="data/evaluation_benchmarks/humaneval_164.jsonl")
    ap.add_argument("--n", type=int, default=164)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/humaneval_paper.json")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    try:
        from human_eval.data import read_problems, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError as exc:
        raise SystemExit("Install the official OpenAI human-eval package before using its harness") from exc
    problems = read_problems(problem_file=a.tasks)
    if len(problems) != a.n or len(problems) != 164:
        raise ValueError(f"expected exactly 164 HumanEval tasks, found {len(problems)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allrec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if a.tag in allrec:
        raise ValueError(f"tag {a.tag!r} already exists in {out}; choose a new tag")
    if not a.base:
        raise ValueError("--base or MODEL_PATH is required")
    import torch
    import common as C
    tok, model = C.load(base=a.base, lora=(a.lora or None))
    prompts = [p["prompt"] for p in problems.values()]
    outputs = []
    for start in range(0, len(prompts), 8):
        enc = tok(prompts[start:start + 8], return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            generated = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                       pad_token_id=tok.pad_token_id)
        outputs.extend(tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                       for row in generated)
    samples = [{"task_id": task_id, "completion": completion}
               for task_id, completion in zip(problems, outputs)]
    sample_file = out.with_name(out.name + f".samples_{a.tag}.jsonl")
    if sample_file.exists():
        raise FileExistsError(sample_file)
    write_jsonl(str(sample_file), samples)
    # Official check_correctness uses its own execution guard and timeouts.
    metrics = evaluate_functional_correctness(str(sample_file), k=[1],
                                               n_workers=a.workers, timeout=10.0,
                                               problem_file=a.tasks)
    rec = {"tag": a.tag, "n": len(samples), "pass1": metrics["pass@1"],
           "passed": round(metrics["pass@1"] * len(samples)),
           "samples": str(sample_file), "results": str(sample_file) + "_results.jsonl",
           "harness": "openai/human-eval", "do_sample": False}
    allrec[a.tag] = rec
    out.write_text(json.dumps(allrec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("RESULT", json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
