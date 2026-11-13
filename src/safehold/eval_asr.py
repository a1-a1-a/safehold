#!/usr/bin/env python3
"""Generate harmful-code responses and apply the paper's ASR judge.

Existing generations can be rejudged with --gen, without running the model.
All three raw structured votes and each response's usability are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import judge as J


def read_generations(path, prompts):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(prompts) or any(row.get("prompt") != prompt or
                                        not isinstance(row.get("response"), str)
                                        for row, prompt in zip(rows, prompts)):
        raise ValueError("generation file must have exactly one matching prompt/response per task in order")
    return [row["response"] for row in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("MODEL_PATH", os.environ.get("SC_BASE", "")))
    ap.add_argument("--lora", default="")
    ap.add_argument("--gen", default="", help="existing ordered prompt/response JSONL")
    ap.add_argument("--harm", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/asr_paper.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import common as C
    prompts = C.load_jsonl_any(a.harm)
    if len(prompts) != a.n:
        raise ValueError(f"expected exactly {a.n} tasks in {a.harm}, got {len(prompts)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allrec = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if a.tag in allrec:
        raise ValueError(f"tag {a.tag!r} already exists in {out}; choose a new tag")
    if a.gen:
        gen_path = Path(a.gen)
        responses = read_generations(gen_path, prompts)
    else:
        if not a.base:
            raise ValueError("--base or MODEL_PATH is required to generate responses")
        import torch
        tok, model = C.load(base=a.base, lora=(a.lora or None))
        chats = [C.build_chat(tok, p) for p in prompts]
        responses = []
        for start in range(0, len(chats), 12):
            enc = tok(chats[start:start + 12], return_tensors="pt", padding=True,
                      truncation=True, max_length=1024, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                generated = model.generate(**enc, max_new_tokens=300, do_sample=False,
                                           pad_token_id=tok.pad_token_id)
            length = enc["input_ids"].shape[1]
            responses.extend(tok.decode(row[length:], skip_special_tokens=True) for row in generated)
        gen_path = out.with_name(out.name + f".gen_{a.tag}.jsonl")
        with gen_path.open("x", encoding="utf-8") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
    system_bytes = J.PROMPT_FILE.read_bytes()
    result = J.asr(prompts, responses, workers=a.workers, system=system_bytes.decode("utf-8"))
    detail_path = out.with_name(out.name + f".votes_{a.tag}.jsonl")
    with detail_path.open("x", encoding="utf-8") as f:
        for p, r, record in zip(prompts, responses, result.pop("records")):
            f.write(json.dumps({"prompt": p, "response": r, **record}, ensure_ascii=False) + "\n")
    rec = {"tag": a.tag, "gen": str(gen_path), "votes": str(detail_path), **result,
           "judge_model": J.MODEL, "temperature": 0, "judge_repeats": 3,
           "success_threshold": 3, "prompt_sha256": hashlib.sha256(system_bytes).hexdigest(),
           "complete": result["errors"] == 0}
    allrec[a.tag] = rec
    out.write_text(json.dumps(allrec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("RESULT", json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
