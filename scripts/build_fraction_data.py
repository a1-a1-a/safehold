#!/usr/bin/env python3
"""Build deterministic 300-example corpora with 0/25/50/75/100% harmful data."""
import json, os, random, re, hashlib

HARM = os.environ.get("HARM_FILE", "data/raw/harmful_train.json")
BENIGN = os.environ.get("BENIGN_FILE", "data/raw/benign_train.json")
PROBES = os.environ.get("PROBE_FILE", "data/advbench_probes8.jsonl")
EVAL_DIR = os.environ.get("EVAL_DIR", "data/eval")
OUT = os.environ.get("DATA_DIR", "data")
SEED = int(os.environ.get("DATA_SEED", "0"))
FRACTIONS = [0, 25, 50, 75, 100]

def prompt(x):
    for t in x["conversations"]:
        if t.get("from") in ("human", "user"):
            return t["value"]
    return ""

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def load_eval_prompts():
    out = set()
    for name in ("rmc178.jsonl", "hb_cyber.jsonl", "mal215.jsonl", "xstest.jsonl"):
        path = os.path.join(EVAL_DIR, name)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ("text", "prompt", "question", "goal", "behavior", "instruction"):
                if r.get(k):
                    out.add(norm(str(r[k])))
                    break
    return out

def main():
    harm = json.load(open(HARM, encoding="utf-8"))
    ben = json.load(open(BENIGN, encoding="utf-8"))
    probes = [json.loads(l)["q"] for l in open(PROBES, encoding="utf-8") if l.strip()]
    blocked = {norm(q) for q in probes} | load_eval_prompts()
    harm = [x for x in harm if norm(prompt(x)) not in blocked]
    ben = [x for x in ben if norm(prompt(x)) not in blocked]
    if len(harm) < 300 or len(ben) < 300:
        raise SystemExit(f"need at least 300 harmful and 300 benign examples, got {len(harm)} and {len(ben)}")
    rng = random.Random(SEED)
    rng.shuffle(harm)
    rng.shuffle(ben)
    os.makedirs(OUT, exist_ok=True)
    for frac in FRACTIONS:
        nh = int(round(300 * frac / 100))
        nb = 300 - nh
        rows = harm[:nh] + ben[:nb]
        rng2 = random.Random(SEED + 100 + frac)
        rng2.shuffle(rows)
        ps = [norm(prompt(x)) for x in rows]
        if len(ps) != 300 or len(set(ps)) != 300:
            raise SystemExit(f"duplicate/incorrect size for {frac}%")
        out = os.path.join(OUT, f"fraction_{frac:03d}.json")
        json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        meta = {"n_total": 300, "n_harmful": nh, "n_benign": nb, "seed": SEED,
                "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
        json.dump(meta, open(out.replace(".json", "_meta.json"), "w", encoding="utf-8"), indent=2)
        print(out, meta)

if __name__ == "__main__":
    main()
