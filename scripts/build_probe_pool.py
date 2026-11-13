#!/usr/bin/env python3
"""Build and verify the probe pools used by the probe-pool study."""
import argparse
import hashlib
import json
import os
import random

ORIGINAL = ("I'm sorry, but I can't help with that. This request asks for something harmful or unsafe, "
            "and I'm not able to assist with it.")
ALTERNATE = ("I cannot provide assistance with this request because it asks for harmful or unsafe content. "
             "I will not help with it.")
DRAW_SEEDS = (1, 11, 22)


def _prompt(row):
    for key in ("q", "text", "prompt", "question", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_prompts(path):
    rows = []
    seen = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = _prompt(json.loads(line))
            if not value or value in seen:
                continue
            seen.add(value)
            rows.append(value)
    return rows


def training_prompts(path):
    rows = json.load(open(path, encoding="utf-8"))
    return {turn["value"].strip().lower()
            for example in rows
            for turn in example["conversations"]
            if turn["from"] == "human"}


def build_advbench(advbench_source, train):
    prompts = load_prompts(advbench_source)
    if len(prompts) != 208:
        raise SystemExit("expected 208 unique AdvBench prompts, found %d" % len(prompts))
    eligible = [prompt for prompt in prompts if prompt.strip().lower() not in train]
    if len(eligible) != 194:
        raise SystemExit("expected 194 eligible AdvBench prompts, found %d" % len(eligible))
    indices = {prompt: i + 1 for i, prompt in enumerate(prompts)}
    return eligible, indices


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pools(out_dir, advbench_source, train_path):
    os.makedirs(out_dir, exist_ok=True)
    train = training_prompts(train_path)
    eligible, indices = build_advbench(advbench_source, train)
    pool = eligible[:188]
    write_jsonl(os.path.join(out_dir, "probe_pool_188.jsonl"),
                [{"q": prompt, "source": "AdvBench", "pool_index_1based": indices[prompt]}
                 for prompt in pool])
    write_jsonl(os.path.join(out_dir, "advbench_eligible_194.jsonl"),
                [{"q": prompt, "source": "AdvBench", "pool_index_1based": indices[prompt]}
                 for prompt in eligible])
    for seed in DRAW_SEEDS:
        probes = random.Random(seed).sample(eligible, 8)
        write_jsonl(os.path.join(out_dir, "draw_seed%d.jsonl" % seed),
                    [{"q": prompt, "source": "AdvBench", "draw_seed": seed}
                     for prompt in probes])
    draw1 = [row["q"] for row in read_jsonl(os.path.join(out_dir, "draw_seed1.jsonl"))]
    write_jsonl(os.path.join(out_dir, "template_replacement.jsonl"),
                [{"q": prompt, "refusal_template": ALTERNATE} for prompt in draw1])
    with open(os.path.join(out_dir, "refusal_target_original.txt"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(ORIGINAL + "\n")
    with open(os.path.join(out_dir, "refusal_target_alternate.txt"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(ALTERNATE + "\n")
    manifest = {
        "probe_pool_188": {
            "n": 188,
            "source": "AdvBench",
            "construction": [
                "remove exact duplicate requests",
                "remove 14 requests matching the submitted 300-pair corpus",
                "use the first 188 of the 194 remaining eligible requests in AdvBench order",
            ],
            "sha256": sha256(os.path.join(out_dir, "probe_pool_188.jsonl")),
        },
        "advbench_eligible_194": {
            "n": 194,
            "full_unique_prompts": 208,
            "training_overlap_removed": 14,
            "sha256": sha256(os.path.join(out_dir, "advbench_eligible_194.jsonl")),
        },
        "draws": {
            str(seed): {
                "n": 8,
                "sampling": "uniform sample without replacement",
                "sha256": sha256(os.path.join(out_dir, "draw_seed%d.jsonl" % seed)),
            }
            for seed in DRAW_SEEDS
        },
        "template_replacement": {
            "n": 8,
            "probe_source": "draw_seed1.jsonl",
            "sha256": sha256(os.path.join(out_dir, "template_replacement.jsonl")),
        },
        "refusal_targets": {
            "original": sha256(os.path.join(out_dir, "refusal_target_original.txt")),
            "alternate": sha256(os.path.join(out_dir, "refusal_target_alternate.txt")),
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def verify_pools(out_dir):
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    pool = read_jsonl(os.path.join(out_dir, "probe_pool_188.jsonl"))
    eligible = read_jsonl(os.path.join(out_dir, "advbench_eligible_194.jsonl"))
    if len(pool) != 188 or len({row["q"] for row in pool}) != 188:
        raise SystemExit("probe_pool_188.jsonl must contain 188 unique requests")
    if len(eligible) != 194 or len({row["q"] for row in eligible}) != 194:
        raise SystemExit("advbench_eligible_194.jsonl must contain 194 unique requests")
    eligible_prompts = [row["q"] for row in eligible]
    if [row["q"] for row in pool] != eligible_prompts[:188]:
        raise SystemExit("probe_pool_188.jsonl must be the first 188 eligible AdvBench requests")
    if any(row["source"] != "AdvBench" for row in pool):
        raise SystemExit("probe_pool_188.jsonl must contain AdvBench requests")
    for seed in DRAW_SEEDS:
        rows = read_jsonl(os.path.join(out_dir, "draw_seed%d.jsonl" % seed))
        if len(rows) != 8 or {row["q"] for row in rows} - set(eligible_prompts):
            raise SystemExit("draw_seed%d.jsonl is not an eight-probe draw from the eligible pool" % seed)
        if rows != [{"q": prompt, "source": "AdvBench", "draw_seed": seed}
                    for prompt in random.Random(seed).sample(eligible_prompts, 8)]:
            raise SystemExit("draw_seed%d.jsonl does not match seed %d" % (seed, seed))
    draw1 = [row["q"] for row in read_jsonl(os.path.join(out_dir, "draw_seed1.jsonl"))]
    template = read_jsonl(os.path.join(out_dir, "template_replacement.jsonl"))
    if [row["q"] for row in template] != draw1:
        raise SystemExit("template_replacement.jsonl must reuse the Draw 1 probes")
    if any(row["refusal_template"] != ALTERNATE for row in template):
        raise SystemExit("template_replacement.jsonl has the wrong refusal target")
    for name in ("probe_pool_188", "advbench_eligible_194"):
        if sha256(os.path.join(out_dir, name + ".jsonl")) != manifest[name]["sha256"]:
            raise SystemExit("%s hash mismatch" % name)
    for seed in DRAW_SEEDS:
        if sha256(os.path.join(out_dir, "draw_seed%d.jsonl" % seed)) != manifest["draws"][str(seed)]["sha256"]:
            raise SystemExit("draw_seed%d hash mismatch" % seed)
    if sha256(os.path.join(out_dir, "template_replacement.jsonl")) != manifest["template_replacement"]["sha256"]:
        raise SystemExit("template_replacement hash mismatch")
    print("verified 188 AdvBench probes, 194 eligible probes, three draws, and template replacement")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/probe_pool")
    parser.add_argument("--advbench-source", default="")
    parser.add_argument("--train-data", default="data/training_corpora/llm_lat_300.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.advbench_source and not args.check:
        write_pools(args.out_dir, args.advbench_source, args.train_data)
    verify_pools(args.out_dir)


if __name__ == "__main__":
    main()
