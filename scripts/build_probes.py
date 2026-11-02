#!/usr/bin/env python3
"""Normalize eight trusted AdvBench probes into a jsonl file with a `q` field."""
import json, os

src = os.environ.get("PROBE_SOURCE", "data/raw/advbench_probes8.jsonl")
out = os.environ.get("PROBE_FILE", "data/advbench_probes8.jsonl")
rows = []
for line in open(src, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    q = (r.get("q") or r.get("text") or r.get("prompt") or "").strip()
    if q:
        rows.append({"q": q, "source": r.get("source", "AdvBench")})
if len(rows) != 8:
    raise SystemExit(f"expected 8 probes, found {len(rows)}")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", out, len(rows))
