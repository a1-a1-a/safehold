#!/usr/bin/env python3
"""Verify and copy the released mixed corpora to an isolated output directory.

This command uses the packaged subsets that match the paper's training inputs.
"""
import hashlib
import json
import os
from pathlib import Path

SOURCE = Path(os.environ.get("FRACTION_SOURCE", "data/training_corpora/mixed_fractions"))
OUTPUT = Path(os.environ.get("DATA_DIR", "data/generated_fractions"))
HARM = Path("data/training_corpora/llm_lat_300.json")
BENIGN = Path("data/training_corpora/benign_300.json")


def prompt(row):
    return next(turn["value"] for turn in row["conversations"] if turn["from"] in ("human", "user"))


def norm(s):
    return " ".join(s.lower().split())


def main():
    if SOURCE.resolve() == OUTPUT.resolve():
        raise ValueError("select a separate output directory to keep packaged data immutable")
    harmful = {norm(prompt(row)) for row in json.loads(HARM.read_text(encoding="utf-8"))}
    benign = {norm(prompt(row)) for row in json.loads(BENIGN.read_text(encoding="utf-8"))}
    if harmful & benign:
        raise ValueError("harmful/benign source corpora overlap")
    for frac in (0, 25, 50, 75, 100):
        src = SOURCE / f"mixed_{frac}.json"
        rows = json.loads(src.read_text(encoding="utf-8"))
        texts = [norm(prompt(row)) for row in rows]
        nh = sum(text in harmful for text in texts)
        nb = sum(text in benign for text in texts)
        if len(rows) != 300 or len(set(texts)) != 300 or nh != 3 * frac or nb != 300 - 3 * frac:
            raise ValueError(f"incorrect released corpus: {src} ({len(rows)} rows, {nh} harmful, {nb} benign)")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        dest = OUTPUT / src.name
        dest.write_bytes(src.read_bytes()) if not dest.exists() else None
        if dest.read_bytes() != src.read_bytes():
            raise ValueError(f"existing copy differs: {dest}")
        print(src, "->", dest, "sha256", hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
