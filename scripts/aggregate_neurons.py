#!/usr/bin/env python3
"""Aggregate per-layer neuron-selection outputs into a model-level summary."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "selected_neurons" not in data or "layer" not in data:
                raise ValueError("%s has no selected_neurons or layer" % path)
            layer = int(data["layer"])
            selected = np.asarray(data["selected_neurons"], dtype=np.int64).reshape(-1)
            base = str(data["base"]) if "base" in data else ""
            data_sha256 = str(data["data_sha256"]) if "data_sha256" in data else ""
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "selected_neurons" not in payload or "layer" not in payload:
            raise ValueError("%s has no selected_neurons or layer" % path)
        layer = int(payload["layer"])
        selected = np.asarray(payload["selected_neurons"], dtype=np.int64).reshape(-1)
        base = str(payload.get("base", ""))
        data_sha256 = str(payload.get("data_sha256", ""))
    if layer < 0:
        raise ValueError("%s has a negative layer" % path)
    if selected.size and int(selected.min()) < 0:
        raise ValueError("%s has a negative neuron index" % path)
    if len(set(selected.tolist())) != selected.size:
        raise ValueError("%s has duplicate neuron indices" % path)
    return layer, selected, base, data_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="per-layer .npz or .json outputs")
    parser.add_argument("--out", required=True, help="output basename")
    args = parser.parse_args()

    by_layer = {}
    bases = set()
    hashes = set()
    for name in args.inputs:
        layer, selected, base, data_sha256 = load(name)
        if layer in by_layer:
            raise SystemExit("duplicate layer %d" % layer)
        by_layer[layer] = selected
        if base:
            bases.add(base)
        if data_sha256:
            hashes.add(data_sha256)
    if not by_layer:
        raise SystemExit("no neuron-selection outputs")
    if len(bases) > 1:
        raise SystemExit("inputs belong to different base models")
    if len(hashes) > 1:
        raise SystemExit("inputs use different analysis data")

    layers = sorted(by_layer)
    per_layer = {str(layer): int(by_layer[layer].size) for layer in layers}
    total = sum(per_layer.values())
    summary = {
        "base": next(iter(bases), ""),
        "data_sha256": next(iter(hashes), ""),
        "n_layers": len(layers),
        "layers": layers,
        "per_layer": per_layer,
        "total_selected": total,
    }

    out = Path(args.out)
    if out.suffix in (".json", ".npz"):
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    npz_path = out.with_suffix(".npz")
    if json_path.exists() or npz_path.exists():
        raise SystemExit("output already exists: %s or %s" % (json_path, npz_path))
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, nSN=total,
                        **{str(layer): by_layer[layer] for layer in layers})
    print("layers=%d total_selected=%d" % (len(layers), total))


if __name__ == "__main__":
    main()
