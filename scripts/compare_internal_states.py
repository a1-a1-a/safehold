#!/usr/bin/env python3
"""Compare original, undefended, and SAFEHOLD diagnostics on the same 96 requests."""
import argparse
import json
from pathlib import Path

import numpy as np


def direction(x, labels, layer):
    delta = x[labels == 1, layer].mean(0) - x[labels == 0, layer].mean(0)
    length = np.linalg.norm(delta)
    if length <= 0:
        raise ValueError("zero harmfulness direction")
    return delta / length


def geometry(original, undefended, defended, labels):
    directions = [direction(original, labels, layer) for layer in range(original.shape[1])]
    projections = [float(original[labels == 1, layer].mean(0) @ directions[layer])
                   for layer in range(original.shape[1])]
    chosen = int(np.argmax(projections))
    base_direction = directions[chosen]
    final_direction = direction(original, labels, -1)
    denom = float(original[labels == 1, -1].mean(0) @ final_direction)
    if abs(denom) < 1e-12:
        raise ValueError("zero reference projection")
    result = {"selected_layer": chosen, "aligned_layer_projection": projections[chosen]}
    for name, states in (("aligned", original), ("undefended", undefended), ("safehold", defended)):
        cosine = float(np.clip(direction(states, labels, chosen) @ base_direction, -1, 1))
        projected = float(states[labels == 1, -1].mean(0) @ final_direction)
        result[name] = {"angle_degrees": float(np.degrees(np.arccos(cosine))),
                        "final_projection_retention": projected / denom}
    return result


def activations(original, undefended, defended, labels, selected):
    if not len(selected):
        raise ValueError("no selected neurons")
    means = [states[labels == 1][:, selected].mean(0)
             for states in (original, undefended, defended)]
    suppressed = means[1] < means[0]
    return {"selected": len(selected), "suppressed": int(suppressed.sum()),
            "suppression_rate": float(suppressed.mean()),
            "rebound_rate": (float((means[2][suppressed] > means[1][suppressed]).mean())
                             if suppressed.any() else None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--undefended", required=True)
    ap.add_argument("--safehold", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with np.load(a.aligned) as base, np.load(a.undefended) as und, np.load(a.safehold) as safe:
        for candidate in (und, safe):
            for key in ("labels", "prompt_sha256"):
                if not np.array_equal(base[key], candidate[key]):
                    raise ValueError("different requests or labels across model states")
            if str(base["base"]) != str(candidate["base"]):
                raise ValueError("states belong to different base models")
        x, y, z, labels = base["vectors"], und["vectors"], safe["vectors"], base["labels"]
        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError("feature dimensions differ across states")
        if x.ndim == 3:
            result = geometry(x, y, z, labels)
        else:
            for candidate in (und, safe):
                if int(candidate["layer"]) != int(base["layer"]):
                    raise ValueError("different neuron layers")
            selected = base["selected_neurons"] if "selected_neurons" in base else None
            if selected is None:
                raise ValueError("aligned activation file lacks fixed original neuron coordinates")
            result = activations(x, y, z, labels, selected.astype(int))
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
