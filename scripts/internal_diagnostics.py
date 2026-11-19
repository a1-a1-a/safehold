#!/usr/bin/env python3
"""Internal analyses: 600 separability, 2400 gated-MLP selection, 96 geometry states.

Run once per model state. Neuron selection is performed on the original aligned
checkpoint, one layer at a time; reuse its saved coordinates for comparisons.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "safehold"))
import common as C
SIZES = {"separability": (600, 300), "neurons": (2400, 1200),
         "geometry96": (96, 48), "activation96": (96, 48)}


def requests(path, mode):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    n, each = SIZES[mode]
    labels = np.array([r["label"] == "harmful" for r in rows], dtype=np.int8)
    if len(rows) != n or int(labels.sum()) != each or any(r["label"] not in ("harmful", "benign") for r in rows):
        raise ValueError(f"{mode} requires exactly {each} harmful and {each} benign requests")
    texts = [r["text"] for r in rows]
    if len(set(" ".join(t.lower().split()) for t in texts)) != n:
        raise ValueError("repeated normalized request")
    hashes = np.array([hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts])
    return texts, labels, hashes


def features(model, tok, texts, mode, layer):
    gated = mode in ("neurons", "activation96")
    if gated:
        mlp = C.inner(model).model.layers[layer].mlp
        if not hasattr(mlp, "gate_proj") or not hasattr(mlp, "up_proj"):
            raise ValueError("gated MLP with gate_proj and up_proj required")
    result = []
    for text in texts:
        ids = tok(C.build_chat(tok, text), return_tensors="pt", truncation=True,
                  max_length=512, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            if gated:
                captured = []
                def hook(_module, inputs):
                    h = inputs[0]
                    g = F.silu(mlp.gate_proj(h)) * mlp.up_proj(h)
                    captured.append(g.amax(dim=1)[0].float().cpu().numpy())
                handle = mlp.register_forward_pre_hook(hook)
                try:
                    model(**ids)
                finally:
                    handle.remove()
                if len(captured) != 1:
                    raise RuntimeError("expected one gated MLP hook per request")
                vector = captured[0]
            else:
                hidden = model(**ids, output_hidden_states=True).hidden_states
                vector = np.stack([h[0, -1].float().cpu().numpy() for h in hidden])
        result.append(vector)
    return np.stack(result).astype(np.float32)


def five_fold(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return [cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                            x[:, layer, :], y, cv=cv, scoring="roc_auc", n_jobs=1).tolist()
            for layer in range(x.shape[1])]


def neuron_selection(x, y):
    from sklearn.metrics import roc_auc_score
    z = torch.from_numpy(x).to("cuda")
    target = torch.from_numpy(y.astype(np.float32)).to("cuda")
    weight = torch.nn.Parameter(torch.zeros(x.shape[1], device="cuda"))
    bias = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer = torch.optim.Adam([weight, bias], lr=1e-2, weight_decay=1e-3)
    for _ in range(3000):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(z @ weight + bias, target).backward()
        optimizer.step()
    with torch.no_grad():
        coeff = weight.cpu().numpy()
        auc = float(roc_auc_score(y, (z @ weight + bias).cpu().numpy()))
    normalized = (coeff - coeff.mean()) / (coeff.std() + 1e-12)
    return coeff, np.flatnonzero((coeff > 0) & (np.abs(normalized) > 3)), auc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=SIZES)
    ap.add_argument("--base", default=os.environ.get("MODEL_PATH", ""))
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--coordinates", default="")
    ap.add_argument("--out", required=True, help="absent .npz output path")
    a = ap.parse_args()
    if not a.base or (a.mode in ("neurons", "activation96") and a.layer is None):
        ap.error("base model and gated-MLP layer are required")
    if a.mode == "neurons" and a.lora:
        ap.error("identify fixed coordinates only on the original checkpoint")
    if a.mode == "activation96" and not a.coordinates:
        ap.error("--coordinates must point to the original checkpoint neuron selection")
    filename = {600: "cysecbench_600.jsonl", 2400: "cysecbench_2400.jsonl",
                96: "cysecbench_96.jsonl"}[SIZES[a.mode][0]]
    source = Path(a.data) if a.data else ROOT / "data" / "internal_analysis" / filename
    texts, labels, hashes = requests(source, a.mode)
    tok, model = C.load(base=a.base, lora=(a.lora or None))
    x = features(model, tok, texts, a.mode, a.layer)
    summary = {"mode": a.mode, "n": len(texts), "layer": a.layer, "base": a.base,
               "lora": a.lora, "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    extra = {}
    if a.mode == "separability":
        summary["auc_folds"] = five_fold(x, labels)
    if a.mode == "neurons":
        coeff, selected, auc = neuron_selection(x, labels)
        extra = {"coefficients": coeff, "selected_neurons": selected}
        summary.update({"selected_neurons": selected.tolist(), "in_sample_auc": auc,
                        "probe_updates": 3000, "rule": "coefficient > 0 and absolute z-score > 3"})
    if a.mode == "activation96":
        with np.load(a.coordinates) as original:
            if int(original["layer"]) != a.layer or str(original["base"]) != a.base:
                raise ValueError("fixed neuron coordinates belong to a different layer or model")
            selected = original["selected_neurons"].astype(int)
        summary["selected_neurons"] = selected.tolist()
        summary["harmful_mean"] = x[labels == 1][:, selected].mean(axis=0).tolist()
        extra = {"selected_neurons": selected}
    dest = Path(a.out)
    if dest.suffix != ".npz" or dest.exists() or dest.with_suffix(".json").exists():
        raise ValueError("choose an absent .npz output with an absent matching .json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, vectors=x, labels=labels, prompt_sha256=hashes,
                        layer=a.layer if a.layer is not None else -1, base=a.base, **extra)
    dest.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", dest, dest.with_suffix(".json"))


if __name__ == "__main__":
    main()
