#!/usr/bin/env python3
"""Evaluate the 22-prefix refusal-opening read-out defined by Eq. (C.10).

For every harmful request, each refusal prefix is appended to the chat prompt.
The read-out multiplies the conditional probabilities of the prefix tokens at a
given layer and sums the resulting sequence probabilities over the 22 prefixes.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "safehold"))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("SC_BASE", C.BASE_DEFAULT))
    ap.add_argument("--lora", default="")
    ap.add_argument("--harm", default=os.path.join(
        ROOT, "data", "internal_analysis", "cysecbench_96.jsonl"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--layer", type=int, default=-1,
                    help="layer to summarize; -1 means the final layer")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "refusal_opening.json"))
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.harm, encoding="utf-8")
            if line.strip()]
    harmful = [r for r in rows if r.get("label", "harmful") == "harmful"]
    prompts = [str(r.get("text") or r.get("prompt") or "").strip()
               for r in harmful]
    prompts = [p for p in prompts if p][:a.n]
    if not prompts:
        raise SystemExit("no harmful prompts found in " + a.harm)

    tok, model = C.load(base=a.base, lora=(a.lora or None))
    res = C.measure(model, tok, prompts, max_len=a.max_len)
    P = res["P"]
    layer = a.layer if a.layer >= 0 else P.shape[1] - 1
    layer = max(0, min(P.shape[1] - 1, layer))

    out = {
        "harm": os.path.abspath(a.harm),
        "n": len(prompts),
        "n_layers": int(res["nL"]),
        "layer": int(layer),
        "prefixes": res["prefixes"],
        "prompts": prompts,
        "mean": float(P[:, layer].mean()),
        "P": P.tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("RESULT", json.dumps({"n": out["n"], "layer": out["layer"],
                                "mean": out["mean"]}, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
