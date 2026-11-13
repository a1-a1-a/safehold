#!/usr/bin/env python3
"""SBR / Safety Anchor paper-protocol implementation.

This file implements the SBR objective described in the paper: cache the final
hidden state at the decision position of the fixed harmful probes from the frozen
aligned model, and add an MSE anchor to the submitted-data SFT loss. No public
upstream repository for this baseline was located when this snapshot was added;
this is an independent implementation of the described objective, not a vendored
copy of official code.

Example:
    python baselines/SBR/paper_protocol.py \
      --model "$MODEL_PATH" \
      --data data/training_corpora/mixed_fractions/mixed_100.json \
      --probes data/fixed_safety_supervision/advbench_probes8.jsonl \
      --out outputs_baseline/sbr
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--probes", default="data/fixed_safety_supervision/advbench_probes8.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxrows", type=int, default=300)
    ap.add_argument("--maxsteps", type=int, default=600)
    ap.add_argument("--epochsteps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def read_json_rows(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def conversation_pair(row):
    conv = row["conversations"]
    q = next(c["value"] for c in conv if c["from"] == "human")
    a = next(c["value"] for c in conv if c["from"] == "gpt")
    return q, a


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    print("SEED", a.seed, flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def chat(messages, generate):
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=generate)

    rows = read_json_rows(a.data)[:a.maxrows]
    train_ids, train_labels = [], []
    for row in rows:
        q, answer = conversation_pair(row)
        messages = [{"role": "system", "content": a.system},
                    {"role": "user", "content": q}]
        prompt = chat(messages, True)
        full = chat(messages + [{"role": "assistant", "content": answer}], False)
        p_ids = tok(prompt, truncation=True, max_length=a.maxlen,
                    add_special_tokens=False)["input_ids"]
        f_ids = tok(full, truncation=True, max_length=a.maxlen,
                    add_special_tokens=False)["input_ids"]
        if len(f_ids) <= len(p_ids) or f_ids[:len(p_ids)] != p_ids:
            raise ValueError("assistant prefix does not match the chat template")
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        train_ids.append(f_ids)
        train_labels.append(labels)
    print("N_ROWS", len(train_ids), flush=True)

    probes = []
    for row in read_jsonl(a.probes):
        probes.append(row.get("q") or row.get("prompt") or row.get("question"))
    if not probes or any(p is None for p in probes):
        raise ValueError("probe file must contain q, prompt, or question fields")
    probe_ids = [tok(chat([{"role": "system", "content": a.system},
                           {"role": "user", "content": p}], True),
                     truncation=True, max_length=a.maxlen,
                     add_special_tokens=False)["input_ids"] for p in probes]
    print("N_PROBES", len(probe_ids), flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.enable_input_require_grads()
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    def pad_batch(seqs):
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        return ids.to("cuda"), mask.to("cuda")

    probe_t, probe_mask = pad_batch(probe_ids)
    probe_rows = torch.arange(len(probe_ids), device="cuda")
    probe_pos = probe_mask.sum(1) - 1

    @torch.no_grad()
    def reference_hidden():
        with model.disable_adapter():
            out = model(input_ids=probe_t, attention_mask=probe_mask,
                        output_hidden_states=True)
        return out.hidden_states[-1][probe_rows, probe_pos].detach().float()

    h_ref = reference_hidden()
    print("HREF", tuple(h_ref.shape), flush=True)

    def anchor_loss():
        out = model(input_ids=probe_t, attention_mask=probe_mask,
                    output_hidden_states=True)
        h = out.hidden_states[-1][probe_rows, probe_pos].float()
        return ((h - h_ref) ** 2).mean()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr)
    order = list(range(len(train_ids)))
    random.shuffle(order)
    step = 0
    done = False
    saved = set()
    model.train()
    while step < a.maxsteps and not done:
        for j in order:
            if step >= a.maxsteps:
                done = True
                break
            ids = torch.tensor([train_ids[j]], device="cuda")
            labels = torch.tensor([train_labels[j]], device="cuda")
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float()
            task = F.cross_entropy(logits, labels[:, 1:].reshape(-1))
            loss = task + a.lam * anchor_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print("STEP", step, "task %.4f sbr %.6f" % (
                    float(task.detach()), float(anchor_loss().detach())), flush=True)
            if a.epochsteps > 0 and step % a.epochsteps == 0 and step // a.epochsteps not in saved:
                ep = step // a.epochsteps
                ckpt = "%s_ep%d" % (a.out, ep)
                model.save_pretrained(ckpt)
                tok.save_pretrained(ckpt)
                saved.add(ep)
                print("EPOCH_SAVED", ckpt, flush=True)
            torch.cuda.empty_cache()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("SAVED", a.out, flush=True)


if __name__ == "__main__":
    main()
