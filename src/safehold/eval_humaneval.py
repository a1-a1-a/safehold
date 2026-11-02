#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""he_eval.py -- HumanEval pass@1 (raw completion + fence strip) for a given LoRA."""
import argparse, json, os, sys, subprocess
import glob as g
import torch
import common as C
from datasets import Dataset

def run_test(program, test, entry, timeout=10):
    code=program+"\n"+test+"\ncheck("+entry+")\n"
    try:
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,timeout=timeout,text=True)
        return r.returncode==0
    except Exception:
        return False

@torch.no_grad()
def gen(model,tok,prompts,mn=256,bs=8):
    outs=[]
    for i in range(0,len(prompts),bs):
        ch=prompts[i:i+bs]
        e=tok(ch,return_tensors='pt',padding=True,truncation=True,max_length=1024).to('cuda')
        gg=model.generate(**e,max_new_tokens=mn,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(ch)):
            outs.append(tok.decode(gg[j][e['input_ids'].shape[1]:],skip_special_tokens=True))
    return outs

STOPS=["\nclass ","\nif __name__","\nprint(","\n# ","\nassert ","\n\n\n"]
def extract(comp, prompt):
    c=comp
    if "```" in c:
        parts=c.split("```")
        cand=[p for p in parts if ("def " in p or "return" in p)]
        if cand: c=max(cand,key=len)
        first=c.split("\n",1)
        if len(first)==2 and first[0].strip().lower() in ("python","py"): c=first[1]
    for st in STOPS:
        i=c.find(st)
        if i>0: c=c[:i]
    return prompt+c

ap=argparse.ArgumentParser()
ap.add_argument('--lora',default=''); ap.add_argument('--tag',required=True)
ap.add_argument('--n',type=int,default=164); ap.add_argument('--out',default=os.environ.get('HUMANEVAL_OUT','results/humaneval.json'))
a=ap.parse_args()
arrow=os.environ.get('HUMANEVAL_ARROW','')
if arrow:
    ds=Dataset.from_file(arrow).select(range(min(a.n,164)))
else:
    from datasets import load_dataset
    ds=load_dataset('openai_humaneval', split='test').select(range(min(a.n,164)))
lora=None if a.lora in ('','base') else a.lora
tok,model=C.load(lora=lora)
prompts=[ds[i]['prompt'] for i in range(len(ds))]
outs=gen(model,tok,prompts)
npass=0
for i in range(len(ds)):
    ok=run_test(extract(outs[i], ds[i]['prompt']), ds[i]['test'], ds[i]['entry_point'])
    npass+=int(ok)
rec={'tag':a.tag,'n':len(ds),'pass1':npass/len(ds)}
os.makedirs(os.path.dirname(a.out),exist_ok=True)
allr=json.load(open(a.out)) if os.path.exists(a.out) else {}
allr[a.tag]=rec; json.dump(allr,open(a.out,'w'),indent=2)
print('RESULT',json.dumps(rec),flush=True)
