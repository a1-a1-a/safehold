#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""def_fix.py -- corrected REARM / anchoring-only trainer.
FIXES:
  (1) Path supervision: logits at st-1+col predict token at st+col (was st-1+col).
  (2) Unified chat template: system+user at BOTH training and eval (was user-only).
  (3) Displacement direction selectable: --anchor_dir m2r = D(p_theta||p_ref) [paper], r2m = D(p_ref||p_theta).
"""
import argparse, json, os, sys, random
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import common as C
ap=argparse.ArgumentParser()
ap.add_argument('--model',required=True); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True)
ap.add_argument('--anchor',default=os.environ.get('PROBE_FILE','data/advbench_probes8.jsonl'))
ap.add_argument('--nprobe',type=int,default=8)
ap.add_argument('--mode',default='anchor_path')    # anchor | anchor_path | path
ap.add_argument('--anchor_dir',default='m2r')      # m2r: D(p_theta||p_ref) | r2m: D(p_ref||p_theta)
ap.add_argument('--lam',type=float,default=0.6)    # boundary-displacement weight
ap.add_argument('--lc',type=float,default=1.0)     # refusal-path weight
ap.add_argument('--maxsteps',type=int,default=220); ap.add_argument('--epochsteps',type=int,default=0); ap.add_argument('--lr',type=float,default=2e-4)
ap.add_argument('--maxlen',type=int,default=512)
ap.add_argument('--system',default='You are a helpful assistant.')
ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--pool',type=str,default=os.environ.get('PROBE_POOL','data/probe_pool.jsonl'))
ap.add_argument('--poolsize',type=int,default=200)
ap.add_argument('--k',type=int,default=8)
ap.add_argument('--sched',default='feedback')
ap.add_argument('--warmup',type=int,default=50)
ap.add_argument('--refresh',type=int,default=100)
ap.add_argument('--tau',type=float,default=0.5)
ap.add_argument('--topiccap',type=int,default=2)
a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); DEV=torch.device("cuda")
import numpy as _np
random.seed(a.seed); _np.random.seed(a.seed)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print('SEED=%d'%a.seed, flush=True)
tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="right"
SYS=a.system
def chat(msgs,gen=True): return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=gen)
def to_msgs(e):
    msgs=[{"role":"system","content":SYS}]
    for t in e["conversations"]:
        msgs.append({"role":"user" if t["from"]=="human" else "assistant","content":t["value"]})
    return msgs
def build_example(e):
    # Assistant-only supervision: system/user tokens get label -100 and are excluded from the loss.
    msgs=to_msgs(e)
    ids=tok(chat(msgs,gen=False),truncation=True,max_length=a.maxlen,add_special_tokens=False)["input_ids"]
    n=0
    if msgs and msgs[-1]["role"]=="assistant":
        pre=tok(chat(msgs[:-1],gen=True),add_special_tokens=False)["input_ids"]
        while n<min(len(pre),len(ids)) and pre[n]==ids[n]: n+=1
        assert n>0, "assistant prefix not found in tokenized conversation"
    labels=[-100]*n+list(ids[n:])
    return {"input_ids":ids,"labels":labels}
_raw=json.load(open(a.data,encoding="utf-8"))
data=[build_example(e) for e in _raw]
SAFE=("I'm sorry, but I can't help with that. This request asks for something harmful or unsafe, "
      "and I'm not able to assist with it.")
# ---- probe pool: unique harmful questions sampled once ----
qs=[]; _seen=set()
for _line in open(a.pool, encoding='utf-8'):
    if not _line.strip(): continue
    _r=json.loads(_line)
    _q=(_r.get('q') or _r.get('text') or '').strip()
    if _q not in _seen:
        _seen.add(_q); qs.append(_q)
# stratified spread: take evenly spaced items so every part of the corpus is represented
_n=min(a.poolsize, len(qs))
_step=len(qs)/_n
_pool=[qs[min(int(i*_step), len(qs)-1)] for i in range(_n)]
# leakage guard: drop pool questions that also appear in the submitted corpus
_train = {t["value"].strip().lower() for e in _raw for t in e["conversations"] if t["from"] == "human"}
_before = len(_pool)
_pool = [q for q in _pool if q.strip().lower() not in _train]
print("POOL_GUARD removed=%d kept=%d" % (_before - len(_pool), len(_pool)), flush=True)
# topic buckets: every probe gets a coarse harmful-topic label (lexicon is label-free and deterministic)
_LEX=[("extremism",r"terror|extremis|radicali|jihad|militia|recruit"),
 ("weapons",r"\bbomb|explosive|weapon|firearm|\bgun|ammunition|grenade|poison|toxin|nerve agent|molotov|incendiar|arson|assassinat|hit list|shooting|shoot"),
 ("sexual",r"sexual|porn|nude|rape|explicit|\bchild|minor|human traffick|sex traffick"),
 ("drugs",r"drug|cocaine|heroin|meth|opioid|fentanyl|narcotic|\blsd\b|cannabis|marijuana"),
 ("hate",r"hate|racis|racial|ethnic|religio|slur|discriminat|genocide|supremac|harass|bully|reputation|defam|violence against"),
 ("violence",r"kill|murder|tortur|assault|kidnap|abduct|hostage|violen|self.harm|suicide|\bharm|physical damage|\battack"),
 ("cyber",r"malware|virus|ransomware|hack|phish|ddos|denial.of.service|botnet|exploit|vulnerab|password|keylog|rootkit|spyware|backdoor|injection|cyberattack|breach|network|software|encrypt|wiretap|eavesdrop|surveil|device|drone|robot|\bapp\b|\bapps\b|script|program|computer|system|database|records|server|online|phone"),
 ("fraud",r"fraud|scam|launder|insider|tax|evad|counterfeit|fake|forg|passport|credit card|identity theft|embezzle|ponzi|bribe|money|financial|investor|stock|bank|\brob|heist|steal|theft|burglar|break into|smuggl|black market|cheat|exam|illegal software|bitcoin|currenc|illegal activities"),
 ("privacy",r"stalk|track|\bspy|dox|private|personal|address|sensitive information|conversation|restricted|control devices|monitor"),
 ("misinfo",r"misinformation|propaganda|disinformation|conspiracy|fake news|persuade|manipulat|false claim|fake review|false information|rumou?r|deceiv"),
 ("illicit",r"dump|waste|emission|permit|inspection|protest|riot|criminal|illegal|crime|abuse|position of authority|evade|avoid getting caught|avoid detection")]
import re as _re
_LEX=[(_n2,_re.compile(_p)) for _n2,_p in _LEX]
def _topic(q):
    _s=q.lower()
    for _n2,_p in _LEX:
        if _p.search(_s): return _n2
    return "other"
_buckets={}
for _i,_q in enumerate(_pool): _buckets.setdefault(_topic(_q),[]).append(_i)
_bkeys=sorted(_buckets, key=lambda t:(-len(_buckets[t]), t))
print('POOL unique=%d chosen=%d k=%d buckets=%s' % (len(qs), len(_pool), a.k,
      {t:len(_buckets[t]) for t in _bkeys}), flush=True)
probes=_pool
rows=[];stays=[]
for p in probes:
    pi=tok(chat([{"role":"system","content":SYS},{"role":"user","content":p}],gen=True),add_special_tokens=False)["input_ids"]
    ai=tok(SAFE,add_special_tokens=False)["input_ids"]
    rows.append(pi+ai); stays.append(len(pi))
def pack(rs):
    mx=max(len(r) for r in rs); ids=torch.full((len(rs),mx),tok.pad_token_id,dtype=torch.long); am=torch.zeros((len(rs),mx),dtype=torch.long)
    for i,r in enumerate(rs): ids[i,:len(r)]=torch.tensor(r); am[i,:len(r)]=1
    return {"input_ids":ids.to(DEV),"attention_mask":am.to(DEV)}
pe=pack(rows); st=torch.tensor(stays,device=DEV); lens=torch.tensor([len(r) for r in rows],device=DEV)
nb=len(rows); ar=torch.arange(nb,device=DEV); dp=(st-1).clamp(min=0)
maxk=int((lens-st).max()); col=torch.arange(maxk,device=DEV)[None,:]
# path alignment: logits at st-1+c predict token st+c
lg_pos=(st[:,None]-1+col); tg_pos=(st[:,None]+col)
path_mask=((tg_pos<lens[:,None])&(lg_pos>=0)).float()
lg_pos=lg_pos.clamp(0,pe["input_ids"].shape[1]-1); tg_pos=tg_pos.clamp(0,pe["input_ids"].shape[1]-1)
_bos = getattr(tok, "bos_token_id", None)
_c = sum(r.count(_bos) for r in rows) if _bos is not None else 0
print("BOS_CHECK single_bos=%s data_ids=%d probe_bos=%d"%(_c == nb, len(data[0]["input_ids"]), _c), flush=True)
assert _bos is None or _c == nb, "probe rows must contain exactly one BOS"
print("mode=%s anchor_dir=%s probes=%d resp_tokens=%d maxlen=%d system=%r seed=%d"%(a.mode,a.anchor_dir,nb,int((lens-st).sum()),a.maxlen,SYS[:20],a.seed),flush=True)
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
m=AutoModelForCausalLM.from_pretrained(a.model,quantization_config=bnb,device_map={"":0},trust_remote_code=True)
m=prepare_model_for_kbit_training(m,use_gradient_checkpointing=True); m.config.use_cache=False
m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
m.enable_input_require_grads()
for p in m.parameters():
    if p.requires_grad: p.data=p.data.float()
tdec_pool=[]; c0_parts=[]
with torch.no_grad(), m.disable_adapter():
    for i in range(0, nb, 16):
        sl=slice(i, min(i+16, nb))
        ids_b=pe["input_ids"][sl]; am_b=pe["attention_mask"][sl]
        st_b=st[sl]; lens_b=lens[sl]; ar_b=torch.arange(ids_b.shape[0], device=DEV)
        lg=m(input_ids=ids_b, attention_mask=am_b).logits.float()
        tdec_pool.append(lg[ar_b, (st_b-1).clamp(min=0)].cpu())
        maxk_b=int((lens_b-st_b).max()); col_b=torch.arange(maxk_b,device=DEV)[None,:]
        lgp=(st_b[:,None]-1+col_b).clamp(0,lg.shape[1]-1)
        tgp=(st_b[:,None]+col_b).clamp(0,lg.shape[1]-1)
        mk=((tgp<lens_b[:,None])&(lgp>=0)&(col_b>=1)).float()
        pr=lg[ar_b[:,None],lgp]; tgt=pe["input_ids"][sl][ar_b[:,None],tgp]
        surp=-F.log_softmax(pr,-1).gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        c0_parts.append(((surp*mk).sum(1)/(mk.sum(1)+1e-6)).cpu())
        del lg,pr,surp
        torch.cuda.empty_cache()
    tdec_pool=torch.cat(tdec_pool,0).to(DEV)
    tdec=tdec_pool
    c0=torch.cat(c0_parts,0).to(DEV)
print('REF_CACHED %s C0_MEAN=%.4f' % (tuple(tdec_pool.shape), float(c0.mean())), flush=True)
def disp_term(student_logits, reference_logits):
    ls=F.log_softmax(student_logits,-1); lt=F.log_softmax(reference_logits,-1)
    if a.anchor_dir=="m2r":
        return (ls.exp()*(ls-lt)).sum(-1).mean()
    return (lt.exp()*(lt-ls)).sum(-1).mean()
def reg_loss(idx=None):
    global pe, st, lens, ar, dp, lg_pos, tg_pos, path_mask, tdec  # noqa: F824
    if idx is None:
        pe_b, st_b, lens_b = pe, st, lens
    else:
        idx = list(idx)
        mx = max(len(rows[i]) for i in idx)
        ids = torch.full((len(idx), mx), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(idx), mx), dtype=torch.long)
        for k, i in enumerate(idx):
            r = rows[i]; ids[k, :len(r)] = torch.tensor(r); am[k, :len(r)] = 1
        pe_b = {"input_ids": ids.to(DEV), "attention_mask": am.to(DEV)}
        st_b = st[torch.tensor(idx, device=DEV)] if torch.is_tensor(st) else st
        lens_b = lens[torch.tensor(idx, device=DEV)] if torch.is_tensor(lens) else lens
    nb_b = pe_b["input_ids"].shape[0]
    ar_b = torch.arange(nb_b, device=DEV)
    dp_b = (st_b - 1).clamp(min=0)
    maxk_b = int((lens_b - st_b).max())
    col_b = torch.arange(maxk_b, device=DEV)[None, :]
    lg_pos_b = (st_b[:, None] - 1 + col_b)
    tg_pos_b = (st_b[:, None] + col_b)
    path_mask_b = ((tg_pos_b < lens_b[:, None]) & (lg_pos_b >= 0)).float()
    lg_pos_b = lg_pos_b.clamp(0, pe_b["input_ids"].shape[1] - 1)
    tg_pos_b = tg_pos_b.clamp(0, pe_b["input_ids"].shape[1] - 1)
    o = m(**pe_b); lg = o.logits.float(); r = torch.zeros((), device=DEV)
    if "anchor" in a.mode:
        td = tdec_pool[torch.tensor(idx, device=DEV)] if idx is not None else tdec
        r = r + a.lam * disp_term(lg[ar_b, dp_b], td)
    if "path" in a.mode:
        pr = lg[ar_b[:, None], lg_pos_b]; tgt = pe_b["input_ids"][ar_b[:, None], tg_pos_b]
        surp = -F.log_softmax(pr, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        r = r + a.lc * (surp * path_mask_b).sum() / (path_mask_b.sum() + 1e-6)
    return r
def collate(sample):
    ids=sample["input_ids"]; lab=sample["labels"]
    return {"input_ids":torch.tensor([ids],device=DEV),"attention_mask":torch.ones(1,len(ids),dtype=torch.long,device=DEV),"labels":torch.tensor([lab],device=DEV)}
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr)
random.seed(0); order=list(range(len(data))); random.shuffle(order); step=0; done=False; m.train()
# ---- feedback-guided probe sampling ----
# A category-interleaved queue supplies the 2 coverage slots and the warm-up. The other
# 6 slots are allocated from detached probe statistics: decision drift from the frozen
# reference and refusal-continuation degradation relative to the aligned model.
_pool_topics=[_topic(p) for p in _pool]
_rng_sched=random.Random(a.seed+2)
_risk=torch.zeros(nb,device=DEV)
_sd=1.0; _su=1.0; _have_feedback=False
_probe_order=[]; _round=0; _poolrng=random.Random(a.seed+1)
def _build_round():
    _q={t:list(v) for t,v in _buckets.items()}
    for _t in _q: _poolrng.shuffle(_q[_t])
    _T=sum(len(v) for v in _q.values()); _tagged=[]
    for _t in _q:
        _m=len(_q[_t])
        for _j,_it in enumerate(_q[_t]):
            _tagged.append((((_j+0.5)*_T/_m), _poolrng.random(), _it))
    _tagged.sort()
    return [_x[2] for _x in _tagged]
def _cover(n):
    global _probe_order, _round
    while len(_probe_order) < n:
        _probe_order += _build_round(); _round += 1
        print('PROBE_ROUND %d (full pool cover %d)' % (_round, len(_pool)), flush=True)
    out, _probe_order = _probe_order[:n], _probe_order[n:]
    return out
def _p90(x):
    # empirical 90th percentile over the supplied (already filtered) values
    x=x.float().flatten()
    if x.numel()==0: return 1.0
    return float(torch.quantile(x,0.9).clamp(min=1e-6))  # default interpolation is linear
@torch.no_grad()
def _refresh_feedback(k):
    global _risk,_sd,_su,_have_feedback
    m.eval(); d=torch.zeros(nb,device=DEV); c=torch.zeros(nb,device=DEV)
    for i in range(0,nb,a.k):
        sl=slice(i,min(i+a.k,nb))
        ids_b=pe["input_ids"][sl]; am_b=pe["attention_mask"][sl]
        st_b=st[sl]; lens_b=lens[sl]; ar_b=torch.arange(ids_b.shape[0],device=DEV)
        lg=m(input_ids=ids_b,attention_mask=am_b).logits.float()
        lp=torch.log_softmax(lg[ar_b,(st_b-1).clamp(min=0)],-1)
        lq=torch.log_softmax(tdec_pool[sl],-1)
        d[sl]=(lp.exp()*(lp-lq)).sum(-1)
        maxk_b=int((lens_b-st_b).max()); col_b=torch.arange(maxk_b,device=DEV)[None,:]
        lgp=(st_b[:,None]-1+col_b).clamp(0,lg.shape[1]-1)
        tgp=(st_b[:,None]+col_b).clamp(0,lg.shape[1]-1)
        mk=((tgp<lens_b[:,None])&(lgp>=0)&(col_b>=1)).float()
        pr=lg[ar_b[:,None],lgp]; tgt=pe["input_ids"][sl][ar_b[:,None],tgp]
        surp=-F.log_softmax(pr,-1).gather(-1,tgt.unsqueeze(-1)).squeeze(-1)
        c[sl]=((surp*mk).sum(1)/(mk.sum(1)+1e-6)); del lg,pr,surp
        torch.cuda.empty_cache()
    u=(c-c0).clamp(min=0); xd=torch.log1p(d)
    if not _have_feedback:
        # calibrate only on strictly positive values, as in Eq. (scales)
        _sd=_p90(xd[d>0]); _su=_p90(u[u>0])
        with open(os.path.join(a.out,'scheduler.json'),'w',encoding='utf-8') as _f:
            json.dump({'sched':a.sched,'pool_size':int(nb),'batch':int(a.k),
                       'warmup':int(a.warmup),'refresh':int(a.refresh),
                       'tau':float(a.tau),'category_cap':int(a.topiccap),
                       's_d':float(_sd),'s_u':float(_su),'seed':int(a.seed)},_f)
    r=0.5*torch.clamp(xd/(_sd+1e-6),min=0.0,max=1.0)+0.5*torch.clamp(u/(_su+1e-6),min=0.0,max=1.0)
    _risk=r; _have_feedback=True; m.train()
    print('FEEDBACK k=%d d_mean=%.4f u_mean=%.4f r_mean=%.3f r_max=%.3f sd=%.3f su=%.3f'%(k,float(d.mean()),float(u.mean()),float(r.mean()),float(r.max()),_sd,_su),flush=True)
    _sched_log.write(json.dumps({'step':int(k),'event':'feedback',
        'd_mean':float(d.mean()),'u_mean':float(u.mean()),
        'r_mean':float(r.mean()),'r_max':float(r.max()),
        's_d':float(_sd),'s_u':float(_su)})+chr(10)); _sched_log.flush()
def _softmax_pick(score,n,exclude,cap):
    cnt={}
    for i in exclude: cnt[_pool_topics[i]]=cnt.get(_pool_topics[i],0)+1
    picked=[]
    for _ in range(n):
        cand=[i for i in range(nb) if (i not in exclude and i not in picked
              and cnt.get(_pool_topics[i],0)<cap)]
        if not cand: break
        w=torch.softmax(score[torch.tensor(cand,device=DEV)]/a.tau,0).cpu().numpy()
        j=int(_rng_sched.choices(range(len(cand)),weights=w.tolist(),k=1)[0])
        v=cand.pop(j); picked.append(v)
        cnt[_pool_topics[v]]=cnt.get(_pool_topics[v],0)+1
    return picked
_sched_log=open(os.path.join(a.out,'scheduler_log.jsonl'),'w',encoding='utf-8')
def _log_batch(k,B,refreshed):
    _sched_log.write(json.dumps({'step':int(k),'probes':[int(x) for x in B],
                                 'refresh':bool(refreshed)})+chr(10))
    if k%50==0: _sched_log.flush()
def select_batch(k):
    if a.sched=='round' or k<a.warmup:
        B=[]
        while len(B)<a.k:
            for i in _cover(1):
                if i not in B: B.append(i)
        _log_batch(k,B,False); return B
    refreshed=False
    if (k-a.warmup)%a.refresh==0:
        _refresh_feedback(k); refreshed=True
    B=_softmax_pick(_risk,max(0,a.k-2),[],a.topiccap)
    tries=0
    while len(B)<a.k and tries<40:
        for i in _cover(1):
            tries+=1
            if i not in B and sum(1 for j in B if _pool_topics[j]==_pool_topics[i])<a.topiccap:
                B.append(i)
            if len(B)>=a.k: break
    if len(B)<a.k:
        for i in range(nb):
            if i not in B and sum(1 for j in B if _pool_topics[j]==_pool_topics[i])<a.topiccap:
                B.append(i)
            if len(B)>=a.k: break
    if len(B)<a.k:
        for i in range(nb):
            if i not in B: B.append(i)
            if len(B)>=a.k: break
    B=B[:a.k]; _log_batch(k,B,refreshed); return B
print('SCHED=%s warmup=%d refresh=%d tau=%.2f topiccap=%d'%(a.sched,a.warmup,a.refresh,a.tau,a.topiccap),flush=True)
while step<a.maxsteps and not done:
    for j in order:
        if step>=a.maxsteps: done=True; break
        idx=select_batch(step)
        opt.zero_grad(); o=m(**collate(data[j])); (o.loss+reg_loss(idx)).backward(); opt.step(); step+=1; torch.cuda.empty_cache()
        if a.epochsteps>0 and step % a.epochsteps == 0:
            ep=step//a.epochsteps; ck=a.out+'_ep%d'%ep
            m.save_pretrained(ck); tok.save_pretrained(ck)
            print('EPOCH_SAVED',ck,flush=True)
        if step%50==0:
            with torch.no_grad():
                rl=float(reg_loss(idx)) if "path" in a.mode or "anchor" in a.mode else 0.0
            print("step%d sft=%.3f reg=%.3f"%(step,float(o.loss),rl),flush=True)
_sched_log.close()
m.save_pretrained(a.out); tok.save_pretrained(a.out); print("SAVED",a.out,flush=True); print("FIX_DONE",flush=True)
