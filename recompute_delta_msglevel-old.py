#!/usr/bin/env python3
"""
Recompute Delta-Accuracy at the SOURCE-MESSAGE level (each message weighted equally)
and print it side-by-side with the current variant-weighted values.

Message-level (new): for each source message with a retained variant of the attack,
  clean_correct   = 1[clean pred == true]
  pert_correct    = mean over the message's retained variants of 1[variant pred == true]
  then average clean_correct and pert_correct across messages; delta = clean - pert.

Variant-level (old): read from accuracy_change_by_prompt_full.csv (n_matched_records).

  python recompute_delta_msglevel.py
"""
import argparse, csv, json, collections, statistics as st

ATTACK_MAP = {"multi": "multi-level"}   # preds attack_class -> CSV attack name
TEMPLATES = ("t1_neutral_plain","t2_expert_plain","t3_neutral_explanation","t4_expert_explanation")

def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line=line.strip()
            if line: yield json.loads(line)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/preds_dedup_2110_final.jsonl")
    ap.add_argument("--csv",   default="outputs/analysis/accuracy_change_by_prompt_full.csv")
    a=ap.parse_args()

    clean={}                                   # (model,template,id)->(pred,true)
    variants=collections.defaultdict(list)     # (model,attack,template,id)->[pred,...]
    for r in read_jsonl(a.preds):
        m,t,i=r.get("model"),r.get("template"),r.get("id")
        if r.get("condition")=="clean":
            clean[(m,t,i)]=(r.get("pred"), r.get("true_label"))
    for r in read_jsonl(a.preds):
        if r.get("condition")=="clean": continue
        m,t,i=r.get("model"),r.get("template"),r.get("id")
        atk=ATTACK_MAP.get(r.get("attack_class"), r.get("attack_class"))
        variants[(m,atk,t,i)].append(r.get("pred"))

    # message-level recompute
    new={}
    keys=set((m,atk,t) for (m,atk,t,i) in variants)
    for (m,atk,t) in keys:
        cc=[]; pc=[]
        for i in set(i for (mm,aa,tt,i) in variants if (mm,aa,tt)==(m,atk,t)):
            cp=clean.get((m,t,i))
            if cp is None: continue
            pred_c,tl=cp; vs=variants[(m,atk,t,i)]
            if not vs: continue
            cc.append(1 if pred_c==tl else 0)
            pc.append(sum(1 for p in vs if p==tl)/len(vs))
        if cc:
            ca=100*st.mean(cc); pa=100*st.mean(pc)
            new[(m,atk,t)]=(len(cc), ca, pa, ca-pa)

    # old variant-weighted from CSV
    old={}
    for row in csv.DictReader(open(a.csv)):
        old[(row["model"],row["attack"],row["prompt"])]=(
            int(row["n_matched_records"]),
            float(row["clean_accuracy_percent"]),
            float(row["perturbed_accuracy_percent"]),
            float(row["delta_accuracy_points"]))

    models=sorted(set(m for (m,a2,t) in new))
    attacks=["character","word","sentence","multi-level"]
    print(f"{'model':32s} {'attack':11s} {'tmpl':4s} | {'OLD dAcc':>9s} {'NEW dAcc':>9s} {'diff':>7s} | "
          f"{'n_rec':>6s} {'n_msg':>6s}")
    print("-"*100)
    ivE=[]
    for m in models:
        for atk in attacks:
            for t in TEMPLATES:
                n=new.get((m,atk,t)); o=old.get((m,atk,t))
                if not n or not o: continue
                nrec,oca,opa,od=o; nmsg,nca,npa,nd=n
                print(f"{m:32s} {atk:11s} {t[:2]:4s} | {od:9.2f} {nd:9.2f} {nd-od:7.2f} | {nrec:6d} {nmsg:6d}")
                # IV-E quoted cells
                if (m.endswith("Llama-3.1-8B-Instruct") and atk=="character" and t in ("t1_neutral_plain","t2_expert_plain")) or \
                   (m.endswith("Llama-3.2-3B-Instruct") and atk=="character" and t in ("t3_neutral_explanation","t4_expert_explanation")):
                    ivE.append((m.split('/')[-1], t, od, nd))
        print()

    print("=== Section IV-E quoted cells (character-level) : OLD -> NEW ===")
    for name,t,od,nd in ivE:
        print(f"  {name:24s} {t:22s}  {od:.2f} -> {nd:.2f}")

if __name__=="__main__":
    main()
