#!/usr/bin/env python3
"""
Message-level Delta-Accuracy with class-stratified, source-message cluster bootstrap CIs.
Self-contained: reads raw predictions, writes a CSV with point estimate + 95% CI per
(model, attack, prompt). No edits to bootstrap_analysis.py needed.

Method (matches the paper's III-E design):
  * Per source message with a retained variant of the attack under a prompt:
      d_m = 1[clean pred == true]  -  mean over its retained variants of 1[variant pred == true]
    (each message contributes ONE value -> equal weight; its variants are baked in).
  * Point estimate dAcc = 100 * mean(d_m over messages).
  * CI: resample source messages WITH replacement, SEPARATELY within each class
    (smishing / ham) so class counts are preserved; recompute mean(d_m); 1000 replicates;
    report the 2.5th and 97.5th percentiles.

Usage:
  python bootstrap_delta_msglevel.py \
     --preds outputs/preds_dedup_2110_final.jsonl \
     --out   outputs/analysis/delta_accuracy_msglevel_ci.csv \
     --B 1000 --seed 0
"""
import argparse, csv, json, collections, random

TEMPLATES = ("t1_neutral_plain","t2_expert_plain","t3_neutral_explanation","t4_expert_explanation")
ATTACK_MAP = {"multi": "multi-level"}      # preds attack_class -> reported attack name
ATTACKS = ["character","word","sentence","multi-level"]

def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line: yield json.loads(line)

def percentile(sorted_vals, q):
    """linear-interpolation percentile, q in [0,100]."""
    if not sorted_vals: return float("nan")
    n = len(sorted_vals)
    if n == 1: return sorted_vals[0]
    pos = (q/100.0)*(n-1)
    lo = int(pos); hi = min(lo+1, n-1); frac = pos-lo
    return sorted_vals[lo]*(1-frac) + sorted_vals[hi]*frac

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/preds_dedup_2110_final.jsonl")
    ap.add_argument("--out",   default="outputs/analysis/delta_accuracy_msglevel_ci.csv")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", default="outputs/analysis/accuracy_change_by_prompt_msglevel.csv",
                    help="optional: cross-check point estimates against this CSV")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    clean = {}                                    # (model,template,id) -> (pred, true)
    variants = collections.defaultdict(list)      # (model,attack,template,id) -> [pred,...]
    for r in read_jsonl(args.preds):
        m,t,i = r.get("model"), r.get("template"), r.get("id")
        if r.get("condition") == "clean":
            clean[(m,t,i)] = (r.get("pred"), r.get("true_label"))
    for r in read_jsonl(args.preds):
        if r.get("condition") == "clean": continue
        m,t,i = r.get("model"), r.get("template"), r.get("id")
        a = ATTACK_MAP.get(r.get("attack_class"), r.get("attack_class"))
        variants[(m,a,t,i)].append(r.get("pred"))

    models = sorted(set(m for (m,a,t,i) in variants))
    rows = []
    for m in models:
        for a in ATTACKS:
            for t in TEMPLATES:
                # per-message d_m, split by class for stratified resampling
                dm_by_class = {"smishing": [], "ham": []}
                cc_all, pf_all = [], []
                for i in set(i2 for (mm,aa,tt,i2) in variants if (mm,aa,tt)==(m,a,t)):
                    cp = clean.get((m,t,i))
                    if cp is None: continue
                    pred_c, tl = cp
                    vs = variants[(m,a,t,i)]
                    if not vs: continue
                    cc = 1 if pred_c == tl else 0
                    pf = sum(1 for p in vs if p == tl)/len(vs)
                    cc_all.append(cc); pf_all.append(pf)
                    if tl in dm_by_class:
                        dm_by_class[tl].append(cc - pf)
                nmsg = len(cc_all)
                if nmsg == 0: continue
                clean_acc = 100*sum(cc_all)/nmsg
                pert_acc  = 100*sum(pf_all)/nmsg
                dacc      = clean_acc - pert_acc
                # class-stratified cluster bootstrap
                sm, hm = dm_by_class["smishing"], dm_by_class["ham"]
                boot = []
                for _ in range(args.B):
                    samp = []
                    if sm: samp += [sm[rng.randrange(len(sm))] for _ in range(len(sm))]
                    if hm: samp += [hm[rng.randrange(len(hm))] for _ in range(len(hm))]
                    boot.append(100*sum(samp)/len(samp))
                boot.sort()
                lo = percentile(boot, 2.5); hi = percentile(boot, 97.5)
                rows.append({"model":m,"attack":a,"prompt":t,"n_source_messages":nmsg,
                             "clean_accuracy_percent":round(clean_acc,4),
                             "perturbed_accuracy_percent":round(pert_acc,4),
                             "delta_accuracy_points":round(dacc,4),
                             "delta_accuracy_ci_low":round(lo,4),
                             "delta_accuracy_ci_high":round(hi,4)})

    fields = ["model","attack","prompt","n_source_messages","clean_accuracy_percent",
              "perturbed_accuracy_percent","delta_accuracy_points",
              "delta_accuracy_ci_low","delta_accuracy_ci_high"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"Wrote {args.out}  ({len(rows)} rows; B={args.B}, seed={args.seed})")

    # optional cross-check: point estimates must match the message-level CSV
    try:
        ref = {}
        for r in csv.DictReader(open(args.check)):
            ref[(r["model"],r["attack"],r["prompt"])] = float(r["delta_accuracy_points"])
        md = 0.0; nchk = 0
        for r in rows:
            k = (r["model"],r["attack"],r["prompt"])
            if k in ref:
                md = max(md, abs(r["delta_accuracy_points"]-ref[k])); nchk += 1
        print(f"cross-check vs {args.check}: {nchk} cells, max |Δ point est.|={md:.4f}pp  "
              f"({'OK' if md<0.01 else 'MISMATCH'})")
    except FileNotFoundError:
        print(f"(cross-check file {args.check} not found; skipped)")

if __name__ == "__main__":
    main()
