#!/usr/bin/env python3
"""
verify_delta.py — independently recompute Delta-Accuracy (Fig 8 / Table III deltas)
from raw predictions and diff against outputs/analysis/accuracy_change_by_prompt_full.csv.

Variant-level (each retained perturbed variant is a record, matching the CSV's
n_matched_records): for each (model, attack, template),
  clean_acc     = mean over variants of [source-message clean pred == true]
  perturbed_acc = mean over variants of [variant pred == true]
  delta         = clean_acc - perturbed_acc

NOTE: accuracy_change uses the NON-dedup set (sentence n = 4911 = 1637x3), so this
defaults to preds_norm_2110_final.jsonl (unlike Fig 4/5 which used the dedup set).

  python verify_delta.py
"""
import argparse, csv, json, collections

def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line: yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/preds_dedup_2110_final.jsonl")
    ap.add_argument("--csv", default="outputs/analysis/accuracy_change_by_prompt_full.csv")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    clean = {}                                    # (model,template,id) -> (pred,true)
    variants = collections.defaultdict(list)      # (model,attack,template) -> [(clean_ok,pert_ok)]
    for r in read_jsonl(args.preds):
        m, t, i = r.get("model"), r.get("template"), r.get("id")
        if r.get("condition") == "clean":
            clean[(m, t, i)] = (r.get("pred"), r.get("true_label"))
    for r in read_jsonl(args.preds):
        if r.get("condition") == "clean": continue
        m, t, i = r.get("model"), r.get("template"), r.get("id")
        a = {"multi": "multi-level"}.get(r.get("attack_class"), r.get("attack_class"))
        cp = clean.get((m, t, i))
        if cp is None: continue
        pred_c, tl = cp
        variants[(m, a, t)].append((pred_c == tl, r.get("pred") == tl))

    rec = {}
    for k, vs in variants.items():
        n = len(vs)
        ca = 100 * sum(1 for c, p in vs if c) / n
        pa = 100 * sum(1 for c, p in vs if p) / n
        rec[k] = (n, ca, pa, ca - pa)

    rows = {}
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            rows[(row["model"], row["attack"], row["prompt"])] = row

    print(f"Recomputed from {args.preds}\nvs {args.csv}\n")
    bad = 0; maxdev = 0.0; checked = 0
    for k, row in sorted(rows.items()):
        r = rec.get(k)
        if r is None:
            print(f"  MISSING recompute for {k}"); continue
        n, ca, pa, d = r
        for field, mine in (("clean_accuracy_percent", ca),
                            ("perturbed_accuracy_percent", pa),
                            ("delta_accuracy_points", d)):
            stored = float(row[field]); checked += 1
            dev = abs(mine - stored); maxdev = max(maxdev, dev)
            if dev > args.tol:
                bad += 1
                if bad <= 15:
                    print(f"  MISMATCH {k} {field}: mine={mine:.4f} stored={stored:.4f} dev={dev:.4f}")
        if int(row["n_matched_records"]) != n and bad <= 15:
            print(f"  N-DIFF {k}: mine={n} stored={row['n_matched_records']}")
    print(f"\n[delta-acc] compared={checked}  violations>{args.tol}pp={bad}  max dev={maxdev:.4f}pp")

    # sanity: paper text-claim cells
    print("\nPaper text-claim cells (character-level delta points):")
    for m, t, want in [("meta-llama/Llama-3.1-8B-Instruct","t1_neutral_plain",19.46),
                       ("meta-llama/Llama-3.1-8B-Instruct","t2_expert_plain",20.08),
                       ("meta-llama/Llama-3.2-3B-Instruct","t3_neutral_explanation",14.07),
                       ("meta-llama/Llama-3.2-3B-Instruct","t4_expert_explanation",14.20)]:
        r = rec.get((m,"character",t))
        if r: print(f"  {m.split('/')[-1]:24s} {t:22s} delta={r[3]:.2f}  (paper {want})")

if __name__ == "__main__":
    main()
