#!/usr/bin/env python3
"""
verify_dim1.py — independently recompute Dimension-1 numbers from raw predictions
and diff against the committed CSV (outputs/analysis/asr_benign_flip_by_prompt.csv).

Recomputes, per (model, attack, template), using the paper's message-level method
(each source message contributes equally; per-message value = fraction of that
message's retained variants that flip):

  * clean accuracy (model x template)              -> Fig 3 / Table III
  * ASR   = smishing->ham, denominator = clean-correct smishing msgs w/ a variant
  * benign-flip = ham->smishing, denominator = clean-correct ham msgs w/ a variant
  * pooled ASR = mean of the 4 templates' ASR       -> Fig 4

Then compares asr_percent / benign_flip_percent / n against the CSV.

Run (model-free, fast; login node is fine):
  python verify_dim1.py
  python verify_dim1.py --preds outputs/preds_norm_2110_final.jsonl   # to test the non-dedup set
"""
import argparse, csv, json, collections

def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

ATTACK_MAP = {"multi": "multi-level"}  # preds attack_class -> CSV attack name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/preds_dedup_2110_final.jsonl")
    ap.add_argument("--asr-csv", default="outputs/analysis/asr_benign_flip_by_prompt.csv")
    ap.add_argument("--tol", type=float, default=0.05)  # percent points
    args = ap.parse_args()

    clean = {}                                   # (model,template,id) -> (pred, true)
    variants = collections.defaultdict(list)     # (model,attack,template,id) -> [pred,...]
    truelab = {}                                 # id -> true_label
    for r in read_jsonl(args.preds):
        m, t, i = r.get("model"), r.get("template"), r.get("id")
        tl = r.get("true_label"); truelab[i] = tl
        if r.get("condition") == "clean":
            clean[(m, t, i)] = (r.get("pred"), tl)
        else:
            atk = ATTACK_MAP.get(r.get("attack_class"), r.get("attack_class"))
            variants[(m, atk, t, i)].append(r.get("pred"))

    # recompute per (model, attack, template)
    rec = {}   # (model,attack,template) -> dict(asr,n_asr,benign,n_benign)
    keys = set((m, a, t) for (m, a, t, i) in variants)
    for (m, a, t) in keys:
        asr_vals, ben_vals = [], []
        for i in set(i for (mm, aa, tt, i) in variants if (mm, aa, tt) == (m, a, t)):
            cp = clean.get((m, t, i))
            if cp is None:
                continue
            pred_c, tl = cp
            vs = variants[(m, a, t, i)]
            if not vs:
                continue
            if tl == "smishing" and pred_c == "smishing":      # clean-correct smishing
                asr_vals.append(sum(1 for p in vs if p == "ham") / len(vs))
            elif tl == "ham" and pred_c == "ham":              # clean-correct ham
                ben_vals.append(sum(1 for p in vs if p == "smishing") / len(vs))
        rec[(m, a, t)] = {
            "asr": 100 * sum(asr_vals) / len(asr_vals) if asr_vals else None,
            "n_asr": len(asr_vals),
            "benign": 100 * sum(ben_vals) / len(ben_vals) if ben_vals else None,
            "n_benign": len(ben_vals),
        }

    # load committed CSV
    csv_rows = {}
    with open(args.asr_csv) as fh:
        for row in csv.DictReader(fh):
            csv_rows[(row["model"], row["attack"], row["prompt"])] = row

    # ---- compare ----
    print(f"Recomputed from: {args.preds}")
    print(f"Comparing against: {args.asr_csv}\n")
    bad = 0; maxdev = 0.0; checked = 0; missing = 0
    for k, row in sorted(csv_rows.items()):
        r = rec.get(k)
        if r is None:
            missing += 1; continue
        for field, mine in (("asr_percent", r["asr"]), ("benign_flip_percent", r["benign"])):
            stored = float(row[field]); checked += 1
            if mine is None:
                bad += 1; continue
            dev = abs(mine - stored); maxdev = max(maxdev, dev)
            if dev > args.tol:
                bad += 1
                if bad <= 12:
                    print(f"  MISMATCH {k} {field}: recomputed={mine:.4f} stored={stored:.4f} (dev={dev:.4f})")
        # n check
        for field, mine in (("n_asr_messages", r["n_asr"]), ("n_benign_flip_messages", r["n_benign"])):
            if int(row[field]) != mine and bad <= 12:
                print(f"  N-DIFF {k} {field}: recomputed={mine} stored={row[field]}")
    print(f"\n[ASR/benign] compared={checked}  violations>{args.tol}pp={bad}  "
          f"max dev={maxdev:.4f}pp  csv-rows-not-recomputed={missing}")

    # ---- pooled ASR (Fig 4) : mean over the 4 templates ----
    print("\nPooled ASR (mean over templates) — compare to Fig 4:")
    models = sorted(set(m for (m, a, t) in rec))
    attacks = ["character", "word", "sentence", "multi-level"]
    for m in models:
        cells = []
        for a in attacks:
            vals = [rec[(m, a, t)]["asr"] for t in
                    ("t1_neutral_plain","t2_expert_plain","t3_neutral_explanation","t4_expert_explanation")
                    if (m, a, t) in rec and rec[(m, a, t)]["asr"] is not None]
            cells.append(f"{a}:{sum(vals)/len(vals):.2f}" if vals else f"{a}:NA")
        print(f"  {m:35s} " + "  ".join(cells))

    # ---- clean accuracy (Fig 3 / Table III) ----
    print("\nClean accuracy (model x template) — compare to Fig 3 / Table III:")
    acc = collections.defaultdict(lambda: [0, 0])
    for (m, t, i), (pred, tl) in clean.items():
        acc[(m, t)][1] += 1
        if pred == tl:
            acc[(m, t)][0] += 1
    for m in models:
        cells = []
        for t in ("t1_neutral_plain","t2_expert_plain","t3_neutral_explanation","t4_expert_explanation"):
            c = acc.get((m, t))
            cells.append(f"{t.split('_')[0]}:{100*c[0]/c[1]:.2f}(n={c[1]})" if c else f"{t}:NA")
        print(f"  {m:35s} " + "  ".join(cells))

if __name__ == "__main__":
    main()
