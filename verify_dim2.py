#!/usr/bin/env python3
"""
verify_dim2.py — independently recompute Dimension-2 (prompt-configuration
sensitivity) and dataset counts from raw predictions, and auto-diff against Fig 6.

  * Dataset: unique source messages and class balance (smishing/ham).
  * Complete agreement (Fig 6): per message/variant, do all four prompt configs
    (T1..T4) produce the same label? Rate by model, by condition
    (clean, character, word, sentence, multi-level).
  * Explanation-requirement effect (Table VIII): clean-message disagreement
    T1 vs T3 (neutral) and T2 vs T4 (expert), by model.

Model-free; login node is fine:
  python verify_dim2.py
"""
import argparse, json, collections

TEMPLATES = ("t1_neutral_plain", "t2_expert_plain",
             "t3_neutral_explanation", "t4_expert_explanation")
ATTACK_MAP = {"multi": "multi-level"}

FIG6 = {   # paper Fig 6 complete-agreement %
 "google/gemma-4-E4B-it":            {"clean":95.2,"character":92.6,"word":92.9,"sentence":95.7,"multi-level":95.2},
 "Qwen/Qwen3.5-4B":                  {"clean":89.9,"character":84.7,"word":88.2,"sentence":94.7,"multi-level":92.5},
 "Qwen/Qwen3.5-9B":                  {"clean":86.2,"character":79.2,"word":83.2,"sentence":94.5,"multi-level":92.9},
 "meta-llama/Llama-3.1-8B-Instruct": {"clean":82.3,"character":74.3,"word":74.7,"sentence":87.5,"multi-level":86.7},
 "meta-llama/Llama-3.2-3B-Instruct": {"clean":83.6,"character":78.3,"word":79.3,"sentence":86.0,"multi-level":87.0},
 "microsoft/Phi-4-mini-instruct":    {"clean":80.9,"character":91.1,"word":84.3,"sentence":84.4,"multi-level":90.6},
}

def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="outputs/preds_dedup_2110_final.jsonl")
    ap.add_argument("--tol", type=float, default=0.2)
    args = ap.parse_args()

    # group predictions by template
    clean = collections.defaultdict(dict)     # (model,id) -> {template: pred}
    pert  = collections.defaultdict(dict)     # (model,attack,id,seed) -> {template: pred}
    truelab = {}
    for r in read_jsonl(args.preds):
        m, t, i = r.get("model"), r.get("template"), r.get("id")
        truelab[i] = r.get("true_label")
        if r.get("condition") == "clean":
            clean[(m, i)][t] = r.get("pred")
        else:
            a = ATTACK_MAP.get(r.get("attack_class"), r.get("attack_class"))
            pert[(m, a, i, r.get("seed"))][t] = r.get("pred")

    # ---- dataset ----
    ids = set(i for (m, i) in clean)
    by = collections.Counter(truelab[i] for i in ids)
    print(f"Dataset: unique source messages = {len(ids)}  "
          f"(smishing={by.get('smishing')}, ham={by.get('ham')})")
    print(f"  paper final set = 2107 (1055 smishing, 1052 ham); raw includes 3 extra ham\n")

    # ---- complete agreement (Fig 6) ----
    def agree_rate(groups):
        n = ok = 0
        for g in groups.values():
            if len(g) < 4:            # need all four prompts present
                continue
            n += 1
            if len(set(g[t] for t in TEMPLATES)) == 1:
                ok += 1
        return (100 * ok / n, n) if n else (None, 0)

    models = sorted(FIG6)
    print("Complete agreement across T1-T4 (recomputed vs Fig 6):")
    print(f"  {'model':35s} {'cond':11s} {'mine':>7s} {'paper':>7s} {'dev':>6s}")
    worst = 0.0; fails = 0
    for m in models:
        # clean
        cg = {k: v for k, v in clean.items() if k[0] == m}
        rate, n = agree_rate(cg)
        for cond, groups in [("clean", cg)] + [
            (a, {k: v for k, v in pert.items() if k[0] == m and k[1] == a})
            for a in ("character", "word", "sentence", "multi-level")]:
            rate, n = agree_rate(groups)
            paper = FIG6[m][cond]
            dev = abs(rate - paper) if rate is not None else float("nan")
            worst = max(worst, dev if rate is not None else 0)
            flag = "" if (rate is not None and dev <= args.tol) else "  <-- CHECK"
            if flag: fails += 1
            print(f"  {m:35s} {cond:11s} {rate:7.1f} {paper:7.1f} {dev:6.2f}{flag}")
    print(f"\n  max dev vs Fig 6 = {worst:.2f}pp   cells flagged (> {args.tol}) = {fails}\n")

    # ---- explanation-requirement effect (Table VIII), clean messages ----
    print("Explanation-requirement disagreement on clean msgs (Table VIII):")
    print(f"  {'model':35s} {'T1vsT3(neut)':>13s} {'T2vsT4(exp)':>12s}")
    for m in models:
        d13 = d24 = n = 0
        for (mm, i), g in clean.items():
            if mm != m or len(g) < 4:
                continue
            n += 1
            if g["t1_neutral_plain"]      != g["t3_neutral_explanation"]: d13 += 1
            if g["t2_expert_plain"]       != g["t4_expert_explanation"]:  d24 += 1
        print(f"  {m:35s} {100*d13/n:12.2f}% {100*d24/n:11.2f}%")
    print("\n  (paper text: neutral 3.27% [Gemma] .. 13.10% [Llama-3.2]; "
          "expert 2.61% [Gemma] .. 15.80% [Llama-3.1])")

if __name__ == "__main__":
    main()
