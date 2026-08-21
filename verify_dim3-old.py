#!/usr/bin/env python3
"""
verify_dim3.py — final checks for Dimension 3:
  * Table VI: pooled true/baseline/gap means over matched_pairs.jsonl (4 metrics)
  * Table V : rationale-provision rate per model x {T3, T4} from expl_all.jsonl
              (usable rationale = >= 10 non-whitespace chars in text before FINAL:)

Model-free; login node is fine:
  python verify_dim3.py
"""
import argparse, json, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
try:
    import rationale_metrics as rm
    extract_rationale = rm.extract_rationale
except Exception:
    import re
    _F = re.compile(r"FINAL:\s*(smishing|ham)", re.IGNORECASE)
    def extract_rationale(raw):
        if not raw: return ""
        m = _F.search(raw); return (raw[:m.start()] if m else raw).strip()

METRICS = [("bertscore_f1","BERTScore F1"),("sentcos","Sentence cosine"),
           ("rougel_f1","ROUGE-L F1"),("jaccard","Jaccard")]
TABLE_VI = {  # paper: (true, baseline, gap)
    "bertscore_f1": (0.9147, 0.8849, 0.0298),
    "sentcos":      (0.8452, 0.6609, 0.1843),
    "rougel_f1":    (0.4485, 0.3124, 0.1361),
    "jaccard":      (0.4250, 0.2820, 0.1430),
}
TABLE_V = {  # paper Table V: (T3 %, T4 %)
    "google/gemma-4-E4B-it":            (100.00,100.00),
    "Qwen/Qwen3.5-4B":                  (100.00,100.00),
    "Qwen/Qwen3.5-9B":                  (100.00,100.00),
    "meta-llama/Llama-3.1-8B-Instruct": (41.45, 78.94),
    "meta-llama/Llama-3.2-3B-Instruct": (99.95, 99.95),
    "microsoft/Phi-4-mini-instruct":    (100.00,100.00),
}

def read_jsonl(p):
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line: yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/analysis/dim2_matched/matched_pairs.jsonl")
    ap.add_argument("--expl",  default="outputs/expl_all.jsonl")
    args = ap.parse_args()

    # ---- Table VI: pooled means ----
    n = 0
    acc = {m: [0.0, 0.0, 0.0] for m, _ in METRICS}
    for r in read_jsonl(args.pairs):
        n += 1
        for m, _ in METRICS:
            acc[m][0] += r["true"][m]; acc[m][1] += r["base"][m]; acc[m][2] += r["gap"][m]
    print(f"Table VI — pooled means over {n} matched pairs:")
    print(f"  {'metric':16s} {'true':>8s}{'base':>8s}{'gap':>8s}   {'paper(t/b/g)':>22s}  ok")
    vi_ok = True
    for m, label in METRICS:
        t, b, g = (x / n for x in acc[m])
        pt, pb, pg = TABLE_VI[m]
        ok = abs(t-pt) < 1e-3 and abs(b-pb) < 1e-3 and abs(g-pg) < 1e-3
        vi_ok = vi_ok and ok
        print(f"  {label:16s} {t:8.4f}{b:8.4f}{g:8.4f}   "
              f"{pt:.4f}/{pb:.4f}/{pg:.4f}  {'OK' if ok else 'CHECK'}")
    print(f"  Table VI: {'VERIFIED' if vi_ok else 'MISMATCH'}\n")

    # ---- Table V: rationale provision ----
    T = {"t3_neutral_explanation": 0, "t4_expert_explanation": 1}
    tot = {}  # (model, tcol) -> [usable, total]
    for r in read_jsonl(args.expl):
        t = r.get("template")
        if t not in T: continue
        key = (r.get("model"), T[t])
        d = tot.setdefault(key, [0, 0]); d[1] += 1
        rat = extract_rationale(r.get("raw", ""))
        if len("".join(rat.split())) >= 10:   # >=10 non-whitespace chars
            d[0] += 1
    print("Table V — rationale provision % (recomputed vs paper):")
    print(f"  {'model':35s} {'T3 mine/paper':>18s} {'T4 mine/paper':>18s}")
    v_ok = True
    for m in TABLE_V:
        row = []
        for c in (0, 1):
            u, n2 = tot.get((m, c), [0, 0])
            pct = 100 * u / n2 if n2 else float("nan")
            paper = TABLE_V[m][c]
            if abs(pct - paper) > 0.1: v_ok = False
            row.append(f"{pct:6.2f}/{paper:6.2f}")
        print(f"  {m:35s} {row[0]:>18s} {row[1]:>18s}")
    print(f"  Table V: {'VERIFIED' if v_ok else 'CHECK flagged cells'}")

if __name__ == "__main__":
    main()
