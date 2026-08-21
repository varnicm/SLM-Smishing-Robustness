#!/usr/bin/env python3
"""
pick_rationale_example.py
-------------------------
Produce ONE real, paper-ready rationale-stability example from the pipeline's
own outputs. No recomputation: it reuses the scores already stored in
matched_pairs.jsonl (true = S_true, base = S_base, gap = G_i) and only JOINS in
the actual rationale text from the preds file.

INPUTS (defaults match your tree; override if needed)
  --pairs  outputs/analysis/dim2_matched/matched_pairs.jsonl
  --preds  outputs/preds_norm_2110_final.jsonl        (may pass several / globs)

SELECTION
  Keeps only eligible pairs where ALL FOUR gaps are positive (clean example of
  message-specific stability), optionally filtered by --model / --attack /
  --template, then ranks by --rank {mean,jaccard,min} (default mean gap).
  Rationales are joined from preds; pairs with a missing/empty rationale are
  skipped. --maxchars keeps the printed rationales readable.

USAGE
  python pick_rationale_example.py
  python pick_rationale_example.py --attack sentence --model Qwen/Qwen3.5-9B
  python pick_rationale_example.py --topk 5 --rank jaccard --maxchars 700
"""
import argparse, glob, json, re, sys

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)", re.IGNORECASE)

# metric key in matched_pairs -> display label
METRICS = [("bertscore_f1", "BERTScore F1"),
           ("sentcos",      "Cosine (MiniLM)"),
           ("rougel_f1",    "ROUGE-L F1"),
           ("jaccard",      "Jaccard")]

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_rationale(raw):
    """Rationale = text before the FINAL: tag (your rationale_metrics.extract_rationale)."""
    if not raw:
        return ""
    m = _FINAL_RE.search(raw)
    return (raw[: m.start()] if m else raw).strip()

def load_preds(paths):
    """Index preds rows for fast lookup.
    clean key:     (model, template, id, 'CLEAN')
    perturbed key: (model, template, id, attack_class, seed)
    """
    clean, pert = {}, {}
    n = 0
    for pat in paths:
        for fp in glob.glob(pat):
            for r in read_jsonl(fp):
                n += 1
                rat = extract_rationale(r.get("raw", ""))
                cond = r.get("condition")
                key_common = (r.get("model"), r.get("template"), r.get("id"))
                if cond == "clean":
                    clean[key_common + ("CLEAN",)] = rat
                else:
                    pert[key_common + (r.get("attack_class"), r.get("seed"))] = rat
    if n == 0:
        sys.exit("No preds rows loaded — check --preds path/glob.")
    return clean, pert

def rank_value(gap, mode):
    vals = [gap[k] for k, _ in METRICS]
    if mode == "min":     return min(vals)
    if mode == "jaccard": return gap["jaccard"]
    return sum(vals) / len(vals)   # mean

def fmt_block(rec, clean_r, true_r, base_r):
    t, b, g = rec["true"], rec["base"], rec["gap"]
    lines = []
    lines.append("=" * 74)
    lines.append(f"model={rec['model']}   template={rec['template']}   "
                 f"attack={rec['attack']}   seed={rec['seed']}   pred={rec['pred_label']}")
    lines.append(f"anchor={rec['anchor_id']}  true_partner={rec['true_partner_id']}  "
                 f"baseline_partner={rec['baseline_partner_id']}")
    lines.append("-" * 74)
    lines.append("A_i  (clean rationale):")
    lines.append("   " + clean_r)
    lines.append("A_i' (perturbed rationale, same message):")
    lines.append("   " + true_r)
    lines.append("B_i  (matched same-label baseline, DIFFERENT message):")
    lines.append("   " + base_r)
    lines.append("-" * 74)
    lines.append(f"{'metric':<18}{'S_true':>10}{'S_base':>10}{'G_i':>10}")
    for k, label in METRICS:
        lines.append(f"{label:<18}{t[k]:>10.4f}{b[k]:>10.4f}{g[k]:>10.4f}")
    mean_gap = sum(g[k] for k, _ in METRICS) / len(METRICS)
    lines.append(f"mean gap = {mean_gap:+.4f}  -> "
                 f"{'message-specific stability (all four gaps > 0)' if all(g[k]>0 for k,_ in METRICS) else 'mixed'}")
    lines.append("=" * 74)
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/analysis/dim2_matched/matched_pairs.jsonl")
    ap.add_argument("--preds", nargs="*", default=["outputs/preds_norm_2110_final.jsonl"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--attack", default=None, help="character|word|sentence|multi")
    ap.add_argument("--template", default=None, help="e.g. t3_neutral_explanation")
    ap.add_argument("--rank", choices=["mean", "jaccard", "min"], default="mean")
    ap.add_argument("--maxchars", type=int, default=600,
                    help="skip pairs whose any rationale exceeds this length")
    ap.add_argument("--topk", type=int, default=1)
    args = ap.parse_args()

    pairs = list(read_jsonl(args.pairs))
    # filter: all four gaps positive + optional facets
    def keep(r):
        if args.model    and r["model"]    != args.model:    return False
        if args.attack   and r["attack"]   != args.attack:   return False
        if args.template and r["template"] != args.template: return False
        return all(r["gap"][k] > 0 for k, _ in METRICS)
    pairs = [r for r in pairs if keep(r)]
    if not pairs:
        sys.exit("No pairs pass the all-positive-gap filter with those facets.")

    clean, pert = load_preds(args.preds)

    def rationales(r):
        a = clean.get((r["model"], r["template"], r["anchor_id"], "CLEAN"))
        tp = pert.get((r["model"], r["template"], r["true_partner_id"], r["attack"], r["seed"]))
        bp = pert.get((r["model"], r["template"], r["baseline_partner_id"], r["attack"], r["seed"]))
        return a, tp, bp

    candidates = []
    for r in pairs:
        a, tp, bp = rationales(r)
        if not (a and tp and bp):                       # missing join
            continue
        if max(len(a), len(tp), len(bp)) > args.maxchars:
            continue
        candidates.append((rank_value(r["gap"], args.rank), r, a, tp, bp))

    if not candidates:
        sys.exit("Pairs found, but rationale text could not be joined (or all exceeded "
                 "--maxchars). Check that --preds contains the explanation-template rows, "
                 "or raise --maxchars.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[: args.topk]
    print(f"\n{len(candidates)} joinable all-positive-gap pairs; showing top {len(chosen)} by {args.rank} gap:\n")
    out = []
    for score, r, a, tp, bp in chosen:
        print(fmt_block(r, a, tp, bp))
        print()
        out.append({**r, "clean_rationale": a, "perturbed_rationale": tp, "baseline_rationale": bp})

    with open("rationale_example_out.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("Wrote rationale_example_out.json")

if __name__ == "__main__":
    main()
