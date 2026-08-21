#!/usr/bin/env python3
"""
pick_rationale_example.py  (v2)
-------------------------------
Produce ONE real, paper-ready rationale-stability example from the pipeline's
own outputs, with a built-in CONSISTENCY GATE so the displayed rationale text
provably matches the stored similarity scores.

It reuses the scores already in matched_pairs.jsonl (true=S_true, base=S_base,
gap=G_i) and JOINS the rationale text from the preds file. It then RECOMPUTES
token-Jaccard (cheap, no models) for both the true pair and the baseline and
compares against the stored Jaccard. Only pairs whose text matches the stored
scores are shown (with --require-consistent, default on) -- this automatically
rejects preds-version mismatches and the Phi-4-mini canned-boilerplate join.

INPUTS (defaults match your tree)
  --pairs  outputs/analysis/dim2_matched/matched_pairs.jsonl
  --preds  outputs/preds_norm_2110_final.jsonl        (pass several / globs ok)

FILTERS / SELECTION
  --model / --attack / --template   restrict facets
  --require-consistent   keep only pairs where recomputed Jaccard matches stored
                         for BOTH the true pair and the baseline (default: on)
  --base-consistent-only if you only care that the baseline text matches
  --minchars N           drop pairs whose clean rationale is shorter than N
                         (default 80; filters terse canned lines)
  --exclude-substr S     drop pairs whose clean rationale contains S
                         (default: the Phi-4-mini canned cue-list)
  --nonidentical         require clean != perturbed (so the four metrics take
                         distinct, non-degenerate values -- best for a figure)
  --label smishing|ham   restrict to that predicted label
  --rank mean|jaccard|min   ranking of the gap (default mean)
  --topk N               show N best (default 1)
  --tol T                Jaccard match tolerance for the gate (default 1e-4)

USAGE
  python pick_rationale_example.py                      # best consistent example
  python pick_rationale_example.py --nonidentical --topk 8
  python pick_rationale_example.py --label smishing --attack sentence --topk 8
  # if nothing passes the baseline gate, the preds file differs from the one
  # that built matched_pairs -- point --preds at the correct file, or add
  # --base-consistent-only off by using --no-require-consistent to inspect.
  python pick_rationale_example.py --no-require-consistent --topk 5
"""
import argparse, glob, json, re, sys

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)", re.IGNORECASE)
_TOK = re.compile(r"[a-z0-9]+")

METRICS = [("bertscore_f1", "BERTScore F1"),
           ("sentcos",      "Cosine (MiniLM)"),
           ("rougel_f1",    "ROUGE-L F1"),
           ("jaccard",      "Jaccard")]

CANNED = "Relevant cues: Urgency, suspicious link, brand impersonation, request for personal info"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_rationale(raw):
    if not raw:
        return ""
    m = _FINAL_RE.search(raw)
    return (raw[: m.start()] if m else raw).strip()

def toks(s):
    return _TOK.findall((s or "").lower())

def jaccard(a, b):
    sa, sb = set(toks(a)), set(toks(b))
    return 0.0 if (not sa or not sb) else len(sa & sb) / len(sa | sb)

def load_preds(paths):
    clean, pert = {}, {}
    n = 0
    for pat in paths:
        for fp in glob.glob(pat):
            for r in read_jsonl(fp):
                n += 1
                rat = extract_rationale(r.get("raw", ""))
                key = (r.get("model"), r.get("template"), r.get("id"))
                if r.get("condition") == "clean":
                    clean[key + ("CLEAN",)] = rat
                else:
                    pert[key + (r.get("attack_class"), r.get("seed"))] = rat
    if n == 0:
        sys.exit("No preds rows loaded — check --preds path/glob.")
    return clean, pert

def rank_value(gap, mode):
    vals = [gap[k] for k, _ in METRICS]
    if mode == "min":     return min(vals)
    if mode == "jaccard": return gap["jaccard"]
    return sum(vals) / len(vals)

def fmt_block(rec, cr, tr, br, jt, jb, ok_t, ok_b):
    t, b, g = rec["true"], rec["base"], rec["gap"]
    L = ["=" * 78,
         f"model={rec['model']}  template={rec['template']}  attack={rec['attack']}  "
         f"seed={rec['seed']}  pred={rec['pred_label']}",
         f"anchor={rec['anchor_id']}  true_partner={rec['true_partner_id']}  "
         f"baseline_partner={rec['baseline_partner_id']}",
         "-" * 78,
         "A_i  (clean rationale):", "   " + cr,
         "A_i' (perturbed rationale, same message):", "   " + tr,
         "B_i  (matched same-label baseline, DIFFERENT message):", "   " + br,
         "-" * 78,
         f"{'metric':<18}{'S_true':>10}{'S_base':>10}{'G_i':>10}"]
    for k, label in METRICS:
        L.append(f"{label:<18}{t[k]:>10.4f}{b[k]:>10.4f}{g[k]:>10.4f}")
    mean_gap = sum(g[k] for k, _ in METRICS) / len(METRICS)
    L.append(f"mean gap = {mean_gap:+.4f}")
    L.append("-" * 78)
    L.append(f"CONSISTENCY (recomputed Jaccard vs stored):")
    L.append(f"   true pair : {jt:.4f} vs {t['jaccard']:.4f}   {'OK' if ok_t else '*** MISMATCH ***'}")
    L.append(f"   baseline  : {jb:.4f} vs {b['jaccard']:.4f}   {'OK' if ok_b else '*** MISMATCH ***'}")
    L.append("=" * 78)
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/analysis/dim2_matched/matched_pairs.jsonl")
    ap.add_argument("--preds", nargs="*", default=["outputs/preds_norm_2110_final.jsonl"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--exclude-model", action="append", default=[],
                    help="drop pairs whose model contains this substring (repeatable)")
    ap.add_argument("--attack", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--label", default=None, choices=["smishing", "ham"])
    ap.add_argument("--rank", choices=["mean", "jaccard", "min"], default="mean")
    ap.add_argument("--minchars", type=int, default=80)
    ap.add_argument("--exclude-substr", default=CANNED)
    ap.add_argument("--nonidentical", action="store_true")
    ap.add_argument("--require-consistent", dest="require_consistent",
                    action="store_true", default=True)
    ap.add_argument("--no-require-consistent", dest="require_consistent",
                    action="store_false")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--topk", type=int, default=1)
    args = ap.parse_args()

    pairs = list(read_jsonl(args.pairs))
    def facet_ok(r):
        if args.model    and r["model"]      != args.model:    return False
        if any(x in r["model"] for x in args.exclude_model):   return False
        if args.attack   and r["attack"]     != args.attack:   return False
        if args.template and r["template"]   != args.template: return False
        if args.label    and r["pred_label"] != args.label:    return False
        return all(r["gap"][k] > 0 for k, _ in METRICS)
    pairs = [r for r in pairs if facet_ok(r)]
    if not pairs:
        sys.exit("No pairs pass the all-positive-gap + facet filter.")

    clean, pert = load_preds(args.preds)
    def texts(r):
        a  = clean.get((r["model"], r["template"], r["anchor_id"], "CLEAN"))
        tp = pert.get((r["model"], r["template"], r["true_partner_id"], r["attack"], r["seed"]))
        bp = pert.get((r["model"], r["template"], r["baseline_partner_id"], r["attack"], r["seed"]))
        return a, tp, bp

    cands = []
    for r in pairs:
        a, tp, bp = texts(r)
        if not (a and tp and bp):                              continue
        if len(a) < args.minchars:                             continue
        if args.exclude_substr and args.exclude_substr in a:   continue
        jt = jaccard(a, tp); jb = jaccard(a, bp)
        ok_t = abs(jt - r["true"]["jaccard"]) <= args.tol
        ok_b = abs(jb - r["base"]["jaccard"]) <= args.tol
        if args.nonidentical and jt >= 0.9999:                 continue
        if args.require_consistent and not (ok_t and ok_b):    continue
        cands.append((rank_value(r["gap"], args.rank), r, a, tp, bp, jt, jb, ok_t, ok_b))

    if not cands:
        sys.exit("No candidates. If you used --require-consistent (default) and got "
                 "nothing, the preds file likely differs from the one that built "
                 "matched_pairs.jsonl. Re-run with --no-require-consistent to inspect, "
                 "or point --preds at the correct preds file.")

    cands.sort(key=lambda x: x[0], reverse=True)
    chosen = cands[: args.topk]
    print(f"\n{len(cands)} candidates; showing top {len(chosen)} by {args.rank} gap "
          f"(consistency gate {'ON' if args.require_consistent else 'OFF'}):\n")
    out = []
    for score, r, a, tp, bp, jt, jb, ok_t, ok_b in chosen:
        print(fmt_block(r, a, tp, bp, jt, jb, ok_t, ok_b) + "\n")
        out.append({**r, "clean_rationale": a, "perturbed_rationale": tp,
                    "baseline_rationale": bp,
                    "recomputed_jaccard_true": jt, "recomputed_jaccard_base": jb,
                    "text_matches_scores": bool(ok_t and ok_b)})
    with open("rationale_example_out.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("Wrote rationale_example_out.json")

if __name__ == "__main__":
    main()
