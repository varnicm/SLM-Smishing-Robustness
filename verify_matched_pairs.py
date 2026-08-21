#!/usr/bin/env python3
"""
verify_matched_pairs.py
-----------------------
Full audit that matched_pairs.jsonl is internally consistent with the preds file
it was scored from. Verifies, beyond Jaccard:

  1. GAP ARITHMETIC  (all rows)      stored gap[m] == true[m] - base[m], 4 metrics
  2. PARTNER RELATIONS (all rows)    true_partner_id == anchor_id;
                                     baseline_partner_id != anchor_id;
                                     anchor/true/baseline rows all present in preds
  3. PREDICTED LABELS (all rows)     clean & perturbed correct (pred==true_label)
                                     and == pair.pred_label; baseline same pred_label
  4. METRIC RECOMPUTE (sample)       ROUGE-L F1 + Jaccard always; BERTScore F1 +
                                     MiniLM cosine with --neural; each vs stored
                                     true AND base, using YOUR rationale_metrics.

Run (from repo root):
  python verify_matched_pairs.py                       # checks 1-3 + lexical recompute
  python verify_matched_pairs.py --neural --sample 300 # + BERTScore & cosine (needs GPU/cached models)

Defaults:
  --pairs outputs/analysis/dim2_matched/matched_pairs.jsonl
  --preds outputs/expl_all.jsonl
"""
import argparse, glob, json, sys

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
try:
    import rationale_metrics as rm
    extract_rationale = rm.extract_rationale
    jaccard = rm.jaccard
    _rouge = rm.RougeL()
    rougeL = lambda a, b: _rouge.f1(a, b)
    HAVE_RM = True
except Exception as e:
    print(f"[warn] could not import scripts/rationale_metrics.py ({e}); "
          f"using built-in ROUGE-L/Jaccard (definitions should still match).")
    import re
    HAVE_RM = False
    _TOK = re.compile(r"[a-z0-9]+")
    _FINAL = re.compile(r"FINAL:\s*(smishing|ham)", re.IGNORECASE)
    def extract_rationale(raw):
        if not raw: return ""
        m = _FINAL.search(raw); return (raw[:m.start()] if m else raw).strip()
    def _toks(s): return _TOK.findall((s or "").lower())
    def jaccard(a, b):
        sa, sb = set(_toks(a)), set(_toks(b))
        return 0.0 if (not sa or not sb) else len(sa & sb) / len(sa | sb)
    def _lcs(x, y):
        n, m = len(x), len(y); dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                dp[i][j] = dp[i-1][j-1]+1 if x[i-1]==y[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[n][m]
    def rougeL(a, b):
        ca, cb = _toks(a), _toks(b)
        if not ca or not cb: return 0.0
        l = _lcs(ca, cb); p = l/len(ca); r = l/len(cb)
        return 0.0 if p+r == 0 else 2*p*r/(p+r)

METRICS = ["bertscore_f1", "sentcos", "rougel_f1", "jaccard"]

def read_jsonl(path):
    for pat in ([path] if isinstance(path, str) else path):
        for fp in glob.glob(pat):
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

def load_preds(paths):
    clean, pert = {}, {}
    n = 0
    for r in read_jsonl(paths):
        n += 1
        key = (r.get("model"), r.get("template"), r.get("id"))
        row = {"rat": extract_rationale(r.get("raw", "")),
               "pred": r.get("pred"), "true_label": r.get("true_label")}
        if r.get("condition") == "clean":
            clean[key + ("CLEAN",)] = row
        else:
            pert[key + (r.get("attack_class"), r.get("seed"))] = row
    if n == 0:
        sys.exit("No preds rows loaded — check --preds.")
    return clean, pert

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/analysis/dim2_matched/matched_pairs.jsonl")
    ap.add_argument("--preds", nargs="*", default=["outputs/expl_all.jsonl"])
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--tol-lex", type=float, default=1e-4)
    ap.add_argument("--tol-neural", type=float, default=2e-3)
    ap.add_argument("--tol-gap", type=float, default=1e-6)
    args = ap.parse_args()

    pairs = list(read_jsonl(args.pairs))
    print(f"Loaded {len(pairs)} matched pairs from {args.pairs}")
    clean, pert = load_preds(args.preds)
    print(f"Indexed preds: {len(clean)} clean rows, {len(pert)} perturbed rows\n")

    # ---- 1. GAP ARITHMETIC (all rows) ----
    gap_bad, gap_maxdev = 0, 0.0
    for r in pairs:
        for m in METRICS:
            dev = abs(r["gap"][m] - (r["true"][m] - r["base"][m]))
            gap_maxdev = max(gap_maxdev, dev)
            if dev > args.tol_gap:
                gap_bad += 1
    print(f"[1] Gap arithmetic (all {len(pairs)} rows x4 metrics): "
          f"{'PASS' if gap_bad==0 else 'FAIL'}  (violations={gap_bad}, max dev={gap_maxdev:.2e})")

    # ---- 2 & 3. PARTNERS + LABELS (all rows) ----
    self_true = sum(1 for r in pairs if r["true_partner_id"] != r["anchor_id"])
    self_base = sum(1 for r in pairs if r["baseline_partner_id"] == r["anchor_id"])
    miss_join = lbl_bad = lbl_checked = 0
    for r in pairs:
        A = clean.get((r["model"], r["template"], r["anchor_id"], "CLEAN"))
        P = pert.get((r["model"], r["template"], r["true_partner_id"], r["attack"], r["seed"]))
        B = pert.get((r["model"], r["template"], r["baseline_partner_id"], r["attack"], r["seed"]))
        if not (A and P and B):
            miss_join += 1
            continue
        lbl_checked += 1
        ok = (A["pred"] == r["pred_label"] and P["pred"] == r["pred_label"]
              and B["pred"] == r["pred_label"]
              and A["pred"] == A["true_label"] and P["pred"] == P["true_label"])
        if not ok:
            lbl_bad += 1
    print(f"[2] Partner relations: true_partner==anchor {'PASS' if self_true==0 else 'FAIL'} "
          f"(bad={self_true}); baseline!=anchor {'PASS' if self_base==0 else 'FAIL'} "
          f"(bad={self_base}); rows joinable to preds: {len(pairs)-miss_join}/{len(pairs)} "
          f"(missing={miss_join})")
    print(f"[3] Predicted labels (clean&perturbed correct, ==pair label, baseline same label): "
          f"{'PASS' if lbl_bad==0 else 'FAIL'}  (checked={lbl_checked}, violations={lbl_bad})\n")

    # ---- 4. METRIC RECOMPUTE (sample) ----
    joinable = []
    for r in pairs:
        A = clean.get((r["model"], r["template"], r["anchor_id"], "CLEAN"))
        P = pert.get((r["model"], r["template"], r["true_partner_id"], r["attack"], r["seed"]))
        B = pert.get((r["model"], r["template"], r["baseline_partner_id"], r["attack"], r["seed"]))
        if A and P and B:
            joinable.append((r, A["rat"], P["rat"], B["rat"]))
    stride = max(1, len(joinable) // args.sample)
    sample = joinable[::stride][:args.sample]
    print(f"[4] Metric recompute on {len(sample)} sampled pairs "
          f"(lexical always; neural={'ON' if args.neural else 'OFF'}):")

    funcs = {"rougel_f1": rougeL, "jaccard": jaccard}
    if args.neural:
        bert_f1, cosine = _load_neural()
        funcs["bertscore_f1"] = bert_f1
        funcs["sentcos"] = cosine

    stats = {m: {"n": 0, "max": 0.0, "bad": 0} for m in funcs}
    for r, a, p, b in sample:
        for m, fn in funcs.items():
            tol = args.tol_neural if m in ("bertscore_f1", "sentcos") else args.tol_lex
            for stored, x, y in ((r["true"][m], a, p), (r["base"][m], a, b)):
                dev = abs(fn(x, y) - stored)
                st = stats[m]; st["n"] += 1; st["max"] = max(st["max"], dev)
                if dev > tol: st["bad"] += 1
    for m in ("bertscore_f1", "sentcos", "rougel_f1", "jaccard"):
        if m not in stats: continue
        s = stats[m]
        print(f"    {m:<13} {'PASS' if s['bad']==0 else 'FAIL'}  "
              f"(compared={s['n']}, violations={s['bad']}, max dev={s['max']:.2e})")

    print("\nDone. Any FAIL above pinpoints exactly what to investigate.")

def _load_neural():
    """BERTScore F1 (roberta-large L17) + MiniLM cosine, matching the paper."""
    # cosine via all-MiniLM-L6-v2 (same encoder the pipeline uses)
    from sentence_transformers import SentenceTransformer
    import numpy as np
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    def cosine(a, b):
        e = enc.encode([a, b], normalize_embeddings=True)
        return float(np.dot(e[0], e[1]))
    # BERTScore: prefer YOUR BERTScorer; fall back to pip bert-score with same config
    bert_f1 = None
    if HAVE_RM and hasattr(rm, "BERTScorer"):
        try:
            bs = rm.BERTScorer()
            for name in ("f1", "score"):
                if hasattr(bs, name):
                    fn = getattr(bs, name)
                    def bert_f1(a, b, fn=fn):
                        out = fn([a], [b]) if name == "score" else fn(a, b)
                        # normalize possible return shapes -> float F1
                        if isinstance(out, (int, float)): return float(out)
                        if isinstance(out, dict): return float(out.get("f1", list(out.values())[-1]))
                        if isinstance(out, (list, tuple)):
                            v = out[-1] if len(out) == 3 else out  # (P,R,F) -> F
                            try: return float(v[0])
                            except Exception: return float(v)
                        return float(out)
                    _ = bert_f1("test one", "test two")  # smoke test
                    break
        except Exception as e:
            print(f"    [warn] your BERTScorer API not auto-detected ({e}); "
                  f"public methods = {[x for x in dir(rm.BERTScorer) if not x.startswith('_')]}")
            bert_f1 = None
    if bert_f1 is None:
        from bert_score import score as bscore
        def bert_f1(a, b):
            P, R, F = bscore([a], [b], model_type="roberta-large", num_layers=17,
                             rescale_with_baseline=False, idf=False, lang="en", verbose=False)
            return float(F[0])
    return bert_f1, cosine

if __name__ == "__main__":
    main()
