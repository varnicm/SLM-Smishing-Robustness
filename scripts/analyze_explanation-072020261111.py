"""STAGE 3 — Dimension 2: explanation stability, compliance, and a MATCHED baseline.

The same-label baseline is constructed one-to-one from the eligible true pairs,
so both populations use EXACTLY the same observations (same anchor rationale,
same weighting across attacks, seeds, models, templates, and repeated messages).

For every eligible true pair:
    clean rationale A   vs   its own perturbed rationale A'      (true comparison)
we build one matched baseline pair using the SAME anchor A:
    clean rationale A   vs   an unrelated clean rationale B      (baseline comparison)
where B comes from a DIFFERENT message with the same model, explanation template,
and predicted label. The true pair's attack class and seed are copied onto the
matched baseline record, so the two sides are identically weighted.

Eligibility (a true pair enters the matched set only if all hold):
  * both-correct: model right on clean AND on perturbed (controls for correctness)
  * both rationales nonempty (>= --min-chars; empty-rationale exclusion)
  * a valid unrelated partner B exists (same model/template/pred, other message)
Pairs with no valid B are excluded from BOTH sides and counted.

Per matched observation and per metric we record:
    similarity(A, A')  ,  similarity(A, B)  ,  pairwise gap = sim(A,A') - sim(A,B)
for BERTScore F1, sentence-cosine (SentCos), ROUGE-L, and token-Jaccard.

Reported for pooled / by model / by (model, template) / by (model, attack):
  * pairwise-gap mean, median, std, n
  * matched true-pair mean and matched baseline mean

Validation: n_matched_true == n_matched_baseline == n_pairwise_gaps.

Scoring is shared with tests/test_bertscore_baseline.py via scripts/rationale_metrics.py.
NON-DESTRUCTIVE: writes to a fresh --out-dir and refuses to overwrite a non-empty
one unless --force. Reads predictions only; never touches preds_*.jsonl.

Usage
-----
  python scripts/analyze_explanation.py \
      --preds outputs/expl_all.jsonl \
      --out-dir outputs/analysis/dim2_matched --min-chars 10 --seed 0 --batch-size 16
  # all four metrics compute by default; disable any with --no-bertscore etc.
"""

import argparse
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

from rationale_metrics import (
    read_jsonl, extract_rationale,
    BERTScorer, SentCos, RougeL, score_pairs,
)

ALL_METRICS = ["bertscore_f1", "sentcos", "rougel_f1", "jaccard"]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_rows(path, min_chars):
    """Explanation-required rows (t3*/t4*) with rationale + has_rationale + pred."""
    rows = []
    for r in read_jsonl(path):
        tmpl = r.get("template", "")
        if "expl" not in tmpl and "cot" not in tmpl:
            continue
        rat = extract_rationale(r.get("raw", ""))
        r["rationale"] = rat
        r["has_rationale"] = len(rat) >= min_chars
        r["pred"] = (r.get("pred") or "").strip().lower()
        rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# (A) Compliance (unchanged)
# --------------------------------------------------------------------------- #
def compliance_table(rows):
    tot, empty = defaultdict(int), defaultdict(int)
    for r in rows:
        key = (r["model"], r["template"])
        tot[key] += 1
        if not r["has_rationale"]:
            empty[key] += 1
    out = {}
    for key in sorted(tot):
        n, e = tot[key], empty[key]
        out[f"{key[0]}|{key[1]}"] = {
            "n_generations": n, "empty": e,
            "empty_rate": round(e / n, 4), "compliance_rate": round(1 - e / n, 4)}
    return out


# --------------------------------------------------------------------------- #
# Matched true + baseline construction
# --------------------------------------------------------------------------- #
def build_matched(rows, rng):
    """Return (true_list, base_list, stats). true_list[i] and base_list[i] share
    the same anchor A and the same (model, template, attack, seed, pred_label).
    """
    clean = {}                       # (model, template, id) -> clean row
    pool = defaultdict(list)         # (model, template, pred) -> [(id, rationale)]
    perturbed = []
    for r in rows:
        cond = r.get("condition")
        if cond == "clean":
            clean[(r["model"], r["template"], r["id"])] = r
            if r["has_rationale"] and r["pred"] in ("smishing", "ham"):
                pool[(r["model"], r["template"], r["pred"])].append((r["id"], r["rationale"]))
        elif cond == "perturbed":
            perturbed.append(r)

    # deterministic order so RNG draws are reproducible
    perturbed.sort(key=lambda r: (r["model"], r["template"], r["id"],
                                  str(r.get("attack_class")), str(r.get("seed"))))
    for k in pool:
        pool[k].sort()

    both_correct = excluded_empty = excluded_no_partner = 0
    true_list, base_list = [], []
    for pr in perturbed:
        cr = clean.get((pr["model"], pr["template"], pr["id"]))
        if cr is None:
            continue
        true_label = pr.get("true_label", cr.get("true_label"))
        if not (cr["pred"] == true_label and pr["pred"] == true_label):
            continue                                   # not both-correct
        both_correct += 1
        if not (cr["has_rationale"] and pr["has_rationale"]):
            excluded_empty += 1                        # empty-rationale exclusion
            continue
        pred_label = true_label                        # both-correct => pred == true_label
        candidates = [(cid, crat) for (cid, crat)
                      in pool.get((pr["model"], pr["template"], pred_label), [])
                      if cid != pr["id"]]
        if not candidates:
            excluded_no_partner += 1                   # exclude from BOTH sides
            continue
        b_id, b_rat = rng.choice(candidates)
        common = {"model": pr["model"], "template": pr["template"],
                  "attack": pr.get("attack_class"), "seed": pr.get("seed"),
                  "anchor_id": pr["id"], "pred_label": pred_label}
        true_list.append({**common, "partner_id": pr["id"],
                          "text_a": cr["rationale"], "text_b": pr["rationale"]})
        base_list.append({**common, "partner_id": b_id,
                          "text_a": cr["rationale"], "text_b": b_rat})

    stats = {
        "both_correct_pairs": both_correct,
        "excluded_empty_pairs": excluded_empty,
        "eligible_after_empty": both_correct - excluded_empty,
        "excluded_no_partner": excluded_no_partner,
        "n_matched": len(true_list),
    }
    return true_list, base_list, stats


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _finite(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _describe(xs):
    xs = _finite(xs)
    if not xs:
        return {"mean": None, "median": None, "std": None, "n": 0}
    return {
        "mean": round(st.mean(xs), 4),
        "median": round(st.median(xs), 4),
        "std": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0,
        "n": len(xs),
    }


def group_stats(records, metrics, keyfn):
    """{group_label: {metric: {gap:{mean,median,std,n}, true_mean, base_mean}}}."""
    groups = defaultdict(list)
    for r in records:
        groups[keyfn(r)].append(r)
    out = {}
    for label in sorted(groups, key=lambda t: tuple(str(x) for x in (t if isinstance(t, tuple) else (t,)))):
        recs = groups[label]
        key = label if isinstance(label, str) else "|".join(str(x) for x in label)
        entry = {"n_matched": len(recs)}
        for m in metrics:
            gap = _describe([r["gap"][m] for r in recs])
            entry[m] = {
                "pairwise_gap": gap,
                "matched_true_mean": _describe([r["true"][m] for r in recs])["mean"],
                "matched_baseline_mean": _describe([r["base"][m] for r in recs])["mean"],
            }
        out[key] = entry
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="outputs/expl_all.jsonl")
    ap.add_argument("--out-dir", default="outputs/analysis/dim2_matched")
    ap.add_argument("--min-chars", type=int, default=10,
                    help="rationale must have >= this many chars (10 = near-empty cutoff)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--no-sentcos", action="store_true")
    ap.add_argument("--no-rougel", action="store_true")
    ap.add_argument("--no-jaccard", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a non-empty --out-dir")
    args = ap.parse_args()

    # non-destructive guard
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.force:
        raise SystemExit(f"[abort] {args.out_dir} exists and is non-empty. "
                         f"Use a different --out-dir or pass --force.")
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = [m for m, off in [
        ("bertscore_f1", args.no_bertscore), ("sentcos", args.no_sentcos),
        ("rougel_f1", args.no_rougel), ("jaccard", args.no_jaccard)] if not off]
    if not metrics:
        raise SystemExit("All metrics disabled — enable at least one.")

    rng = random.Random(args.seed)
    rows = load_rows(args.preds, args.min_chars)
    if not rows:
        raise SystemExit("No explanation-required rows found in --preds.")
    print(f"[load] {len(rows)} explanation rows (t3/t4)")

    compliance = compliance_table(rows)

    true_list, base_list, stats = build_matched(rows, rng)
    print(f"[matched] both_correct={stats['both_correct_pairs']}  "
          f"empty_excluded={stats['excluded_empty_pairs']}  "
          f"no_partner_excluded={stats['excluded_no_partner']}  "
          f"matched={stats['n_matched']}")
    if stats["n_matched"] == 0:
        raise SystemExit("No matched pairs — nothing to score.")

    # shared scorer objects (one warm cache reused for both sides)
    bertscorer = BERTScorer(batch_size=args.batch_size) if "bertscore_f1" in metrics else None
    sentcos = SentCos() if "sentcos" in metrics else None
    rougel = RougeL() if "rougel_f1" in metrics else None
    jaccard_on = "jaccard" in metrics

    score_pairs(true_list, bertscorer, sentcos, rougel, jaccard_on)
    score_pairs(base_list, bertscorer, sentcos, rougel, jaccard_on)

    # assemble matched records with per-observation gaps.
    # NOTE: records carry only IDs, grouping variables, scores, and gaps — never
    # the rationale text — so matched_pairs.jsonl stays small at ~160k rows.
    matched = []
    for t, b in zip(true_list, base_list):
        assert t["text_a"] == b["text_a"], "anchor mismatch between true and baseline"
        rec = {k: t[k] for k in ("model", "template", "attack", "seed",
                                 "anchor_id", "pred_label")}
        rec["true_partner_id"] = t["partner_id"]
        rec["baseline_partner_id"] = b["partner_id"]
        rec["true"] = {m: t.get(m) for m in metrics}
        rec["base"] = {m: b.get(m) for m in metrics}
        rec["gap"] = {}
        for m in metrics:
            tv, bv = t.get(m), b.get(m)
            rec["gap"][m] = (tv - bv) if (tv is not None and bv is not None
                                          and not (isinstance(tv, float) and math.isnan(tv))
                                          and not (isinstance(bv, float) and math.isnan(bv))) else float("nan")
        matched.append(rec)
    # free the rationale strings now that scoring is done (keeps memory bounded)
    del true_list, base_list

    # validation
    validation = {
        "n_matched_true": len(true_list),
        "n_matched_baseline": len(base_list),
        "n_pairwise_gaps": len(matched),
        "ok": len(true_list) == len(base_list) == len(matched),
    }
    assert validation["ok"], validation

    groupings = {
        "pooled": group_stats(matched, metrics, lambda r: "ALL"),
        "by_model": group_stats(matched, metrics, lambda r: r["model"]),
        "by_model_template": group_stats(matched, metrics, lambda r: (r["model"], r["template"])),
        "by_model_attack": group_stats(matched, metrics, lambda r: (r["model"], r["attack"])),
    }

    report = {
        "config": vars(args),
        "metrics": metrics,
        "matched_stats": stats,
        "validation": validation,
        "compliance": compliance,
        "groupings": groupings,
        "note": ("A small positive gap suggests that much of the observed similarity "
                 "may reflect model-, prompt-, and label-conditioned response patterns "
                 "rather than message-specific consistency."),
    }
    out_json = os.path.join(args.out_dir, "explanation_dim2_matched.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # per-observation records (IDs + grouping + scores + gaps only; no text),
    # written incrementally with rounded floats to keep the file compact.
    def _round(d):
        return {k: (round(v, 6) if isinstance(v, float) and not math.isnan(v)
                    else (None if isinstance(v, float) and math.isnan(v) else v))
                for k, v in d.items()}

    out_pairs = os.path.join(args.out_dir, "matched_pairs.jsonl")
    with open(out_pairs, "w", encoding="utf-8") as fh:
        for r in matched:
            slim = {k: r[k] for k in ("model", "template", "attack", "seed",
                                      "anchor_id", "true_partner_id",
                                      "baseline_partner_id", "pred_label")}
            slim["true"] = _round(r["true"])
            slim["base"] = _round(r["base"])
            slim["gap"] = _round(r["gap"])
            fh.write(json.dumps(slim) + "\n")

    # ---- human-readable summary ----
    print("\n=== (A) Instruction compliance ===")
    for k, v in compliance.items():
        print(f"  {k:45s} compliance={v['compliance_rate']:.3f}  (empty {v['empty']}/{v['n_generations']})")

    print(f"\n=== Validation ===\n  n_matched_true=n_matched_baseline=n_pairwise_gaps="
          f"{validation['n_pairwise_gaps']}  ok={validation['ok']}")

    prim = metrics[0]
    print(f"\n=== Pooled pairwise gap  sim(A,A') - sim(A,B) ===")
    for m in metrics:
        g = groupings["pooled"]["ALL"][m]["pairwise_gap"]
        tm = groupings["pooled"]["ALL"][m]["matched_true_mean"]
        bm = groupings["pooled"]["ALL"][m]["matched_baseline_mean"]
        print(f"  {m:14s} gap mean={g['mean']}  median={g['median']}  std={g['std']}  "
              f"n={g['n']}   (true={tm}  baseline={bm})")

    print(f"\n=== Pairwise gap by model — {prim} ===")
    print(f"  {'model':22s} {'gap_mean':>9s} {'gap_med':>8s} {'gap_std':>8s} "
          f"{'true':>7s} {'base':>7s} {'n':>6s}")
    for mdl, e in groupings["by_model"].items():
        g = e[prim]["pairwise_gap"]
        print(f"  {mdl:22s} {str(g['mean']):>9s} {str(g['median']):>8s} "
              f"{str(g['std']):>8s} {str(e[prim]['matched_true_mean']):>7s} "
              f"{str(e[prim]['matched_baseline_mean']):>7s} {g['n']:>6d}")

    print("\n  A small positive gap suggests that much of the observed similarity may")
    print("  reflect model-, prompt-, and label-conditioned response patterns rather")
    print("  than message-specific consistency.")
    print(f"\n[write] {out_json}\n[write] {out_pairs}")


if __name__ == "__main__":
    main()
