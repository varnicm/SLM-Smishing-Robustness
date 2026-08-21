"""STAGE 3 — Dimension 2: explanation stability, compliance, and the baseline.

Reports, in one run and side by side:

  (A) Instruction compliance — per (model, template), the share of
      explanation-required generations that actually produced a rationale.
      Empty (near-empty) rationales are the finding, not noise: one model omits
      the rationale on 80%+ of neutral prompts.

  (B) True-pair stability — on the BOTH-CORRECT subset (model right on clean AND
      perturbed), each clean rationale vs ITS OWN perturbed rationale. This is
      the "does the justification survive perturbation" number. Empty rationales
      excluded before scoring; the exclusion rate is reported.

  (C) Same-label baseline (the control) — each nonempty rationale vs one from a
      DIFFERENT message sharing the same model, template, and predicted label.

  (D) Side-by-side (B) vs (C) with the gap. A small gap means the high stability
      is label-conditioned boilerplate, not preserved message-specific evidence,
      so we report similarity as descriptive only.

Metrics: BERTScore F1 (primary), plus optional SentCos / ROUGE-L / token-Jaccard.
Scoring is shared with tests/test_bertscore_baseline.py via scripts/rationale_metrics.py.

NON-DESTRUCTIVE: writes to a fresh --out-dir and REFUSES to overwrite an existing
non-empty directory unless --force is given. It only READS the predictions; it
never touches preds_*.jsonl or any prior analysis outputs.

Input row schema (expl_all.jsonl): model, template, id, condition(clean|perturbed),
attack_class, seed, true_label, pred, raw.

Usage
-----
  python scripts/analyze_explanation.py \
      --preds outputs/explanations/expl_all.jsonl \
      --out-dir outputs/analysis/dim2 \
      --bertscore --sentcos --rougel --min-chars 10 --seed 0
"""

import argparse
import json
import os
import random
import statistics as st
from collections import defaultdict

from rationale_metrics import (
    read_jsonl, extract_rationale, jaccard,
    BERTScorer, SentCos, RougeL, score_pairs,
)

METRIC_KEYS = ["bertscore_f1", "sentcos", "rougel_f1", "jaccard"]


# --------------------------------------------------------------------------- #
# Load: index rows and attach nonempty rationale text
# --------------------------------------------------------------------------- #
def load_rows(path, min_chars):
    """Return list of rows with 'rationale' and 'has_rationale' attached.

    Only keeps explanation-required templates (t3*/t4*, i.e. name contains
    'expl' or 'cot'); label-only templates carry no rationale by design.
    """
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
# (A) Compliance
# --------------------------------------------------------------------------- #
def compliance_table(rows):
    tot = defaultdict(int)
    empty = defaultdict(int)
    for r in rows:
        key = (r["model"], r["template"])
        tot[key] += 1
        if not r["has_rationale"]:
            empty[key] += 1
    out = {}
    for key in sorted(tot):
        n, e = tot[key], empty[key]
        out[f"{key[0]}|{key[1]}"] = {
            "n_generations": n,
            "empty": e,
            "empty_rate": round(e / n, 4),
            "compliance_rate": round(1 - e / n, 4),
        }
    return out


# --------------------------------------------------------------------------- #
# (B) True-pair stability on the both-correct subset
# --------------------------------------------------------------------------- #
def build_true_pairs(rows):
    """Pair each perturbed rationale with its own clean rationale, keeping only
    (both classified correctly) AND (both rationales nonempty).

    Returns (pairs, stats) where stats records eligibility and empty exclusions.
    """
    clean = {}       # (model, template, id) -> row
    perturbed = []   # rows with condition == perturbed
    for r in rows:
        if r.get("condition") == "clean":
            clean[(r["model"], r["template"], r["id"])] = r
        elif r.get("condition") == "perturbed":
            perturbed.append(r)

    both_correct = 0
    excluded_empty = 0
    pairs = []
    for pr in perturbed:
        cr = clean.get((pr["model"], pr["template"], pr["id"]))
        if cr is None:
            continue
        tl = pr.get("true_label", cr.get("true_label"))
        if not (cr["pred"] == tl and pr["pred"] == tl):
            continue  # not both-correct
        both_correct += 1
        if not (cr["has_rationale"] and pr["has_rationale"]):
            excluded_empty += 1
            continue
        pairs.append({
            "model": pr["model"], "template": pr["template"],
            "attack": pr.get("attack_class"), "seed": pr.get("seed"),
            "id": pr["id"], "text_a": cr["rationale"], "text_b": pr["rationale"],
        })
    stats = {
        "both_correct_pairs": both_correct,
        "excluded_empty_pairs": excluded_empty,
        "empty_exclusion_rate": round(excluded_empty / both_correct, 4) if both_correct else None,
        "scored_pairs": len(pairs),
    }
    return pairs, stats


# --------------------------------------------------------------------------- #
# (C) Same-label baseline: different message, same model+template+pred label
# --------------------------------------------------------------------------- #
def build_baseline_pairs(rows, condition, rng):
    groups = defaultdict(dict)  # (model, template, pred) -> {id: rationale}
    for r in rows:
        if condition and r.get("condition") != condition:
            continue
        if not r["has_rationale"] or r["pred"] not in ("smishing", "ham"):
            continue
        groups[(r["model"], r["template"], r["pred"])].setdefault(r["id"], r["rationale"])

    pairs = []
    for (model, template, pred), by_msg in groups.items():
        ids = list(by_msg)
        if len(ids) < 2:
            continue
        for mid in ids:
            partner = rng.choice([o for o in ids if o != mid])
            pairs.append({
                "model": model, "template": template, "pred": pred,
                "id": mid, "partner": partner,
                "text_a": by_msg[mid], "text_b": by_msg[partner],
            })
    return pairs


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _mean(vals):
    vals = [v for v in vals if v == v]  # drop NaN
    return round(st.mean(vals), 4) if vals else None


def aggregate(pairs, group_keys):
    """Mean of each metric per group tuple and pooled."""
    buckets = defaultdict(lambda: defaultdict(list))
    pooled = defaultdict(list)
    for p in pairs:
        gk = tuple(p.get(k) for k in group_keys)
        for m in METRIC_KEYS:
            if m in p:
                buckets[gk][m].append(p[m])
                pooled[m].append(p[m])
    per_group = {}
    for gk in sorted(buckets, key=lambda t: tuple(str(x) for x in t)):
        label = "|".join(str(x) for x in gk)
        entry = {"n": len(next(iter(buckets[gk].values())))}
        for m in METRIC_KEYS:
            if buckets[gk][m]:
                entry[m] = _mean(buckets[gk][m])
        per_group[label] = entry
    pooled_out = {"n": len(pairs)}
    for m in METRIC_KEYS:
        if pooled[m]:
            pooled_out[m] = _mean(pooled[m])
    return {"per_group": per_group, "pooled": pooled_out}


def side_by_side(true_agg, base_agg, metric="bertscore_f1"):
    """Per-model true-pair vs baseline mean and the gap (pooled over templates)."""
    def by_model(agg):
        acc = defaultdict(list)
        for label, e in agg["per_group"].items():
            model = label.split("|")[0]
            if metric in e:
                acc[model].append((e[metric], e["n"]))
        # n-weighted mean per model
        out = {}
        for model, rows in acc.items():
            num = sum(v * n for v, n in rows)
            den = sum(n for _, n in rows)
            out[model] = round(num / den, 4) if den else None
        return out

    tp, bl = by_model(true_agg), by_model(base_agg)
    models = sorted(set(tp) | set(bl))
    table = {}
    for mdl in models:
        t, b = tp.get(mdl), bl.get(mdl)
        gap = round(t - b, 4) if (t is not None and b is not None) else None
        table[mdl] = {"true_pair": t, "same_label_baseline": b, "gap": gap}
    return table


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="outputs/explanations/expl_all.jsonl")
    ap.add_argument("--out-dir", default="outputs/analysis/dim2",
                    help="fresh directory for this report (not overwritten by default)")
    ap.add_argument("--baseline-condition", default="clean",
                    help="which rows feed the same-label baseline pool (default clean; "
                         "'' for all)")
    ap.add_argument("--min-chars", type=int, default=10,
                    help="rationale must have >= this many chars (10 = near-empty cutoff)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bertscore", action="store_true", help="compute BERTScore F1 (primary)")
    ap.add_argument("--sentcos", action="store_true")
    ap.add_argument("--rougel", action="store_true")
    ap.add_argument("--no-jaccard", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a non-empty --out-dir (default: refuse)")
    args = ap.parse_args()

    # --- non-destructive guard ---------------------------------------------- #
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.force:
        raise SystemExit(
            f"[abort] {args.out_dir} exists and is non-empty. Refusing to overwrite "
            f"existing results. Use a different --out-dir or pass --force."
        )
    os.makedirs(args.out_dir, exist_ok=True)

    if not (args.bertscore or args.sentcos or args.rougel or not args.no_jaccard):
        raise SystemExit("No metric selected. Enable at least one of "
                         "--bertscore/--sentcos/--rougel (or leave jaccard on).")

    rng = random.Random(args.seed)
    rows = load_rows(args.preds, args.min_chars)
    if not rows:
        raise SystemExit("No explanation-required rows found in --preds.")
    print(f"[load] {len(rows)} explanation rows (t3/t4)")

    # (A) compliance
    compliance = compliance_table(rows)

    # scorers (built once, caches shared across (B) and (C))
    bertscorer = BERTScorer(batch_size=args.batch_size) if args.bertscore else None
    sentcos = SentCos() if args.sentcos else None
    rougel = RougeL() if args.rougel else None
    jaccard_on = not args.no_jaccard

    # (B) true-pair stability
    true_pairs, true_stats = build_true_pairs(rows)
    print(f"[true-pair] both-correct={true_stats['both_correct_pairs']}  "
          f"empty-excluded={true_stats['excluded_empty_pairs']} "
          f"({true_stats['empty_exclusion_rate']})  scored={true_stats['scored_pairs']}")
    score_pairs(true_pairs, bertscorer, sentcos, rougel, jaccard_on)
    true_by_model = aggregate(true_pairs, ["model"])
    true_by_model_attack = aggregate(true_pairs, ["model", "attack"])

    # (C) same-label baseline
    base_pairs = build_baseline_pairs(rows, args.baseline_condition or None, rng)
    print(f"[baseline] {len(base_pairs)} same-label / different-message pairs")
    score_pairs(base_pairs, bertscorer, sentcos, rougel, jaccard_on)
    base_by_model = aggregate(base_pairs, ["model"])
    base_by_template = aggregate(base_pairs, ["model", "template"])

    # (D) side-by-side (BERTScore if available, else first enabled metric)
    metric = ("bertscore_f1" if args.bertscore else
              "sentcos" if args.sentcos else
              "rougel_f1" if args.rougel else "jaccard")
    comparison = side_by_side(true_by_model, base_by_model, metric=metric)

    report = {
        "config": vars(args),
        "primary_metric": metric,
        "compliance": compliance,
        "true_pair": {"stats": true_stats,
                      "by_model": true_by_model,
                      "by_model_attack": true_by_model_attack},
        "same_label_baseline": {"by_model": base_by_model,
                                "by_model_template": base_by_template},
        "true_vs_baseline": comparison,
    }
    out_json = os.path.join(args.out_dir, "explanation_dim2.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # human-readable summary
    print("\n=== (A) Instruction compliance (share producing a rationale) ===")
    for k, v in compliance.items():
        print(f"  {k:45s} compliance={v['compliance_rate']:.3f}  (empty {v['empty']}/{v['n_generations']})")

    print(f"\n=== (D) True-pair vs same-label baseline — {metric} (per model) ===")
    print(f"  {'model':22s} {'true-pair':>10s} {'baseline':>10s} {'gap':>8s}")
    for mdl, v in sorted(comparison.items()):
        t = "-" if v["true_pair"] is None else f"{v['true_pair']:.3f}"
        b = "-" if v["same_label_baseline"] is None else f"{v['same_label_baseline']:.3f}"
        g = "-" if v["gap"] is None else f"{v['gap']:+.3f}"
        print(f"  {mdl:22s} {t:>10s} {b:>10s} {g:>8s}")
    print("\n  A small gap => high similarity is label-conditioned boilerplate, "
          "not preserved\n  message-specific justification (report as descriptive only).")
    print(f"\n[write] {out_json}")


if __name__ == "__main__":
    main()
