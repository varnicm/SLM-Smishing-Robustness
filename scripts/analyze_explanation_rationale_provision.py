"""STAGE 3 — Dimension 2: rationale provision, explanation stability, and a matched baseline.

The same rationale extractor is used for all three analyses:
  1. rationale provision,
  2. true-pair explanation stability, and
  3. the matched same-label baseline.

For every eligible true pair:
    clean rationale A_i  vs. its own perturbed rationale A'_i
we construct one matched baseline pair using the same clean anchor:
    clean rationale A_i  vs. an unrelated perturbed rationale A'_j

The baseline partner comes from a different source message and shares the same
model, explanation-required template, predicted label, perturbation type, and
seed. Sentence paraphrases use seed=None because they are deterministic and
were deduplicated before analysis.

Eligibility requires both the clean and perturbed predictions to be correct and
both outputs to contain usable rationales. Baseline candidates are drawn only
from these eligible true pairs. One candidate is selected per pair using a fixed
random seed, and that assignment is held fixed for the analysis.

Per matched observation and metric, the script records:
    sim(A_i, A'_i), sim(A_i, A'_j), and gap = sim(A_i, A'_i) - sim(A_i, A'_j)
for BERTScore F1, sentence cosine, ROUGE-L F1, and token Jaccard.

NON-DESTRUCTIVE: writes to a fresh --out-dir and refuses to overwrite a
non-empty directory unless --force is supplied. Reads prediction files only.

Usage
-----
python scripts/analyze_explanation.py \
    --preds outputs/expl_all.jsonl \
    --out-dir outputs/analysis/dim2_matched_rationale_anywhere \
    --min-chars 10 --seed 0 --batch-size 16
"""

import argparse
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

from rationale_metrics import (
    read_jsonl,
    BERTScorer,
    SentCos,
    RougeL,
    score_pairs,
)
from rationale_extract import (
    baseline_key,
    extract_rationale_anywhere,
    has_leading_label_token,
    is_eligible_pair,
    normalize_label,
    normalized_seed,
)

ALL_METRICS = ["bertscore_f1", "sentcos", "rougel_f1", "jaccard"]


def valid_number(value):
    """Return True for a finite real number, including a legitimate 0.0."""
    return (
        value is not None
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def pairwise_gap(true_value, base_value):
    """Return sim(A_i,A'_i)-sim(A_i,A'_j), or None for invalid scores."""
    if valid_number(true_value) and valid_number(base_value):
        return true_value - base_value
    return None


# --------------------------------------------------------------------------- #
# Load and shared rationale extraction
# --------------------------------------------------------------------------- #
def load_rows(path, min_chars, raw_field):
    """Load explanation-required rows and apply the shared rationale extractor."""
    rows = []
    for row in read_jsonl(path):
        template = str(row.get("template", ""))
        if "expl" not in template.lower() and "cot" not in template.lower():
            continue

        raw_output = row.get(raw_field, "")
        rationale = extract_rationale_anywhere(raw_output, min_chars=min_chars)

        row["raw_output"] = raw_output
        row["rationale"] = rationale
        row["has_rationale"] = rationale is not None
        row["pred"] = normalize_label(row.get("pred"))
        row["true_label"] = normalize_label(row.get("true_label"))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Rationale provision and diagnostics
# --------------------------------------------------------------------------- #
def rationale_provision_table(rows):
    totals = defaultdict(int)
    missing = defaultdict(int)
    leading_label = defaultdict(int)

    for row in rows:
        key = (row["model"], row["template"])
        totals[key] += 1
        if not row["has_rationale"]:
            missing[key] += 1
        elif has_leading_label_token(row["rationale"]):
            leading_label[key] += 1

    out = {}
    for key in sorted(totals):
        n = totals[key]
        absent = missing[key]
        leading = leading_label[key]
        out[f"{key[0]}|{key[1]}"] = {
            "n_generations": n,
            "usable_rationales": n - absent,
            "missing_or_too_short": absent,
            "rationale_provision_rate": round((n - absent) / n, 4),
            "leading_label_token_count": leading,
            "leading_label_token_rate": round(leading / n, 4),
        }
    return out


# --------------------------------------------------------------------------- #
# Eligible true-pair and matched-baseline construction
# --------------------------------------------------------------------------- #
def build_matched(rows, rng, min_chars):
    """Build one true and one matched baseline pair for each retained observation.

    Baseline candidates are perturbed rationales from other *eligible* true pairs
    with the same model, template, canonical predicted label, attack, and
    normalized seed.
    """
    clean = {}
    perturbed = []

    for row in rows:
        condition = row.get("condition")
        key = (row["model"], row["template"], row["id"])
        if condition == "clean":
            clean[key] = row
        elif condition == "perturbed":
            perturbed.append(row)

    perturbed.sort(
        key=lambda r: (
            str(r["model"]),
            str(r["template"]),
            str(r["id"]),
            str(r.get("attack_class")),
            str(r.get("seed")),
        )
    )

    missing_clean = 0
    incorrect_pairs = 0
    unusable_pairs = 0
    eligible_records = []

    for pr in perturbed:
        cr = clean.get((pr["model"], pr["template"], pr["id"]))
        if cr is None:
            missing_clean += 1
            continue

        true_label = pr.get("true_label") or cr.get("true_label")
        eligible, clean_rationale, perturbed_rationale = is_eligible_pair(
            clean_pred=cr.get("pred"),
            perturbed_pred=pr.get("pred"),
            true_label=true_label,
            clean_output=cr.get("raw_output"),
            perturbed_output=pr.get("raw_output"),
            min_chars=min_chars,
        )

        canonical_gold = normalize_label(true_label)
        both_correct = (
            canonical_gold is not None
            and normalize_label(cr.get("pred")) == canonical_gold
            and normalize_label(pr.get("pred")) == canonical_gold
        )
        if not both_correct:
            incorrect_pairs += 1
            continue
        if not eligible:
            unusable_pairs += 1
            continue

        attack = pr.get("attack_class")
        seed = normalized_seed(attack, pr.get("seed"))
        match_key = baseline_key(
            pr["model"], pr["template"], canonical_gold, attack, seed
        )
        eligible_records.append(
            {
                "model": pr["model"],
                "template": pr["template"],
                "attack": match_key[3],
                "seed": match_key[4],
                "anchor_id": pr["id"],
                "pred_label": canonical_gold,
                "clean_rationale": clean_rationale,
                "perturbed_rationale": perturbed_rationale,
                "baseline_key": match_key,
            }
        )

    # Candidate pool consists only of perturbed rationales from eligible pairs.
    pool = defaultdict(list)
    for rec in eligible_records:
        pool[rec["baseline_key"]].append(
            (str(rec["anchor_id"]), rec["perturbed_rationale"])
        )
    for key in pool:
        pool[key].sort(key=lambda x: (x[0], x[1]))

    true_list = []
    base_list = []
    excluded_no_partner = 0

    for rec in eligible_records:
        candidates = [
            (candidate_id, candidate_rationale)
            for candidate_id, candidate_rationale in pool[rec["baseline_key"]]
            if candidate_id != str(rec["anchor_id"])
        ]
        if not candidates:
            excluded_no_partner += 1
            continue

        baseline_id, baseline_rationale = rng.choice(candidates)
        common = {
            "model": rec["model"],
            "template": rec["template"],
            "attack": rec["attack"],
            "seed": rec["seed"],
            "anchor_id": rec["anchor_id"],
            "pred_label": rec["pred_label"],
        }
        true_list.append(
            {
                **common,
                "partner_id": rec["anchor_id"],
                "text_a": rec["clean_rationale"],
                "text_b": rec["perturbed_rationale"],
            }
        )
        base_list.append(
            {
                **common,
                "partner_id": baseline_id,
                "text_a": rec["clean_rationale"],
                "text_b": baseline_rationale,
            }
        )

    stats = {
        "perturbed_rows": len(perturbed),
        "missing_clean_rows": missing_clean,
        "excluded_not_both_correct": incorrect_pairs,
        "both_correct_pairs": len(perturbed) - missing_clean - incorrect_pairs,
        "excluded_unusable_rationale_pairs": unusable_pairs,
        "eligible_before_baseline_matching": len(eligible_records),
        "excluded_no_partner": excluded_no_partner,
        "n_matched": len(true_list),
    }
    return true_list, base_list, stats


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _finite(xs):
    return [x for x in xs if valid_number(x)]


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
    groups = defaultdict(list)
    for record in records:
        groups[keyfn(record)].append(record)

    out = {}
    for label in sorted(
        groups,
        key=lambda value: tuple(
            str(x) for x in (value if isinstance(value, tuple) else (value,))
        ),
    ):
        recs = groups[label]
        key = label if isinstance(label, str) else "|".join(str(x) for x in label)
        entry = {"n_matched": len(recs)}
        for metric in metrics:
            entry[metric] = {
                "pairwise_gap": _describe([r["gap"][metric] for r in recs]),
                "matched_true_mean": _describe(
                    [r["true"][metric] for r in recs]
                )["mean"],
                "matched_baseline_mean": _describe(
                    [r["base"][metric] for r in recs]
                )["mean"],
            }
        out[key] = entry
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preds", default="outputs/expl_all.jsonl")
    parser.add_argument(
        "--out-dir",
        default="outputs/analysis/dim2_matched_rationale_anywhere",
    )
    parser.add_argument("--raw-field", default="raw")
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-sentcos", action="store_true")
    parser.add_argument("--no-rougel", action="store_true")
    parser.add_argument("--no-jaccard", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.min_chars < 0:
        raise SystemExit("--min-chars must be non-negative")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir) and not args.force:
        raise SystemExit(
            f"[abort] {args.out_dir} exists and is non-empty. "
            "Use a different --out-dir or pass --force."
        )
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = [
        metric
        for metric, disabled in [
            ("bertscore_f1", args.no_bertscore),
            ("sentcos", args.no_sentcos),
            ("rougel_f1", args.no_rougel),
            ("jaccard", args.no_jaccard),
        ]
        if not disabled
    ]
    if not metrics:
        raise SystemExit("All metrics disabled — enable at least one.")

    rng = random.Random(args.seed)
    rows = load_rows(args.preds, args.min_chars, args.raw_field)
    if not rows:
        raise SystemExit("No explanation-required rows found in --preds.")

    print(f"[load] {len(rows)} explanation-required rows")
    print(f"[load] raw-output field: {args.raw_field!r}")

    rationale_provision = rationale_provision_table(rows)
    total_usable = sum(v["usable_rationales"] for v in rationale_provision.values())
    total_leading = sum(
        v["leading_label_token_count"] for v in rationale_provision.values()
    )
    print(f"[rationale provision] usable={total_usable}/{len(rows)}")
    print(f"[diagnostic] leading label token={total_leading}/{len(rows)}")

    true_list, base_list, stats = build_matched(rows, rng, args.min_chars)
    print(
        f"[matched] both_correct={stats['both_correct_pairs']}  "
        f"unusable_excluded={stats['excluded_unusable_rationale_pairs']}  "
        f"no_partner_excluded={stats['excluded_no_partner']}  "
        f"matched={stats['n_matched']}"
    )
    if stats["n_matched"] == 0:
        raise SystemExit("No matched pairs — inspect the printed validation counts.")

    bertscorer = (
        BERTScorer(batch_size=args.batch_size)
        if "bertscore_f1" in metrics
        else None
    )
    sentcos = SentCos() if "sentcos" in metrics else None
    rougel = RougeL() if "rougel_f1" in metrics else None
    jaccard_on = "jaccard" in metrics

    score_pairs(true_list, bertscorer, sentcos, rougel, jaccard_on)
    score_pairs(base_list, bertscorer, sentcos, rougel, jaccard_on)

    matched = []
    for true_rec, base_rec in zip(true_list, base_list):
        assert true_rec["text_a"] == base_rec["text_a"], "anchor mismatch"
        assert true_rec["anchor_id"] == base_rec["anchor_id"], "anchor ID mismatch"
        assert true_rec["partner_id"] != base_rec["partner_id"], (
            "baseline partner must be a different source message"
        )

        rec = {
            key: true_rec[key]
            for key in (
                "model",
                "template",
                "attack",
                "seed",
                "anchor_id",
                "pred_label",
            )
        }
        rec["true_partner_id"] = true_rec["partner_id"]
        rec["baseline_partner_id"] = base_rec["partner_id"]
        rec["true"] = {metric: true_rec.get(metric) for metric in metrics}
        rec["base"] = {metric: base_rec.get(metric) for metric in metrics}
        rec["gap"] = {
            metric: pairwise_gap(true_rec.get(metric), base_rec.get(metric))
            for metric in metrics
        }
        matched.append(rec)

    validation = {
        "n_matched_true": len(true_list),
        "n_matched_baseline": len(base_list),
        "n_pairwise_gaps": len(matched),
        "all_baseline_partners_different_message": all(
            str(r["anchor_id"]) != str(r["baseline_partner_id"]) for r in matched
        ),
        "ok": (
            len(true_list) == len(base_list) == len(matched)
            and all(
                str(r["anchor_id"]) != str(r["baseline_partner_id"])
                for r in matched
            )
        ),
    }
    assert validation["ok"], validation

    groupings = {
        "pooled": group_stats(matched, metrics, lambda r: "ALL"),
        "by_model": group_stats(matched, metrics, lambda r: r["model"]),
        "by_model_template": group_stats(
            matched, metrics, lambda r: (r["model"], r["template"])
        ),
        "by_model_attack": group_stats(
            matched, metrics, lambda r: (r["model"], r["attack"])
        ),
    }

    report = {
        "config": vars(args),
        "analysis_design": {
            "rationale_definition": (
                "all generated text remaining after FINAL-label removal; "
                "usable if at least min_chars non-whitespace characters"
            ),
            "true_pair": "clean_i versus perturbed_i",
            "baseline_pair": "clean_i versus perturbed_j",
            "baseline_partner_constraint": (
                "different source message; same model, template, predicted label, "
                "attack, and normalized seed"
            ),
            "baseline_candidate_pool": "perturbed rationales from eligible true pairs",
            "baseline_strategy": "one fixed-seed matched candidate per true pair",
            "baseline_random_seed": args.seed,
            "baseline_assignment_fixed": True,
            "sentence_seed_normalization": None,
        },
        "metrics": metrics,
        "matched_stats": stats,
        "validation": validation,
        "rationale_provision": rationale_provision,
        "groupings": groupings,
        "interpretation_note": (
            "A positive gap means that the clean rationale is more similar to its "
            "own perturbed rationale than to a matched perturbed rationale from a "
            "different message."
        ),
    }

    out_json = os.path.join(
        args.out_dir, "explanation_dim2_matched_rationale_anywhere.json"
    )
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    def _round(values):
        return {
            key: (
                round(value, 6)
                if isinstance(value, float) and math.isfinite(value)
                else None
                if isinstance(value, float) and not math.isfinite(value)
                else value
            )
            for key, value in values.items()
        }

    out_pairs = os.path.join(args.out_dir, "matched_pairs_rationale_anywhere.jsonl")
    with open(out_pairs, "w", encoding="utf-8") as handle:
        for rec in matched:
            slim = {
                key: rec[key]
                for key in (
                    "model",
                    "template",
                    "attack",
                    "seed",
                    "anchor_id",
                    "true_partner_id",
                    "baseline_partner_id",
                    "pred_label",
                )
            }
            slim["true"] = _round(rec["true"])
            slim["base"] = _round(rec["base"])
            slim["gap"] = _round(rec["gap"])
            handle.write(json.dumps(slim) + "\n")

    print("\n=== Rationale provision ===")
    for key, value in rationale_provision.items():
        print(
            f"  {key:45s} provision={value['rationale_provision_rate']:.4f} "
            f"({value['usable_rationales']}/{value['n_generations']}) "
            f"leading_label={value['leading_label_token_count']}"
        )

    print("\n=== Validation ===")
    print(
        f"  n_matched_true={validation['n_matched_true']}  "
        f"n_matched_baseline={validation['n_matched_baseline']}  "
        f"n_pairwise_gaps={validation['n_pairwise_gaps']}"
    )
    print(
        "  all baseline partners use a different source message: "
        f"{validation['all_baseline_partners_different_message']}"
    )
    print(f"  validation: {'PASS' if validation['ok'] else 'FAIL'}")

    print("\n=== Pooled pairwise gap: sim(A_i,A'_i) - sim(A_i,A'_j) ===")
    for metric in metrics:
        gap = groupings["pooled"]["ALL"][metric]["pairwise_gap"]
        true_mean = groupings["pooled"]["ALL"][metric]["matched_true_mean"]
        base_mean = groupings["pooled"]["ALL"][metric]["matched_baseline_mean"]
        print(
            f"  {metric:14s} gap mean={gap['mean']} median={gap['median']} "
            f"std={gap['std']} n={gap['n']} "
            f"(true={true_mean} baseline={base_mean})"
        )

    print(f"\n[write] {out_json}\n[write] {out_pairs}")


if __name__ == "__main__":
    main()
