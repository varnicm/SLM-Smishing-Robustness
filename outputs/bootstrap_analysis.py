"""
bootstrap_analysis.py
---------------------
Dimension 1 & 3 metrics with cluster-bootstrap confidence intervals.

This version fixes six real defects found in review. Each is documented at the point
where it is fixed, because each changes the numbers.

  FIX 1 — MATCHED dAcc.
      Clean accuracy was computed over ALL clean rows, but perturbed accuracy only over
      variants that survived the semantic-similarity gate. Those are different populations,
      so the difference conflated "the perturbation hurt" with "the surviving subset was
      easier/harder". We now compute BOTH accuracies over EXACTLY the same
      (message, template, attack, seed) rows: for every retained perturbed row we look up
      its own clean counterpart. dAcc = matched_clean_acc - perturbed_acc.

  FIX 2 — MESSAGE-LEVEL WEIGHTING for ASR / benign-flip.
      Denominators previously counted perturbation ROWS, so a message with three valid
      seeds contributed three times and a message with one contributed once. That is
      "success per valid variant", but the paper describes these rates over MESSAGES.
      We now aggregate seeds within a source message first (mean outcome across that
      message's valid variants), then average across messages — every message gets equal
      weight. The variant-weighted figure is still emitted, as *_variant, so both can be
      reported and they cannot be confused.

  FIX 3 — PAR: TWO DISTINCT QUANTITIES, both reported.
      par_majority : mean proportion of prompt configurations agreeing with the majority
                     label. With 4 configs this takes values 0.50 / 0.75 / 1.00, so
                     (1 - par_majority) is NOT the fraction of inputs showing disagreement.
      par_unanimous: fraction of inputs on which ALL configurations give the same label.
                     THIS is the quantity to use for "the model disagrees with itself on
                     x% of inputs".

  FIX 4 — PAR IS SPLIT BY CONDITION.
      Previously PAR pooled clean rows with every attack and seed, so it measured prompt
      agreement over the whole evaluation set, not clean-message prompt sensitivity, and
      it was influenced by how many valid variants each attack produced. We now report PAR
      for the clean condition and for each attack configuration separately.

  FIX 5 — STRATIFIED RESAMPLING.
      The corpus is balanced by design. Resampling all message ids together lets the
      class balance drift between replicates. We resample smishing ids and ham ids
      SEPARATELY, preserving the class counts in every replicate.

  FIX 6 — DATA INTEGRITY + UNPARSED ACCOUNTING.
      We now check for duplicate (message, model, template) clean rows, missing templates,
      and inconsistent true labels, and we report them instead of silently overwriting.
      We also report the unparseable-output rate. NOTE THE CONVENTION, which must be
      stated in the paper: an unparsed prediction is counted as INCORRECT, and as a FLIP
      if the clean prediction was parseable and differed; it is NOT counted as a
      successful evasion (ASR) or as a benign flip, because neither "ham" nor "smishing"
      was actually asserted.

CAVEAT ON PAIRED COMPARISONS (raised in review, and correct):
      Pairing cancels the message-level variance common to both models for UNCONDITIONAL
      metrics (accuracy, flip rate). For the CONDITIONAL metrics (ASR, benign-flip) the
      two models have DIFFERENT eligible subsets — each conditions on its own
      clean-correct predictions — so pairing reduces, but does not eliminate, the shared
      variance. We therefore describe pairwise comparisons as EXPLORATORY and do not
      apply a multiplicity correction; they are reported to indicate which gaps are large
      relative to sampling noise, not as confirmatory tests.

Usage:
    python bootstrap_analysis.py --preds outputs/preds_norm.jsonl \\
        --out-dir outputs/bootstrap --B 1000 --attack character
"""
import argparse, json, os, random, sys
from collections import defaultdict

ATTACKS = ["character", "word", "sentence", "multi"]
UNPARSED = "unparsed"


def norm_label(x):
    """'spam' deliberately NOT mapped to smishing: it can include non-phishing promo."""
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x                       # 'unparsed' and anything unexpected passes through


# ---------------------------------------------------------------------------
# LOAD + INTEGRITY CHECKS  (FIX 6)
# ---------------------------------------------------------------------------
def load(path):
    """-> by_msg[mid][model] = {"clean": {tmpl: pred},
                                "pert":  {(tmpl, attack, seed): pred},
                                "true":  label}
       plus a diagnostics dict."""
    by_msg = defaultdict(lambda: defaultdict(lambda: {"clean": {}, "pert": {}, "true": None}))
    diag = {"rows": 0, "dup_clean": 0, "dup_pert": 0, "label_conflict": 0,
            "unparsed_clean": 0, "unparsed_pert": 0}

    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        diag["rows"] += 1
        mid, model, tmpl = r["id"], r["model"], r["template"]
        pred = norm_label(r["pred"])
        true = norm_label(r["true_label"])
        slot = by_msg[mid][model]

        # true label must be consistent for a message
        if slot["true"] is None:
            slot["true"] = true
        elif slot["true"] != true:
            diag["label_conflict"] += 1

        if r["condition"] == "clean":
            if tmpl in slot["clean"]:                     # do NOT silently overwrite
                diag["dup_clean"] += 1
            slot["clean"][tmpl] = pred
            if pred == UNPARSED:
                diag["unparsed_clean"] += 1
        else:
            key = (tmpl, r["attack_class"], r["seed"])
            if key in slot["pert"]:
                diag["dup_pert"] += 1
            slot["pert"][key] = pred
            if pred == UNPARSED:
                diag["unparsed_pert"] += 1

    # missing templates
    all_t = sorted({t for mid in by_msg for m in by_msg[mid] for t in by_msg[mid][m]["clean"]})
    missing = 0
    for mid in by_msg:
        for m in by_msg[mid]:
            missing += len(all_t) - len(by_msg[mid][m]["clean"])
    diag["missing_clean_template_cells"] = missing
    diag["n_templates"] = len(all_t)
    return by_msg, diag


# ---------------------------------------------------------------------------
# METRICS over a list of message ids (ids may repeat: a bootstrap draw)
# ---------------------------------------------------------------------------
def metrics_for(by_msg, ids, model, attack):
    # --- matched accuracy accumulators (FIX 1) -----------------------------
    # every retained perturbed row contributes ONE clean and ONE perturbed observation,
    # from exactly the same (message, template, attack, seed) cell.
    mc_ok = mp_ok = m_n = 0

    fl = fl_n = 0                       # flip rate (all retained perturbed rows)

    # --- message-level ASR / benign-flip (FIX 2) ---------------------------
    asr_msg, ben_msg = [], []           # per-message mean outcome
    ev = ev_n = bf = bf_n = 0           # variant-weighted, reported alongside

    # --- PAR, split by condition (FIX 3 + FIX 4) ---------------------------
    par_maj = {"clean": [], **{a: [] for a in ATTACKS}}
    par_uni = {"clean": [], **{a: [] for a in ATTACKS}}

    unparsed = tot_pred = 0

    for mid in ids:
        slot = by_msg.get(mid, {}).get(model)
        if not slot:
            continue
        true = slot["true"]
        clean = slot["clean"]

        # ---- PAR on the CLEAN condition ----
        if len(clean) > 1:
            labels = list(clean.values())
            maj = max(set(labels), key=labels.count)
            par_maj["clean"].append(labels.count(maj) / len(labels))
            par_uni["clean"].append(1.0 if len(set(labels)) == 1 else 0.0)

        # ---- group perturbed rows by (attack, seed) for PAR ----
        groups = defaultdict(dict)                    # (attack,seed) -> {tmpl: pred}
        for (tmpl, a, seed), pred in slot["pert"].items():
            groups[(a, seed)][tmpl] = pred
        for (a, seed), tmap in groups.items():
            if a in par_maj and len(tmap) > 1:
                labels = list(tmap.values())
                maj = max(set(labels), key=labels.count)
                par_maj[a].append(labels.count(maj) / len(labels))
                par_uni[a].append(1.0 if len(set(labels)) == 1 else 0.0)

        # ---- perturbed metrics for the requested attack ----
        # per-message outcome lists, so seeds are averaged within a message (FIX 2)
        m_asr_hits, m_asr_n = 0, 0
        m_ben_hits, m_ben_n = 0, 0

        for (tmpl, a, seed), pred in slot["pert"].items():
            if a != attack:
                continue
            cp = clean.get(tmpl)
            if cp is None:                       # no matched clean cell -> cannot use
                continue

            tot_pred += 1
            if pred == UNPARSED:
                unparsed += 1

            # matched accuracy (FIX 1): SAME rows on both sides
            m_n += 1
            if cp == true:
                mc_ok += 1
            if pred == true:
                mp_ok += 1

            # flip rate: label changed clean -> perturbed (unparsed counts as a flip
            # only if the clean side WAS parseable and differs -- stated convention)
            fl_n += 1
            if pred != cp:
                fl += 1

            # ASR: conditional on clean-correct SMISHING. An unparsed perturbed output is
            # NOT an evasion (no 'ham' was asserted), but it stays in the denominator.
            if true == "smishing" and cp == "smishing":
                ev_n += 1
                m_asr_n += 1
                if pred == "ham":
                    ev += 1
                    m_asr_hits += 1

            # benign-flip: conditional on clean-correct HAM. Unparsed is NOT a benign flip.
            if true == "ham" and cp == "ham":
                bf_n += 1
                m_ben_n += 1
                if pred == "smishing":
                    bf += 1
                    m_ben_hits += 1

        if m_asr_n:
            asr_msg.append(m_asr_hits / m_asr_n)     # this MESSAGE's rate
        if m_ben_n:
            ben_msg.append(m_ben_hits / m_ben_n)

    nan = float("nan")
    mean = lambda v: sum(v) / len(v) if v else nan

    clean_matched = 100 * mc_ok / m_n if m_n else nan
    pert_matched = 100 * mp_ok / m_n if m_n else nan

    return {
        # FIX 1: both sides over identical rows
        "clean_acc_matched": clean_matched,
        "pert_acc":          pert_matched,
        "dacc":              clean_matched - pert_matched if m_n else nan,
        "flip":              100 * fl / fl_n if fl_n else nan,

        # FIX 2: message-weighted primary, variant-weighted secondary
        "asr":               100 * mean(asr_msg),
        "benign":            100 * mean(ben_msg),
        "asr_variant":       100 * ev / ev_n if ev_n else nan,
        "benign_variant":    100 * bf / bf_n if bf_n else nan,

        # FIX 3 + 4: PAR, both definitions, split by condition
        "par_majority_clean":  mean(par_maj["clean"]),
        "par_unanimous_clean": mean(par_uni["clean"]),
        "par_majority_attack":  mean(par_maj[attack]),
        "par_unanimous_attack": mean(par_uni[attack]),

        "unparsed_rate": 100 * unparsed / tot_pred if tot_pred else nan,

        # denominators
        "_n_matched":     m_n,
        "_n_asr_msgs":    len(asr_msg),
        "_n_benign_msgs": len(ben_msg),
        "_n_asr_variants":    ev_n,
        "_n_benign_variants": bf_n,
    }


METRICS = ["clean_acc_matched", "pert_acc", "dacc", "flip",
           "asr", "benign", "asr_variant", "benign_variant",
           "par_majority_clean", "par_unanimous_clean",
           "par_majority_attack", "par_unanimous_attack", "unparsed_rate"]


def pct(vals, p):
    v = sorted(x for x in vals if x == x)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-dir", default="outputs/bootstrap")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--attack", default="character", choices=ATTACKS)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    by_msg, diag = load(args.preds)

    # ---- FIX 5: stratify the resample by class ----
    smish_ids, ham_ids = [], []
    for mid, models in by_msg.items():
        t = next(iter(models.values()))["true"]
        (smish_ids if t == "smishing" else ham_ids).append(mid)
    smish_ids.sort(); ham_ids.sort()
    all_ids = smish_ids + ham_ids
    models = sorted({m for mid in by_msg for m in by_msg[mid]})

    print(f"[data] rows={diag['rows']} messages={len(all_ids)} "
          f"(smishing={len(smish_ids)}, ham={len(ham_ids)}) models={len(models)}",
          file=sys.stderr)
    print(f"[integrity] dup_clean={diag['dup_clean']} dup_pert={diag['dup_pert']} "
          f"label_conflicts={diag['label_conflict']} "
          f"missing_clean_cells={diag['missing_clean_template_cells']}", file=sys.stderr)
    print(f"[integrity] unparsed: clean={diag['unparsed_clean']} "
          f"perturbed={diag['unparsed_pert']}", file=sys.stderr)

    point = {m: metrics_for(by_msg, all_ids, m, args.attack) for m in models}

    reps = {m: defaultdict(list) for m in models}
    ns, nh = len(smish_ids), len(ham_ids)
    for b in range(args.B):
        # stratified: preserve class counts in every replicate (FIX 5)
        draw = ([smish_ids[rng.randrange(ns)] for _ in range(ns)] +
                [ham_ids[rng.randrange(nh)] for _ in range(nh)])
        for m in models:
            r = metrics_for(by_msg, draw, m, args.attack)
            for k in METRICS:
                reps[m][k].append(r[k])
        if (b + 1) % 50 == 0:
            print(f"  [boot] {b+1}/{args.B}", file=sys.stderr)

    results = {}
    for m in models:
        results[m] = {k: {"point": point[m][k],
                          "ci_lo": pct(reps[m][k], 0.025),
                          "ci_hi": pct(reps[m][k], 0.975)} for k in METRICS}
        results[m]["denominators"] = {k: point[m][k] for k in point[m] if k.startswith("_n_")}

    # EXPLORATORY paired differences (see caveat in the header docstring)
    pairwise = {}
    for k in ["clean_acc_matched", "dacc", "flip", "asr", "benign"]:
        pairwise[k] = {}
        for i, a in enumerate(models):
            for b2 in models[i+1:]:
                d = [x - y for x, y in zip(reps[a][k], reps[b2][k]) if x == x and y == y]
                lo, hi = pct(d, 0.025), pct(d, 0.975)
                pairwise[k][f"{a} vs {b2}"] = {
                    "diff": point[a][k] - point[b2][k],
                    "ci_lo": lo, "ci_hi": hi,
                    "excludes_zero": bool(lo > 0 or hi < 0),
                }

    with open(os.path.join(args.out_dir, "bootstrap_results.json"), "w") as f:
        json.dump({"attack": args.attack, "B": args.B,
                   "n_messages": len(all_ids), "n_smishing": ns, "n_ham": nh,
                   "integrity": diag, "per_model": results, "pairwise": pairwise},
                  f, indent=2)

    S = lambda m: m.split("/")[-1]
    L = []
    L.append("=" * 100)
    L.append(f"CLUSTER BOOTSTRAP — resampled by ORIGINAL MESSAGE, STRATIFIED by class")
    L.append(f"B={args.B}   messages={len(all_ids)} (smishing={ns}, ham={nh})   "
             f"attack={args.attack}")
    L.append("=" * 100)
    L.append("Variants of one message are correlated, so the resampling unit is the MESSAGE,")
    L.append("not the row. Intervals are 95% percentile CIs over the replicates.")
    L.append("")
    L.append("clean_acc / pert_acc / dAcc : MATCHED — both sides over the SAME retained rows.")
    L.append("ASR / benign-flip           : MESSAGE-WEIGHTED (seeds averaged within a message).")
    L.append("                              variant-weighted values also shown, for comparison.")
    L.append("PAR(majority)               : mean share of prompt configs agreeing with the majority.")
    L.append("PAR(unanimous)              : share of inputs where ALL configs agree.")
    L.append("                              Use PAR(unanimous) for 'disagrees with itself on x%'.")
    L.append("unparsed                    : counted as INCORRECT and as a FLIP, but NOT as an")
    L.append("                              evasion or a benign flip (no label was asserted).")
    L.append("")
    L.append(f"[integrity] dup_clean={diag['dup_clean']}  dup_pert={diag['dup_pert']}  "
             f"label_conflicts={diag['label_conflict']}  "
             f"missing_clean_cells={diag['missing_clean_template_cells']}")
    L.append("")

    def col(m, k, f="{:.1f}"):
        d = results[m][k]
        return f"{f.format(d['point'])} [{f.format(d['ci_lo'])},{f.format(d['ci_hi'])}]"

    L.append(f"{'model':16s} {'clean_acc':>22s} {'dAcc':>22s} {'flip':>22s}")
    L.append("-" * 100)
    for m in models:
        L.append(f"{S(m):16s} {col(m,'clean_acc_matched'):>22s} {col(m,'dacc'):>22s} "
                 f"{col(m,'flip'):>22s}")
    L.append("")
    L.append(f"{'model':16s} {'ASR (msg)':>22s} {'ASR (variant)':>22s} "
             f"{'benign (msg)':>22s} {'benign (variant)':>22s}")
    L.append("-" * 100)
    for m in models:
        L.append(f"{S(m):16s} {col(m,'asr'):>22s} {col(m,'asr_variant'):>22s} "
                 f"{col(m,'benign'):>22s} {col(m,'benign_variant'):>22s}")
    L.append("")
    L.append(f"{'model':16s} {'PARmaj clean':>22s} {'PARunan clean':>22s} "
             f"{'PARmaj '+args.attack:>22s} {'unparsed %':>18s}")
    L.append("-" * 100)
    for m in models:
        L.append(f"{S(m):16s} {col(m,'par_majority_clean','{:.3f}'):>22s} "
                 f"{col(m,'par_unanimous_clean','{:.3f}'):>22s} "
                 f"{col(m,'par_majority_attack','{:.3f}'):>22s} "
                 f"{col(m,'unparsed_rate','{:.2f}'):>18s}")
    L.append("")
    L.append("Estimates rest on:")
    for m in models:
        d = results[m]["denominators"]
        L.append(f"  {S(m):16s} matched rows={d['_n_matched']:6d}  "
                 f"ASR messages={d['_n_asr_msgs']:5d} (variants={d['_n_asr_variants']:6d})  "
                 f"benign messages={d['_n_benign_msgs']:5d} (variants={d['_n_benign_variants']:6d})")

    open(os.path.join(args.out_dir, "bootstrap_summary.txt"), "w").write("\n".join(L))
    print("\n".join(L))

    for k in ["benign", "asr", "clean_acc_matched", "dacc"]:
        out = [f"EXPLORATORY PAIRED DIFFERENCES — {k}   (attack={args.attack}, B={args.B})",
               "",
               "Same resampled messages for both models in each replicate. For UNCONDITIONAL",
               "metrics this cancels the shared message-level variance. For the CONDITIONAL",
               "metrics (ASR, benign-flip) the two models condition on DIFFERENT clean-correct",
               "subsets, so pairing REDUCES but does not eliminate shared variance.",
               "No multiplicity correction is applied: these comparisons are EXPLORATORY,",
               "reported to show which gaps are large relative to sampling noise.",
               "'*' = 95% CI of the difference excludes zero.", ""]
        for pair, d in sorted(pairwise[k].items(), key=lambda x: -abs(x[1]["diff"])):
            mark = "*" if d["excludes_zero"] else " "
            p = pair.replace("/", "_")
            out.append(f" {mark} {p:62s} diff={d['diff']:+7.2f}  "
                       f"95% CI [{d['ci_lo']:+7.2f}, {d['ci_hi']:+7.2f}]")
        open(os.path.join(args.out_dir, f"pairwise_{k}.txt"), "w").write("\n".join(out))

    print(f"\n[write] {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
