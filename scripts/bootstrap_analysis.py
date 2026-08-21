"""
bootstrap_analysis.py
---------------------
Dimension 1 & 3 metrics with cluster-bootstrap confidence intervals.

This version incorporates every correction raised in review. Each is documented where it
is applied, because each changes the numbers.

  1. MATCHED dAcc.  Clean accuracy was previously taken over ALL clean rows while
     perturbed accuracy was taken only over gate-surviving variants — different
     populations, so the difference conflated "perturbation hurt" with "the surviving
     subset was easier". dAcc is now computed with BOTH sides over exactly the same
     (message, template, attack, seed) rows.

     NOTE: the clean side of that matched pair is NOT overall clean accuracy. It is
     restricted to the attack-specific retained subset and varies by attack. Section 5.1
     must use `clean_acc_overall`, which is computed over ALL clean rows and is
     attack-independent. Both are reported, and they are never conflated.

  2. MESSAGE-WEIGHTED ASR / benign-flip.  Denominators previously counted perturbation
     ROWS, so a message with three valid seeds counted three times. That estimand is
     "success per valid variant", but the paper describes these rates over MESSAGES.
     Seeds are now averaged WITHIN a message first, then across messages, so every
     message carries equal weight. The variant-weighted figures are still emitted
     (`*_variant`) so both can be reported and cannot be confused.

  3. PAR — TWO DISTINCT QUANTITIES.
       par_majority  : mean share of prompt configurations agreeing with the majority
                       label. With 4 configurations this takes values 0.50/0.75/1.00, so
                       (1 - par_majority) is NOT the fraction of inputs showing any
                       disagreement.
       par_unanimous : fraction of inputs on which ALL configurations agree. THIS is the
                       quantity for "the model disagrees with itself on x% of inputs".

  4. PAR SPLIT BY CONDITION.  Previously PAR pooled clean rows with every attack and
     seed, so it measured agreement over the whole evaluation set and was influenced by
     how many valid variants each attack happened to produce. Now reported separately for
     the clean condition and for each attack.

  5. STRATIFIED RESAMPLING.  The corpus is balanced by design, so smishing and ham ids
     are resampled SEPARATELY, preserving class counts in every replicate. Any label that
     is neither smishing nor ham raises, rather than being silently swept into ham.

  6. INTEGRITY IS FATAL.  Duplicate rows, missing template cells, or a message carrying
     two different true labels (checked GLOBALLY, across models) abort the run. Continuing
     would produce numbers that depend on which duplicate happened to be read last.

  7. UNPARSED OUTPUTS — explicit convention, and two separate metrics.
       label_flip     : both clean and perturbed outputs parseable AND the labels differ.
                        This is the strict quantity.
       output_change  : the outputs differ at all, INCLUDING changes to/from unparsed.
     An unparsed perturbed output counts as INCORRECT for accuracy, but is NEITHER an
     evasion NOR a benign flip — no label was asserted — while remaining in those
     denominators. The unparsed rate is reported separately.

CAVEAT ON PAIRED COMPARISONS.  Pairing (same resampled messages for both models) cancels
the shared message-level variance for UNCONDITIONAL metrics. For the CONDITIONAL metrics
(ASR, benign-flip) the two models condition on DIFFERENT clean-correct subsets, so pairing
REDUCES but does not eliminate shared variance. Pairwise comparisons are therefore
reported as EXPLORATORY, with no multiplicity correction: they indicate which gaps are
large relative to sampling noise, not confirmatory tests.

Usage:
    python bootstrap_analysis.py --preds outputs/preds_norm.jsonl \
        --out-dir outputs/bootstrap_character --B 1000 --attack character
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
    return x                        # 'unparsed' and anything unexpected passes through


# ---------------------------------------------------------------------------
# LOAD + INTEGRITY (fatal)
# ---------------------------------------------------------------------------
def load(path):
    by_msg = defaultdict(lambda: defaultdict(lambda: {"clean": {}, "pert": {}}))
    true_label = {}                 # ONE global true label per message, across models
    diag = {"rows": 0, "dup_clean": 0, "dup_pert": 0, "label_conflict": 0,
            "unparsed_clean": 0, "unparsed_pert": 0}
    conflicts = []

    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        diag["rows"] += 1
        mid, model, tmpl = r["id"], r["model"], r["template"]
        pred = norm_label(r["pred"])
        true = norm_label(r["true_label"])

        if mid not in true_label:
            if true not in ("smishing", "ham"):
                raise ValueError(f"Unexpected true label for {mid}: {true!r}. "
                                 "Expected 'smishing' or 'ham'.")
            true_label[mid] = true
        elif true_label[mid] != true:
            diag["label_conflict"] += 1
            if len(conflicts) < 10:
                conflicts.append(f"{mid}: {true_label[mid]!r} vs {true!r} (model={model})")

        slot = by_msg[mid][model]
        if r["condition"] == "clean":
            if tmpl in slot["clean"]:
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

    all_t = sorted({t for mid in by_msg for m in by_msg[mid] for t in by_msg[mid][m]["clean"]})
    missing = sum(len(all_t) - len(by_msg[mid][m]["clean"])
                  for mid in by_msg for m in by_msg[mid])
    diag["missing_clean_template_cells"] = missing
    diag["n_templates"] = len(all_t)

    bad = {k: diag[k] for k in ("dup_clean", "dup_pert", "label_conflict",
                                "missing_clean_template_cells") if diag[k]}
    if bad:
        msg = ["INTEGRITY CHECK FAILED — analysis aborted.", ""]
        msg += [f"  {k} = {v}" for k, v in bad.items()]
        if conflicts:
            msg += ["", "  label conflicts (first 10):"] + [f"    {c}" for c in conflicts]
        msg += ["", "All of these must be zero. A duplicate means one row was silently",
                "discarded; a missing cell means a metric uses a different denominator for",
                "that model; a label conflict means the ground truth is inconsistent.",
                "Fix the prediction files before analysing them."]
        raise SystemExit("\n".join(msg))

    return by_msg, true_label, diag


# ---------------------------------------------------------------------------
# METRICS over a list of message ids (ids may repeat: that is a bootstrap draw)
# ---------------------------------------------------------------------------
def metrics_for(by_msg, true_label, ids, model, attack):
    ov_ok = ov_n = 0                          # overall clean accuracy (ALL clean rows)
    ov_t = defaultdict(lambda: [0, 0])        # ... per template
    mc_ok = mp_ok = m_n = 0                   # matched clean / perturbed
    oc = oc_n = lf = lf_n = 0                 # output-change / strict label-flip
    asr_msg, ben_msg = [], []                 # message-weighted conditionals
    ev = ev_n = bf = bf_n = 0                 # variant-weighted conditionals
    par_maj = {"clean": [], **{a: [] for a in ATTACKS}}
    par_uni = {"clean": [], **{a: [] for a in ATTACKS}}
    unparsed = tot = 0

    for mid in ids:
        slot = by_msg.get(mid, {}).get(model)
        if not slot:
            continue
        true = true_label[mid]
        clean = slot["clean"]

        for tmpl, pred in clean.items():
            ov_n += 1
            ov_t[tmpl][1] += 1
            if pred == true:
                ov_ok += 1
                ov_t[tmpl][0] += 1

        if len(clean) > 1:
            labels = list(clean.values())
            maj = max(set(labels), key=labels.count)
            par_maj["clean"].append(labels.count(maj) / len(labels))
            par_uni["clean"].append(1.0 if len(set(labels)) == 1 else 0.0)

        groups = defaultdict(dict)
        for (tmpl, a, seed), pred in slot["pert"].items():
            groups[(a, seed)][tmpl] = pred
        for (a, seed), tmap in groups.items():
            if a in par_maj and len(tmap) > 1:
                labels = list(tmap.values())
                maj = max(set(labels), key=labels.count)
                par_maj[a].append(labels.count(maj) / len(labels))
                par_uni[a].append(1.0 if len(set(labels)) == 1 else 0.0)

        m_ah = m_an = m_bh = m_bn = 0
        for (tmpl, a, seed), pred in slot["pert"].items():
            if a != attack:
                continue
            cp = clean.get(tmpl)
            if cp is None:
                continue
            tot += 1
            if pred == UNPARSED:
                unparsed += 1

            m_n += 1
            if cp == true:
                mc_ok += 1
            if pred == true:
                mp_ok += 1

            oc_n += 1
            if pred != cp:
                oc += 1
            if cp != UNPARSED and pred != UNPARSED:
                lf_n += 1
                if pred != cp:
                    lf += 1

            if true == "smishing" and cp == "smishing":
                ev_n += 1; m_an += 1
                if pred == "ham":
                    ev += 1; m_ah += 1
            if true == "ham" and cp == "ham":
                bf_n += 1; m_bn += 1
                if pred == "smishing":
                    bf += 1; m_bh += 1

        if m_an:
            asr_msg.append(m_ah / m_an)
        if m_bn:
            ben_msg.append(m_bh / m_bn)

    nan = float("nan")
    mean = lambda v: sum(v) / len(v) if v else nan
    cm = 100 * mc_ok / m_n if m_n else nan
    pm = 100 * mp_ok / m_n if m_n else nan

    out = {
        "clean_acc_overall":  100 * ov_ok / ov_n if ov_n else nan,   # Section 5.1
        "clean_acc_matched":  cm,                                     # for dAcc ONLY
        "pert_acc_matched":   pm,
        "dacc":               cm - pm if m_n else nan,
        "output_change":      100 * oc / oc_n if oc_n else nan,
        "label_flip":         100 * lf / lf_n if lf_n else nan,
        "asr":                100 * mean(asr_msg),
        "benign":             100 * mean(ben_msg),
        "asr_variant":        100 * ev / ev_n if ev_n else nan,
        "benign_variant":     100 * bf / bf_n if bf_n else nan,
        "par_majority_clean":   mean(par_maj["clean"]),
        "par_unanimous_clean":  mean(par_uni["clean"]),
        "par_majority_attack":  mean(par_maj[attack]),
        "par_unanimous_attack": mean(par_uni[attack]),
        "unparsed_rate":      100 * unparsed / tot if tot else nan,
        "_n_clean_rows":  ov_n, "_n_matched": m_n, "_n_labelflip": lf_n,
        "_n_asr_msgs": len(asr_msg), "_n_benign_msgs": len(ben_msg),
        "_n_asr_variants": ev_n, "_n_benign_variants": bf_n,
    }
    for t, (ok, n) in ov_t.items():
        out[f"clean_acc__{t}"] = 100 * ok / n if n else nan
    return out


METRICS = ["clean_acc_overall", "clean_acc_matched", "pert_acc_matched", "dacc",
           "output_change", "label_flip", "asr", "benign",
           "asr_variant", "benign_variant",
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
    ap.add_argument("--out-dir", required=True,
                    help="a FRESH folder per run; refuses to overwrite")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--attack", default="character", choices=ATTACKS)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"--out-dir {args.out_dir} exists and is not empty. "
                         "Use a fresh folder so runs cannot be confused.")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    by_msg, true_label, diag = load(args.preds)      # aborts on integrity failure

    smish_ids, ham_ids = [], []
    for mid, t in true_label.items():
        if t == "smishing":
            smish_ids.append(mid)
        elif t == "ham":
            ham_ids.append(mid)
        else:
            raise ValueError(f"Unexpected true label for {mid}: {t!r}")
    smish_ids.sort(); ham_ids.sort()
    all_ids = smish_ids + ham_ids
    models = sorted({m for mid in by_msg for m in by_msg[mid]})
    ns, nh = len(smish_ids), len(ham_ids)

    print(f"[data] rows={diag['rows']}  messages={len(all_ids)} "
          f"(smishing={ns}, ham={nh})  models={len(models)}  "
          f"templates={diag['n_templates']}", file=sys.stderr)
    print("[integrity] PASSED (dup_clean=0 dup_pert=0 label_conflicts=0 "
          "missing_clean_cells=0)", file=sys.stderr)
    print(f"[unparsed] clean={diag['unparsed_clean']} perturbed={diag['unparsed_pert']}",
          file=sys.stderr)

    point = {m: metrics_for(by_msg, true_label, all_ids, m, args.attack) for m in models}
    tkeys = sorted(k for k in point[models[0]] if k.startswith("clean_acc__"))
    keys = METRICS + tkeys

    reps = {m: defaultdict(list) for m in models}
    for b in range(args.B):
        draw = ([smish_ids[rng.randrange(ns)] for _ in range(ns)] +
                [ham_ids[rng.randrange(nh)] for _ in range(nh)])
        for m in models:
            r = metrics_for(by_msg, true_label, draw, m, args.attack)
            for k in keys:
                reps[m][k].append(r[k])
        if (b + 1) % 50 == 0:
            print(f"  [boot] {b+1}/{args.B}", file=sys.stderr)

    results = {}
    for m in models:
        results[m] = {k: {"point": point[m][k],
                          "ci_lo": pct(reps[m][k], 0.025),
                          "ci_hi": pct(reps[m][k], 0.975)} for k in keys}
        results[m]["denominators"] = {k: point[m][k] for k in point[m] if k.startswith("_n_")}

    pairwise = {}
    for k in ["clean_acc_overall", "dacc", "label_flip", "asr", "benign"]:
        pairwise[k] = {}
        for i, a in enumerate(models):
            for b2 in models[i+1:]:
                d = [x - y for x, y in zip(reps[a][k], reps[b2][k]) if x == x and y == y]
                lo, hi = pct(d, 0.025), pct(d, 0.975)
                pairwise[k][f"{a} vs {b2}"] = {"diff": point[a][k] - point[b2][k],
                                               "ci_lo": lo, "ci_hi": hi,
                                               "excludes_zero": bool(lo > 0 or hi < 0)}

    with open(os.path.join(args.out_dir, "bootstrap_results.json"), "w") as f:
        json.dump({"attack": args.attack, "B": args.B, "n_messages": len(all_ids),
                   "n_smishing": ns, "n_ham": nh, "integrity": diag,
                   "per_model": results, "pairwise": pairwise}, f, indent=2)

    S = lambda m: m.split("/")[-1]
    def col(m, k, f="{:.1f}"):
        d = results[m][k]
        return f"{f.format(d['point'])} [{f.format(d['ci_lo'])},{f.format(d['ci_hi'])}]"

    L = ["=" * 104,
         "CLUSTER BOOTSTRAP — resampled by ORIGINAL MESSAGE, STRATIFIED by class",
         f"B={args.B}  messages={len(all_ids)} (smishing={ns}, ham={nh})  attack={args.attack}",
         "=" * 104, "",
         "DEFINITIONS — state these in the paper exactly as written:",
         "  clean_acc (overall) : accuracy over ALL clean rows. Attack-independent.",
         "                        This is Section 5.1's clean accuracy.",
         "  dAcc                : clean - perturbed, both over the SAME retained rows. The",
         "                        clean side of that pair is NOT the overall clean accuracy",
         "                        and must not be reported as such.",
         "  label_flip          : both outputs parseable, labels differ (strict).",
         "  output_change       : outputs differ at all, INCLUDING to/from unparsed.",
         "  ASR / benign-flip   : MESSAGE-weighted (seeds averaged within a message, then",
         "                        across messages). Variant-weighted shown alongside.",
         "                        An unparsed perturbed output is NEITHER an evasion NOR a",
         "                        benign flip (no label asserted), but stays in the",
         "                        denominator and counts as incorrect for accuracy.",
         "  PAR(majority)       : mean share of prompt configs agreeing with the majority.",
         "  PAR(unanimous)      : share of inputs where ALL configs agree. Use THIS for",
         "                        'disagrees with itself on x% of inputs'.",
         "",
         "[integrity] PASSED — dup_clean=0, dup_pert=0, label_conflicts=0, missing_cells=0",
         f"[unparsed]  clean rows={diag['unparsed_clean']}, "
         f"perturbed rows={diag['unparsed_pert']}", ""]

    L += ["-" * 104, "(1) CLEAN ACCURACY — all clean rows (Section 5.1)", "-" * 104,
          f"{'model':16s} {'overall':>21s} " +
          "".join(f"{k.replace('clean_acc__',''):>21s}" for k in tkeys)]
    for m in models:
        L.append(f"{S(m):16s} {col(m,'clean_acc_overall'):>21s} " +
                 "".join(f"{col(m,k):>21s}" for k in tkeys))

    L += ["", "-" * 104, f"(2) ROBUSTNESS under {args.attack}-level perturbation", "-" * 104,
          f"{'model':16s} {'dAcc (matched)':>21s} {'label_flip':>21s} "
          f"{'output_change':>21s} {'unparsed %':>16s}"]
    for m in models:
        L.append(f"{S(m):16s} {col(m,'dacc'):>21s} {col(m,'label_flip'):>21s} "
                 f"{col(m,'output_change'):>21s} {col(m,'unparsed_rate','{:.2f}'):>16s}")

    L += ["", "-" * 104,
          "(3) CONDITIONAL METRICS — message-weighted, with variant-weighted alongside",
          "-" * 104,
          f"{'model':16s} {'ASR (msg)':>21s} {'ASR (variant)':>21s} "
          f"{'benign (msg)':>21s} {'benign (variant)':>21s}"]
    for m in models:
        L.append(f"{S(m):16s} {col(m,'asr'):>21s} {col(m,'asr_variant'):>21s} "
                 f"{col(m,'benign'):>21s} {col(m,'benign_variant'):>21s}")

    L += ["", "-" * 104, "(4) PROMPT SENSITIVITY — clean, and under this attack", "-" * 104,
          f"{'model':16s} {'PARmaj clean':>21s} {'PARunan clean':>21s} "
          f"{'PARmaj '+args.attack:>21s} {'PARunan '+args.attack:>21s}"]
    for m in models:
        L.append(f"{S(m):16s} {col(m,'par_majority_clean','{:.3f}'):>21s} "
                 f"{col(m,'par_unanimous_clean','{:.3f}'):>21s} "
                 f"{col(m,'par_majority_attack','{:.3f}'):>21s} "
                 f"{col(m,'par_unanimous_attack','{:.3f}'):>21s}")

    L += ["", "Estimates rest on:"]
    for m in models:
        d = results[m]["denominators"]
        L.append(f"  {S(m):16s} clean rows={d['_n_clean_rows']:6d}  "
                 f"matched={d['_n_matched']:6d}  "
                 f"ASR msgs={d['_n_asr_msgs']:5d} (var={d['_n_asr_variants']:6d})  "
                 f"benign msgs={d['_n_benign_msgs']:5d} (var={d['_n_benign_variants']:6d})")

    open(os.path.join(args.out_dir, "bootstrap_summary.txt"), "w").write("\n".join(L))
    print("\n".join(L))

    for k in ["benign", "asr", "clean_acc_overall", "dacc", "label_flip"]:
        out = [f"EXPLORATORY PAIRED DIFFERENCES — {k}  (attack={args.attack}, B={args.B})", "",
               "Same resampled messages for both models in each replicate. For UNCONDITIONAL",
               "metrics this cancels the shared message-level variance. For CONDITIONAL",
               "metrics (ASR, benign-flip) the models condition on DIFFERENT clean-correct",
               "subsets, so pairing REDUCES but does not eliminate shared variance.",
               "No multiplicity correction: EXPLORATORY, not confirmatory.",
               "'*' = 95% CI of the difference excludes zero.", ""]
        for pair, d in sorted(pairwise[k].items(), key=lambda x: -abs(x[1]["diff"])):
            out.append(f" {'*' if d['excludes_zero'] else ' '} "
                       f"{pair.replace('/', '_'):60s} diff={d['diff']:+7.2f}  "
                       f"95% CI [{d['ci_lo']:+7.2f}, {d['ci_hi']:+7.2f}]")
        open(os.path.join(args.out_dir, f"pairwise_{k}.txt"), "w").write("\n".join(out))

    print(f"\n[write] {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
