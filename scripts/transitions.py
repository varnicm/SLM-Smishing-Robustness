"""
transitions.py
--------------
Decompose the accuracy change under perturbation into its four constituent transitions,
so that WHERE an accuracy gain or loss comes from is visible rather than inferred.

WHY THIS EXISTS
    Under sentence-level paraphrasing, matched accuracy INCREASES for four models
    (positive transition net, equivalently negative dAcc in bootstrap_analysis.py where
    dAcc = clean - perturbed; intervals excluding zero). The tempting explanation is that
    PEGASUS normalises informal SMS into regular English, removing the surface noise the
    models were over-reacting to, and thereby correcting false positives.

    That explanation is NOT established by dAcc, and it is not established by the lower
    benign-flip rate either: benign-flip only counts NEW errors among ham messages that
    were ALREADY correct. It cannot see whether paraphrasing FIXED ham messages that were
    already being misclassified. The two quantities are about different populations.

    The four transitions below settle it. Each matched message-prompt-variant row is placed
    into exactly one cell, defined by the clean prediction and the perturbed prediction:

      For HAM messages (true label = ham):
        FP_corrected   clean=smishing -> pert=ham        a false positive was FIXED
        FP_new         clean=ham      -> pert=smishing    a false positive was CREATED
                                                          (this is the benign-flip)
        (ham->ham and smishing->smishing are unchanged)

      For SMISHING messages (true label = smishing):
        FN_corrected   clean=ham      -> pert=smishing    a miss was FIXED
        FN_new         clean=smishing -> pert=ham         the attack EVADED (this is ASR)

    Accuracy change (transition net) = (FP_corrected + FN_corrected) - (FP_new + FN_new),
    over the matched rows. SIGN: transition net POSITIVE = improvement. This is the exact
    negative of dAcc in bootstrap_analysis.py, where dAcc = clean - perturbed and POSITIVE
    = degradation. On the both-parseable matched set, under identical weighting,
    net = -dAcc exactly; on the full matched set the two differ by the unparsed-involved
    rows, which net routes to lost_to_unparsed/gained_from_unparsed rather than the four
    cells. If the gain under paraphrasing is dominated by FP_corrected, the
    surface-normalisation reading is supported. If it is not, that reading must be dropped.

    RECONCILIATION NOTE. The paper's PRIMARY dAcc (bootstrap_analysis.py) is
    VARIANT-weighted: computed over matched message-prompt-variant rows. The HEADLINE net
    printed here is MESSAGE-weighted (comparable with ASR / benign-flip). Because the two
    use different weightings, net(message) is NOT the exact algebraic inverse of the
    variant-weighted dAcc. For a numerically exact check, this script ALSO prints a
    ROW-weighted net (net_gain_pct, including the unparsed-involved terms); that quantity
    equals -dAcc_variant exactly on the same rows. Use the message-weighted net for the
    narrative, the row-weighted net for the identity check.

    Bootstrap CIs are cluster-resampled by original message, stratified by class, exactly
    as in bootstrap_analysis.py — the transitions inherit the same dependence structure.

Usage:
    python transitions.py --preds outputs/preds_dedup.jsonl \\
        --out-dir outputs/transitions_sentence --attack sentence --B 1000
"""
import argparse, json, os, random, sys
from collections import defaultdict

ATTACKS = ["character", "word", "sentence", "multi"]
UNPARSED = "unparsed"


def norm_label(x):
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x


def load(path):
    by_msg = defaultdict(lambda: defaultdict(lambda: {"clean": {}, "pert": {}}))
    true_label = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        mid, model, tmpl = r["id"], r["model"], r["template"]
        pred, true = norm_label(r["pred"]), norm_label(r["true_label"])
        if mid not in true_label:
            if true not in ("smishing", "ham"):
                raise ValueError(f"unexpected true label for {mid}: {true!r}")
            true_label[mid] = true
        slot = by_msg[mid][model]
        if r["condition"] == "clean":
            slot["clean"][tmpl] = pred
        else:
            slot["pert"][(tmpl, r["attack_class"], r["seed"])] = pred
    return by_msg, true_label


def transitions_for(by_msg, true_label, ids, model, attack):
    """Returns (row_counts, message_weighted_rates).

    TWO WEIGHTINGS, because they answer different questions and must not be mixed:

      row_counts  : every matched message-prompt-variant row counted once. A message with
                    more surviving variants contributes more. These are the raw counts.
                    net_gain_pct(row_counts) is the ROW-weighted net, which equals
                    -dAcc_variant exactly on the same matched rows.

      msg_rates   : the outcome is first averaged WITHIN a source message (over its
                    prompts and surviving variants), then averaged ACROSS messages, so
                    each message carries equal weight. This is the SAME weighting used
                    for ASR and the conditional benign-flip rate, so the transition rates
                    are directly comparable with them. Report these for the narrative.
    """
    c = defaultdict(int)
    per_msg = defaultdict(list)     # transition -> [per-message rate]

    for mid in ids:
        slot = by_msg.get(mid, {}).get(model)
        if not slot:
            continue
        true = true_label[mid]
        m_hits = defaultdict(int)
        m_n = 0

        for (tmpl, a, seed), pert in slot["pert"].items():
            if a != attack:
                continue
            clean = slot["clean"].get(tmpl)
            if clean is None:
                continue
            c["matched"] += 1
            m_n += 1

            if clean == UNPARSED or pert == UNPARSED:
                c["unparsed_involved"] += 1
                if clean == true and pert != true:
                    c["lost_to_unparsed"] += 1
                elif clean != true and pert == true:
                    c["gained_from_unparsed"] += 1
                continue

            if true == "ham":
                if clean == "smishing" and pert == "ham":
                    c["FP_corrected"] += 1; m_hits["FP_corrected"] += 1
                elif clean == "ham" and pert == "smishing":
                    c["FP_new"] += 1;       m_hits["FP_new"] += 1
                elif clean == "ham":
                    c["ham_stable_correct"] += 1
                else:
                    c["ham_stable_wrong"] += 1
            else:
                if clean == "ham" and pert == "smishing":
                    c["FN_corrected"] += 1; m_hits["FN_corrected"] += 1
                elif clean == "smishing" and pert == "ham":
                    c["FN_new"] += 1;       m_hits["FN_new"] += 1
                elif clean == "smishing":
                    c["smish_stable_correct"] += 1
                else:
                    c["smish_stable_wrong"] += 1

        if m_n:
            for k in ("FP_corrected", "FP_new", "FN_corrected", "FN_new"):
                per_msg[k].append(m_hits[k] / m_n)
            gained = m_hits["FP_corrected"] + m_hits["FN_corrected"]
            lost = m_hits["FP_new"] + m_hits["FN_new"]
            per_msg["net"].append((gained - lost) / m_n)

    mean = lambda v: sum(v) / len(v) if v else float("nan")
    msg_rates = {k: 100 * mean(v) for k, v in per_msg.items()}
    msg_rates["_n_messages"] = len(per_msg.get("net", []))
    return c, msg_rates


def net_gain_pct(c):
    """ROW-weighted accuracy change, as a % of matched message-prompt-variant rows.
    POSITIVE = perturbation improved accuracy. Includes the unparsed-involved terms, so
    over the FULL matched set this equals -dAcc_variant exactly (dAcc = clean - perturbed,
    variant/row-weighted, in bootstrap_analysis.py). The HEADLINE reported figure is the
    MESSAGE-weighted net; this row-weighted net is printed alongside for the exact
    algebraic identity check net_row = -dAcc_variant."""
    if not c["matched"]:
        return float("nan")
    gained = c["FP_corrected"] + c["FN_corrected"] + c["gained_from_unparsed"]
    lost = c["FP_new"] + c["FN_new"] + c["lost_to_unparsed"]
    return 100 * (gained - lost) / c["matched"]


def pct_of(c, key):
    return 100 * c[key] / c["matched"] if c["matched"] else float("nan")


def pctile(v, p):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--attack", default="sentence", choices=ATTACKS)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"--out-dir {args.out_dir} is not empty; use a fresh folder.")
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    by_msg, true_label = load(args.preds)
    smish = sorted(m for m, t in true_label.items() if t == "smishing")
    ham = sorted(m for m, t in true_label.items() if t == "ham")
    all_ids = smish + ham
    models = sorted({m for mid in by_msg for m in by_msg[mid]})
    ns, nh = len(smish), len(ham)
    print(f"[data] {len(all_ids)} messages (smishing={ns}, ham={nh})  attack={args.attack}",
          file=sys.stderr)

    KEYS = ["FP_corrected", "FN_corrected", "FP_new", "FN_new"]
    point = {}
    for m in models:
        c, r = transitions_for(by_msg, true_label, all_ids, m, args.attack)
        point[m] = {"counts": c, "rates": r, "net_row": net_gain_pct(c)}

    reps = {m: defaultdict(list) for m in models}
    for b in range(args.B):
        draw = ([smish[rng.randrange(ns)] for _ in range(ns)] +
                [ham[rng.randrange(nh)] for _ in range(nh)])
        for m in models:
            c_b, r = transitions_for(by_msg, true_label, draw, m, args.attack)
            for k in KEYS + ["net"]:
                reps[m][k].append(r.get(k, float("nan")))
            reps[m]["net_row"].append(net_gain_pct(c_b))
        if (b + 1) % 100 == 0:
            print(f"  [boot] {b+1}/{args.B}", file=sys.stderr)

    S = lambda m: m.split("/")[-1]
    def col(m, k):
        v = point[m]["rates"].get(k, float("nan"))
        lo, hi = pctile(reps[m][k], 0.025), pctile(reps[m][k], 0.975)
        return f"{v:.2f} [{lo:.2f},{hi:.2f}]"
    def col_row(m):
        v = point[m]["net_row"]
        lo, hi = pctile(reps[m]["net_row"], 0.025), pctile(reps[m]["net_row"], 0.975)
        return f"{v:.2f} [{lo:.2f},{hi:.2f}]"

    L = ["=" * 116,
         f"TRANSITION DECOMPOSITION — attack={args.attack}   B={args.B}",
         "cluster bootstrap by original message, stratified by class",
         "=" * 116, "",
         "Every matched MESSAGE-PROMPT-VARIANT ROW falls in exactly one cell, determined by the",
         "clean prediction and the perturbed prediction.",
         "",
         "  FP_corrected   ham   : smishing -> ham        a pre-existing FALSE POSITIVE was corrected",
         "  FN_corrected   smish : ham -> smishing        a pre-existing MISS was corrected",
         "  FP_new         ham   : ham -> smishing        a false positive was CREATED (the benign-flip)",
         "  FN_new         smish : smishing -> ham        the attack EVADED           (the ASR numerator)",
         "",
         "  net = (FP_corrected + FN_corrected) - (FP_new + FN_new).",
         "        POSITIVE = the perturbation IMPROVED accuracy.",
         "        net = -dAcc  (dAcc = clean - perturbed, POSITIVE = degradation, in",
         "        bootstrap_analysis.py). Exact on both-parseable rows; on the full matched",
         "        set they differ by the unparsed-involved rows shown below.",
         "",
         "SIGN SUMMARY (state in the paper exactly):",
         "  dAcc          = clean - perturbed   POSITIVE = degradation, NEGATIVE = improvement",
         "  transition net= corrections - errors POSITIVE = improvement,  NEGATIVE = degradation",
         "  => transition net = -dAcc, on the SAME matched set under the SAME weighting.",
         "",
         "WEIGHTING: the transition rates below are MESSAGE-WEIGHTED — averaged within a",
         "source message (over its prompts and surviving variants), then across messages, so",
         "every message carries equal weight, matching ASR and the conditional benign-flip.",
         "NOTE: the paper's primary dAcc (bootstrap_analysis.py) is VARIANT-weighted, over",
         "matched message-prompt-variant rows — a DIFFERENT weighting. net = -dAcc therefore",
         "holds exactly only when compared on the same (row) basis; the message-weighted net",
         "here is the comparable-with-ASR view, not the exact algebraic inverse of dAcc.",
         "The ROW-weighted net column below IS that exact inverse (net_row = -dAcc_variant).",
         "Raw row counts are given further down as supporting detail ONLY: they must NOT be",
         "compared across attacks, because the number of retained variants differs by attack.",
         ""]

    L += [f"{'model':16s} {'FP_corrected':>19s} {'FN_corrected':>19s} {'FP_new':>19s} "
          f"{'FN_new (evasion)':>19s} {'net (msg)':>19s} {'net_row (=-dAcc)':>21s}",
          "-" * 138]
    for m in models:
        L.append(f"{S(m):16s} {col(m,'FP_corrected'):>19s} {col(m,'FN_corrected'):>19s} "
                 f"{col(m,'FP_new'):>19s} {col(m,'FN_new'):>19s} {col(m,'net'):>19s} "
                 f"{col_row(m):>21s}")

    L += ["", "-" * 116,
          "Raw counts over matched message-prompt-variant rows (supporting detail; do NOT compare",
          "across attacks — denominators differ)", "-" * 116,
          f"{'model':16s} {'matched rows':>14s} {'FP_corr':>9s} {'FN_corr':>9s} "
          f"{'FP_new':>9s} {'FN_new':>9s} {'unpars':>9s} {'messages':>9s}"]
    for m in models:
        c, r = point[m]["counts"], point[m]["rates"]
        L.append(f"{S(m):16s} {c['matched']:>14d} {c['FP_corrected']:>9d} "
                 f"{c['FN_corrected']:>9d} {c['FP_new']:>9d} {c['FN_new']:>9d} "
                 f"{c['unparsed_involved']:>9d} {r['_n_messages']:>9d}")

    L += ["", "-" * 116, "READING", "-" * 116]
    for m in models:
        r = point[m]["rates"]
        corr = r["FP_corrected"] + r["FN_corrected"]
        share = 100 * r["FP_corrected"] / corr if corr else float("nan")
        L.append(f"  {S(m):16s} net(msg) {r['net']:+.2f}%   net_row {point[m]['net_row']:+.2f}%   "
                 f"of all corrections, {share:.0f}% are false positives corrected on ham")
    L += ["",
          "net_row equals the negative of the variant-weighted dAcc in bootstrap_analysis.py",
          "on the same matched rows; net(msg) is the message-weighted view comparable with",
          "ASR and benign-flip. They differ only through weighting, not through arithmetic.",
          "",
          "These are behavioural measurements. They show that surface form materially",
          "affects predictions; they do not establish the models' internal decision",
          "mechanism, and semantic preservation is only automatically approximated."]

    open(os.path.join(args.out_dir, "transitions.txt"), "w").write("\n".join(L))
    print("\n".join(L))

    with open(os.path.join(args.out_dir, "transitions.json"), "w") as f:
        json.dump({"attack": args.attack, "B": args.B,
                   "weighting_headline": "message", "weighting_net_row": "row",
                   "sign": {"dacc": "clean-perturbed; +degradation",
                            "net": "corrections-errors; +improvement; net=-dacc"},
                   "counts": {m: dict(point[m]["counts"]) for m in models},
                   "net_row": {m: {"point": point[m]["net_row"],
                                   "ci_lo": pctile(reps[m]["net_row"], 0.025),
                                   "ci_hi": pctile(reps[m]["net_row"], 0.975)}
                               for m in models},
                   "rates": {m: {k: {"point": point[m]["rates"].get(k),
                                     "ci_lo": pctile(reps[m][k], 0.025),
                                     "ci_hi": pctile(reps[m][k], 0.975)}
                                 for k in KEYS + ["net"]} for m in models}}, f, indent=2)
    print(f"\n[write] {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
