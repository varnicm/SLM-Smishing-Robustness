"""
transitions.py
--------------
Decompose the accuracy change under perturbation into its four constituent transitions,
so that WHERE an accuracy gain or loss comes from is visible rather than inferred.

WHY THIS EXISTS
    Under sentence-level paraphrasing, matched accuracy INCREASES for four models
    (negative dAcc, intervals excluding zero). The tempting explanation is that PEGASUS
    normalises informal SMS into regular English, removing the surface noise the models
    were over-reacting to, and thereby correcting false positives.

    That explanation is NOT established by dAcc, and it is not established by the lower
    benign-flip rate either: benign-flip only counts NEW errors among ham messages that
    were ALREADY correct. It cannot see whether paraphrasing FIXED ham messages that were
    already being misclassified. The two quantities are about different populations.

    The four transitions below settle it. Each matched (message, template) pair is placed
    into exactly one cell, defined by the clean prediction and the perturbed prediction:

      For HAM messages (true label = ham):
        FP_corrected   clean=smishing -> pert=ham        a false positive was FIXED
        FP_new         clean=ham      -> pert=smishing    a false positive was CREATED
                                                          (this is the benign-flip)
        (ham->ham and smishing->smishing are unchanged)

      For SMISHING messages (true label = smishing):
        FN_corrected   clean=ham      -> pert=smishing    a miss was FIXED
        FN_new         clean=smishing -> pert=ham         the attack EVADED (this is ASR)

    Accuracy change = (FP_corrected + FN_corrected) - (FP_new + FN_new), over the matched
    rows. If the gain under paraphrasing is dominated by FP_corrected, the
    surface-normalisation reading is supported. If it is not, that reading must be dropped.

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
    """Counts of each transition over the MATCHED rows for one model."""
    c = defaultdict(int)
    for mid in ids:
        slot = by_msg.get(mid, {}).get(model)
        if not slot:
            continue
        true = true_label[mid]
        for (tmpl, a, seed), pert in slot["pert"].items():
            if a != attack:
                continue
            clean = slot["clean"].get(tmpl)
            if clean is None:
                continue
            c["matched"] += 1

            # unparsed on either side: cannot be assigned a transition, but it IS an
            # accuracy event. Counted separately so the arithmetic still closes.
            if clean == UNPARSED or pert == UNPARSED:
                c["unparsed_involved"] += 1
                if clean == true and pert != true:
                    c["lost_to_unparsed"] += 1
                elif clean != true and pert == true:
                    c["gained_from_unparsed"] += 1
                continue

            if true == "ham":
                if clean == "smishing" and pert == "ham":
                    c["FP_corrected"] += 1              # false positive FIXED
                elif clean == "ham" and pert == "smishing":
                    c["FP_new"] += 1                    # false positive CREATED (benign-flip)
                elif clean == "ham" and pert == "ham":
                    c["ham_stable_correct"] += 1
                else:
                    c["ham_stable_wrong"] += 1
            else:                                        # true == smishing
                if clean == "ham" and pert == "smishing":
                    c["FN_corrected"] += 1              # miss FIXED
                elif clean == "smishing" and pert == "ham":
                    c["FN_new"] += 1                    # EVASION (ASR numerator)
                elif clean == "smishing" and pert == "smishing":
                    c["smish_stable_correct"] += 1
                else:
                    c["smish_stable_wrong"] += 1
    return c


def net_gain_pct(c):
    """Accuracy change attributable to the transitions, as a % of matched rows.
    POSITIVE = perturbation improved accuracy."""
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

    KEYS = ["FP_corrected", "FP_new", "FN_corrected", "FN_new"]
    point = {m: transitions_for(by_msg, true_label, all_ids, m, args.attack) for m in models}

    reps = {m: defaultdict(list) for m in models}
    for b in range(args.B):
        draw = ([smish[rng.randrange(ns)] for _ in range(ns)] +
                [ham[rng.randrange(nh)] for _ in range(nh)])
        for m in models:
            c = transitions_for(by_msg, true_label, draw, m, args.attack)
            for k in KEYS:
                reps[m][k].append(pct_of(c, k))
            reps[m]["net"].append(net_gain_pct(c))
        if (b + 1) % 100 == 0:
            print(f"  [boot] {b+1}/{args.B}", file=sys.stderr)

    S = lambda m: m.split("/")[-1]
    L = ["=" * 108,
         f"TRANSITION DECOMPOSITION — attack={args.attack}   B={args.B}",
         "cluster bootstrap by original message, stratified by class",
         "=" * 108, "",
         "Every matched (message, prompt) pair falls in exactly one cell, from its clean",
         "prediction and its perturbed prediction. Percentages are of matched rows.",
         "",
         "  FP_corrected  ham:  smishing -> ham       a FALSE POSITIVE was FIXED by the perturbation",
         "  FP_new        ham:  ham -> smishing       a false positive was CREATED  (the benign-flip)",
         "  FN_corrected  smish: ham -> smishing      a MISS was fixed",
         "  FN_new        smish: smishing -> ham      the attack EVADED             (the ASR numerator)",
         "",
         "  net = (FP_corrected + FN_corrected) - (FP_new + FN_new), + unparsed transitions.",
         "        POSITIVE net = the perturbation IMPROVED accuracy.",
         "",
         "THE QUESTION THIS ANSWERS: if a model's accuracy rises under paraphrasing, is that",
         "because paraphrasing CORRECTS pre-existing false positives (FP_corrected large)?",
         "If yes, the surface-normalisation reading is supported. If not, it must be dropped.",
         ""]

    def col(m, k):
        p = pctile(reps[m][k], 0.025), pctile(reps[m][k], 0.975)
        v = pct_of(point[m], k) if k != "net" else net_gain_pct(point[m])
        return f"{v:.2f} [{p[0]:.2f},{p[1]:.2f}]"

    L += [f"{'model':16s} {'FP_corrected':>20s} {'FP_new':>20s} {'FN_corrected':>20s} "
          f"{'FN_new':>20s} {'net gain':>20s}", "-" * 108]
    for m in models:
        L.append(f"{S(m):16s} {col(m,'FP_corrected'):>20s} {col(m,'FP_new'):>20s} "
                 f"{col(m,'FN_corrected'):>20s} {col(m,'FN_new'):>20s} {col(m,'net'):>20s}")

    L += ["", "Raw counts (full sample):", ""]
    L.append(f"{'model':16s} {'matched':>9s} {'FP_corr':>9s} {'FP_new':>9s} "
             f"{'FN_corr':>9s} {'FN_new':>9s} {'unparsed':>9s}")
    for m in models:
        c = point[m]
        L.append(f"{S(m):16s} {c['matched']:>9d} {c['FP_corrected']:>9d} {c['FP_new']:>9d} "
                 f"{c['FN_corrected']:>9d} {c['FN_new']:>9d} {c['unparsed_involved']:>9d}")

    L += ["", "-" * 108, "READING", "-" * 108]
    for m in models:
        c = point[m]
        net = net_gain_pct(c)
        gain = c["FP_corrected"] + c["FN_corrected"]
        if gain:
            share = 100 * c["FP_corrected"] / gain
            L.append(f"  {S(m):16s} net {net:+.2f}%  |  of all corrections, "
                     f"{share:.0f}% are false positives fixed on ham")
        else:
            L.append(f"  {S(m):16s} net {net:+.2f}%  |  no corrections")

    open(os.path.join(args.out_dir, "transitions.txt"), "w").write("\n".join(L))
    print("\n".join(L))

    with open(os.path.join(args.out_dir, "transitions.json"), "w") as f:
        json.dump({"attack": args.attack, "B": args.B,
                   "counts": {m: dict(point[m]) for m in models},
                   "pct": {m: {k: {"point": pct_of(point[m], k),
                                   "ci_lo": pctile(reps[m][k], 0.025),
                                   "ci_hi": pctile(reps[m][k], 0.975)} for k in KEYS}
                           for m in models}}, f, indent=2)
    print(f"\n[write] {args.out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
