"""
analyze.py  (Task 4)
--------------------
Turn the raw predictions (preds.jsonl) into Dimension-1 results:

  - Clean accuracy per model x template
  - ASR (Attack Success Rate): of smishing messages the model got RIGHT when clean,
    what fraction flip to 'ham' after perturbation  (the core robustness metric)
  - dAcc: clean accuracy - perturbed accuracy (overall degradation)
  - Benign-flip rate (control): of ham messages correct when clean, fraction that
    flip to 'smishing' after perturbation
  - PAR (Prompt Agreement Rate): per (model, message, condition), agreement of the
    predicted label across the 4 templates vs the majority label  -> prompt sensitivity
  - Breakdowns by attack_class (character/word/sentence/multi)

Runs on CPU in seconds. Usage:
    python analyze.py --preds outputs/preds.jsonl --out-dir outputs/analysis
Works on PARTIAL data too (skips model/template cells with no data).
"""
import argparse, json, os, sys
from collections import defaultdict


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        rows.append(json.loads(line))
    return rows


def key_clean(model, template, mid):
    return (model, template, mid)


def norm_label(x):
    """Canonicalize labels so 0/1, 'benign', 'phishing', case variants, etc. all
    map to 'smishing'/'ham'. Prevents subset/denominator checks from silently
    failing on label-format differences.
    NOTE: 'spam' is deliberately NOT mapped to 'smishing' — spam can include
    non-phishing promotional messages, so collapsing it into smishing would
    corrupt ground truth. An unexpected 'spam' label passes through unmapped
    (and thus visibly) rather than being silently absorbed."""
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load(args.preds)
    print(f"[data] {len(rows)} predictions", file=sys.stderr)

    # normalize all labels up front so every comparison downstream is consistent
    for r in rows:
        r["pred"] = norm_label(r.get("pred"))
        r["true_label"] = norm_label(r.get("true_label"))

    # Index clean predictions: (model, template, id) -> pred
    clean_pred = {}
    for r in rows:
        if r["condition"] == "clean":
            clean_pred[key_clean(r["model"], r["template"], r["id"])] = r["pred"]

    # ---------- Clean accuracy per model x template ----------
    clean_stats = defaultdict(lambda: [0, 0])  # (model,template) -> [correct, total]
    for r in rows:
        if r["condition"] != "clean":
            continue
        c, t = clean_stats[(r["model"], r["template"])]
        clean_stats[(r["model"], r["template"])][1] += 1
        if r["pred"] == r["true_label"]:
            clean_stats[(r["model"], r["template"])][0] += 1

    # ---------- ASR + benign-flip + flip-rate, per model x template x attack ----------
    # Three DIFFERENT denominators (adviser-specified):
    #   flip_rate : ALL perturbed examples  -> general behavioral sensitivity
    #               (does the label change clean->perturbed at all, regardless of
    #                correctness). NOT attack success.
    #   ASR       : only CLEAN-CORRECT smishing -> genuine evasion. Restricting the
    #               denominator avoids counting cases already wrong before perturbation
    #               (or accidentally 'fixed' by it).
    #   benign    : only CLEAN-CORRECT ham -> false-alarm control (ham->smishing).
    asr = defaultdict(lambda: [0, 0])        # key -> [flipped, eligible(clean-correct smish)]
    benign = defaultdict(lambda: [0, 0])     # key -> [flipped, eligible(clean-correct ham)]
    flip = defaultdict(lambda: [0, 0])       # key -> [flipped(clean!=pert), all perturbed]
    pacc = defaultdict(lambda: [0, 0])       # perturbed accuracy [correct,total]

    for r in rows:
        if r["condition"] != "perturbed":
            continue
        model, tmpl, attack = r["model"], r["template"], r["attack_class"]
        ck = key_clean(model, tmpl, r["id"])
        clean_of = clean_pred.get(ck)
        # perturbed accuracy
        pacc[(model, tmpl, attack)][1] += 1
        if r["pred"] == r["true_label"]:
            pacc[(model, tmpl, attack)][0] += 1
        # label flip rate — ALL perturbed examples (general sensitivity, not attack success)
        if clean_of is not None:
            flip[(model, tmpl, attack)][1] += 1
            if r["pred"] != clean_of:
                flip[(model, tmpl, attack)][0] += 1
        # ASR (smishing side) — only if model was correct on clean
        if r["true_label"] == "smishing" and clean_of == "smishing":
            asr[(model, tmpl, attack)][1] += 1
            if r["pred"] == "ham":
                asr[(model, tmpl, attack)][0] += 1
        # benign-flip control (ham side) — only if model was correct on clean
        if r["true_label"] == "ham" and clean_of == "ham":
            benign[(model, tmpl, attack)][1] += 1
            if r["pred"] == "smishing":
                benign[(model, tmpl, attack)][0] += 1

    # ---------- PAR: prompt agreement across templates ----------
    # For each (model, id, condition, attack, seed) gather preds across templates;
    # PAR = fraction of templates agreeing with the majority label.
    bucket = defaultdict(dict)  # (model,cond,attack,seed,id) -> {template: pred}
    for r in rows:
        k = (r["model"], r["condition"], r["attack_class"], r["seed"], r["id"])
        bucket[k][r["template"]] = r["pred"]
    par_by_model = defaultdict(lambda: [0.0, 0])
    for k, preds in bucket.items():
        labels = list(preds.values())
        if len(labels) < 2:
            continue
        # majority label. NOTE: a 2-2 split across the 4 templates makes the chosen
        # majority arbitrary, but the agreement VALUE is still well-defined (0.5),
        # so PAR is unaffected; only the (unused) majority label itself is ambiguous.
        maj = max(set(labels), key=labels.count)
        agree = labels.count(maj) / len(labels)
        par_by_model[k[0]][0] += agree
        par_by_model[k[0]][1] += 1

    # ---------- write summary ----------
    def pct(pair):
        c, n = pair
        return f"{100*c/n:.1f}" if n else "n/a"

    out = os.path.join(args.out_dir, "summary.txt")
    with open(out, "w") as f:
        models = sorted({r["model"] for r in rows})
        templates = sorted({r["template"] for r in rows})
        attacks = ["character", "word", "sentence", "multi"]

        f.write("=== CLEAN ACCURACY (model x template) ===\n")
        for m in models:
            row = [f"{m.split('/')[-1]:28s}"]
            for t in templates:
                row.append(f"{t}:{pct(clean_stats.get((m,t),[0,0]))}")
            f.write("  ".join(row) + "\n")

        # ---- clean acc, perturbed acc, and dAcc (degradation), pooled over templates+attacks ----
        f.write("\n=== ACCURACY DEGRADATION by model: clean_acc, perturbed_acc, dAcc (pooled) ===\n")
        for m in models:
            # clean accuracy pooled over templates
            cc = ct = 0
            for t in templates:
                x = clean_stats.get((m, t), [0, 0]); cc += x[0]; ct += x[1]
            clean_a = (100 * cc / ct) if ct else float("nan")
            # perturbed accuracy pooled over templates+attacks
            pc = pt = 0
            for t in templates:
                for a in attacks:
                    x = pacc.get((m, t, a), [0, 0]); pc += x[0]; pt += x[1]
            pert_a = (100 * pc / pt) if pt else float("nan")
            dacc = clean_a - pert_a
            f.write(f"{m.split('/')[-1]:28s} clean={clean_a:5.1f}  "
                    f"perturbed={pert_a:5.1f}  dAcc={dacc:+5.1f}\n")

        f.write("\n=== LABEL FLIP RATE % (all perturbed; general sensitivity, NOT attack success) by model x attack (pooled over templates) ===\n")
        for m in models:
            row = [f"{m.split('/')[-1]:28s}"]
            for a in attacks:
                fl = tot = 0
                for t in templates:
                    x = flip.get((m,t,a),[0,0]); fl += x[0]; tot += x[1]
                row.append(f"{a}:{(100*fl/tot):.1f}" if tot else f"{a}:n/a")
            f.write("  ".join(row) + "\n")

        f.write("\n=== ASR % (smishing->ham flips, CLEAN-CORRECT denominator) by model x attack (pooled over templates) ===\n")
        for m in models:
            row = [f"{m.split('/')[-1]:28s}"]
            for a in attacks:
                fl = tot = 0
                for t in templates:
                    x = asr.get((m,t,a),[0,0]); fl += x[0]; tot += x[1]
                row.append(f"{a}:{(100*fl/tot):.1f}" if tot else f"{a}:n/a")
            f.write("  ".join(row) + "\n")

        f.write("\n=== BENIGN-FLIP % (ham->smishing, CLEAN-CORRECT control) by model x attack (pooled over templates) ===\n")
        for m in models:
            row = [f"{m.split('/')[-1]:28s}"]
            for a in attacks:
                fl = tot = 0
                for t in templates:
                    x = benign.get((m,t,a),[0,0]); fl += x[0]; tot += x[1]
                row.append(f"{a}:{(100*fl/tot):.1f}" if tot else f"{a}:n/a")
            f.write("  ".join(row) + "\n")

        f.write("\n=== PAR (prompt agreement rate) by model ===\n")
        for m in models:
            s, n = par_by_model.get(m, [0,0])
            f.write(f"{m.split('/')[-1]:28s} PAR={s/n:.3f} (n={n})\n" if n else f"{m}: n/a\n")

    print(open(out).read())
    print(f"[write] {out}", file=sys.stderr)

    # Also dump machine-readable JSON for making paper tables/plots later
    dump = {
        "clean_accuracy": {f"{m}|{t}": clean_stats[(m,t)] for (m,t) in clean_stats},
        "asr": {f"{m}|{t}|{a}": asr[(m,t,a)] for (m,t,a) in asr},
        "label_flip_rate": {f"{m}|{t}|{a}": flip[(m,t,a)] for (m,t,a) in flip},
        "benign_flip": {f"{m}|{t}|{a}": benign[(m,t,a)] for (m,t,a) in benign},
        "perturbed_acc": {f"{m}|{t}|{a}": pacc[(m,t,a)] for (m,t,a) in pacc},
        "par_by_model": {m: par_by_model[m] for m in par_by_model},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[write] {os.path.join(args.out_dir,'metrics.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
