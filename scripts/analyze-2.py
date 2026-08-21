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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out-dir", default="outputs/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = load(args.preds)
    print(f"[data] {len(rows)} predictions", file=sys.stderr)

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

    # ---------- ASR + benign-flip, per model x template x attack ----------
    # ASR: smishing that was correct-when-clean, flips to ham after perturb
    # benign-flip: ham correct-when-clean, flips to smishing after perturb
    asr = defaultdict(lambda: [0, 0])        # key -> [flipped, eligible]
    benign = defaultdict(lambda: [0, 0])
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
        # ASR (smishing side) — only if model was correct on clean
        if r["true_label"] == "smishing" and clean_of == "smishing":
            asr[(model, tmpl, attack)][1] += 1
            if r["pred"] == "ham":
                asr[(model, tmpl, attack)][0] += 1
        # benign-flip control (ham side)
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
        # majority label
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

        f.write("\n=== ASR % (smishing->ham flips) by model x attack (avg over templates) ===\n")
        for m in models:
            row = [f"{m.split('/')[-1]:28s}"]
            for a in attacks:
                fl = tot = 0
                for t in templates:
                    x = asr.get((m,t,a),[0,0]); fl += x[0]; tot += x[1]
                row.append(f"{a}:{(100*fl/tot):.1f}" if tot else f"{a}:n/a")
            f.write("  ".join(row) + "\n")

        f.write("\n=== BENIGN-FLIP % (ham->smishing, control) by model x attack ===\n")
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
        "benign_flip": {f"{m}|{t}|{a}": benign[(m,t,a)] for (m,t,a) in benign},
        "perturbed_acc": {f"{m}|{t}|{a}": pacc[(m,t,a)] for (m,t,a) in pacc},
        "par_by_model": {m: par_by_model[m] for m in par_by_model},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(dump, f, indent=2)
    print(f"[write] {os.path.join(args.out_dir,'metrics.json')}", file=sys.stderr)


if __name__ == "__main__":
    main()
