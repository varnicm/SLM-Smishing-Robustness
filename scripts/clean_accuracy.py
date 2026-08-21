import argparse, json
from collections import defaultdict


def norm(x):
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x                       # 'unparsed' passes through and counts as incorrect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    args = ap.parse_args()

    overall = defaultdict(lambda: [0, 0])          # model -> [correct, total]
    by_tmpl = defaultdict(lambda: [0, 0])          # (model, template) -> [correct, total]
    by_class = defaultdict(lambda: [0, 0])         # (model, true_label) -> [correct, total]

    for line in open(args.preds, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "clean":              # ONLY unperturbed messages
            continue
        m = r["model"]
        pred, true = norm(r["pred"]), norm(r["true_label"])
        ok = int(pred == true)
        overall[m][0] += ok;                overall[m][1] += 1
        by_tmpl[(m, r["template"])][0] += ok; by_tmpl[(m, r["template"])][1] += 1
        by_class[(m, true)][0] += ok;       by_class[(m, true)][1] += 1

    models = sorted(overall, key=lambda m: -overall[m][0] / overall[m][1])
    templates = sorted({t for (_, t) in by_tmpl})
    S = lambda m: m.split("/")[-1]
    pct = lambda c: f"{100*c[0]/c[1]:.1f}" if c[1] else "n/a"

    print("=" * 92)
    print("CLEAN-MESSAGE ACCURACY  (unperturbed messages only)")
    print("=" * 92)
    print(f"{'model':22s} {'overall':>8s}" + "".join(f"{t[:18]:>20s}" for t in templates))
    print("-" * 92)
    for m in models:
        print(f"{S(m):22s} {pct(overall[m]):>8s}" +
              "".join(f"{pct(by_tmpl[(m,t)]):>20s}" for t in templates))

    print()
    print("-" * 92)
    print("Broken down by true class — this is where a biased model shows itself:")
    print("-" * 92)
    print(f"{'model':22s} {'smishing':>10s} {'ham':>10s} {'gap':>10s}")
    for m in models:
        s, h = by_class[(m, "smishing")], by_class[(m, "ham")]
        gs = 100 * s[0] / s[1] if s[1] else float("nan")
        gh = 100 * h[0] / h[1] if h[1] else float("nan")
        print(f"{S(m):22s} {gs:>10.1f} {gh:>10.1f} {gs-gh:>+10.1f}")

    print()
    print(f"clean rows per model: {overall[models[0]][1]}")
    print("(= n messages x n prompt configurations)")
    print()
    print("A large positive gap means the model catches smishing but misclassifies")
    print("legitimate messages — the same directional bias the benign-flip rate exposes")
    print("under perturbation, already visible before any attack is applied.")


if __name__ == "__main__":
    main()
