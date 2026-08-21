"""
check_progress.py
-----------------
Two reports over the current prediction files:
  (A) COMPLETION  — how many predictions each model has vs the ~80,932 target,
                    and how many templates finished.
  (B) CoT COVERAGE — for the CoT templates (t3, t4), how many rationales fit
                     within the 400-char stored cap vs were truncated (== 400).
                     Truncated rationales are NOT usable for Dimension 2 as-is.

Usage:
    python check_progress.py                      # scans outputs/preds_*.jsonl
    python check_progress.py --dir outputs        # custom dir
"""
import argparse, glob, json, os
from collections import defaultdict

TARGET_PER_MODEL = 80932   # 2107 clean + 18126 perturbed, x4 templates
CAP = 400                  # the raw[:400] storage cap in run_inference.py


def scan(path):
    """Return per-file stats without loading everything into memory at once."""
    n = 0
    templates = set()
    cot_total = 0
    cot_truncated = 0
    cot_lens = []
    unparsed_cot = 0
    by_template = defaultdict(int)
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        t = r.get("template", "?")
        templates.add(t)
        by_template[t] += 1
        if "cot" in t:
            cot_total += 1
            L = len(r.get("raw", ""))
            cot_lens.append(L)
            if L >= CAP:
                cot_truncated += 1
            if r.get("pred") == "unparsed":
                unparsed_cot += 1
    return {
        "n": n,
        "templates": sorted(templates),
        "by_template": dict(by_template),
        "cot_total": cot_total,
        "cot_truncated": cot_truncated,
        "cot_avg_len": (sum(cot_lens) // len(cot_lens)) if cot_lens else 0,
        "cot_max_len": max(cot_lens) if cot_lens else 0,
        "unparsed_cot": unparsed_cot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "preds_*.jsonl")))
    # exclude pilot / speed-test / combined files
    files = [f for f in files if not any(x in f for x in
             ("pilot", "test", "speedtest", "qwentest", "preds.jsonl", "v2_"))]

    if not files:
        print("no per-model prediction files found in", args.dir)
        return

    print("=" * 78)
    print("(A) COMPLETION  (target per model = {:,})".format(TARGET_PER_MODEL))
    print("=" * 78)
    print(f"{'model':42s} {'records':>9s} {'%':>5s}  {'templates done':>14s}")
    print("-" * 78)
    stats = {}
    for f in files:
        s = scan(f)
        stats[f] = s
        name = os.path.basename(f).replace("preds_", "").replace(".jsonl", "")
        pct = 100 * s["n"] / TARGET_PER_MODEL
        # a template is "done" if it has ~20,233 records (2107 clean + 18126 pert)
        done = sum(1 for t, c in s["by_template"].items() if c >= 20000)
        print(f"{name:42s} {s['n']:>9,} {pct:>4.0f}%  {done:>3d} / 4  {s['templates']}")

    print()
    print("=" * 78)
    print("(B) CoT COVERAGE  (rationales at {} chars are TRUNCATED -> not usable for Dim 2)".format(CAP))
    print("=" * 78)
    print(f"{'model':42s} {'CoT recs':>9s} {'trunc':>7s} {'trunc%':>7s} {'avg_len':>8s} {'unparsed':>9s}")
    print("-" * 78)
    for f in files:
        s = stats[f]
        if s["cot_total"] == 0:
            name = os.path.basename(f).replace("preds_", "").replace(".jsonl", "")
            print(f"{name:42s} {'--- no CoT records yet ---':>40s}")
            continue
        name = os.path.basename(f).replace("preds_", "").replace(".jsonl", "")
        tp = 100 * s["cot_truncated"] / s["cot_total"]
        print(f"{name:42s} {s['cot_total']:>9,} {s['cot_truncated']:>7,} "
              f"{tp:>6.1f}% {s['cot_avg_len']:>8d} {s['unparsed_cot']:>9,}")

    print()
    print("Interpretation:")
    print("  trunc% low  (<~5%)  -> CoT rationales mostly fit; usable for Dim 2 as-is.")
    print("  trunc% high (>~20%) -> that model needs a CoT re-run with a larger cap")
    print("                          (only t3/t4) to capture full rationales.")
    print("  unparsed (CoT)      -> if >0, the FINAL tag was missed; check that model.")


if __name__ == "__main__":
    main()
