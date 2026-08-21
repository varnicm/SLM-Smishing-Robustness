#!/usr/bin/env python3
"""
Reverse-engineer how the 1,055 benign messages were selected from smish-mandely.csv.
Matches the ham messages in smishtank_clean.csv back to their rows in the Mendeley
file and reports the selection pattern.

  python compare_benign_selection.py --mendeley /path/to/smish-mandely.csv \
                                     --clean outputs/smishtank_clean.csv
"""
import argparse, csv, collections, statistics

def norm(s):
    return " ".join(str(s or "").split())

def load_rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty: {path}")
    cols = rows[0].keys()
    # detect text column = the one with the longest average string
    def avglen(c): return statistics.mean(len(str(r.get(c) or "")) for r in rows)
    text_col = max(cols, key=avglen)
    # detect label column = values look like ham/spam/smishing/0/1
    label_col = None
    for c in cols:
        vals = {str(r.get(c) or "").strip().lower() for r in rows[:200]}
        if vals & {"ham", "spam", "smishing", "smish", "0", "1", "legit", "benign"}:
            label_col = c; break
    return rows, text_col, label_col

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mendeley", required=True)
    ap.add_argument("--clean", default="outputs/smishtank_clean.csv")
    args = ap.parse_args()

    mrows, mtext, mlabel = load_rows(args.mendeley)
    print(f"Mendeley: {len(mrows)} rows | text col='{mtext}' | label col='{mlabel}'")
    if mlabel:
        print("  label counts:", dict(collections.Counter(str(r.get(mlabel)).strip().lower() for r in mrows)))

    # index mendeley by normalized text -> list of row positions
    idx = collections.defaultdict(list)
    for i, r in enumerate(mrows):
        idx[norm(r.get(mtext))].append(i)

    # selected ham from clean file
    with open(args.clean, newline="", encoding="utf-8", errors="replace") as fh:
        clean = [r for r in csv.DictReader(fh) if (r.get("label") or "").strip().lower() == "ham"]
    print(f"\nSelected benign in clean file: {len(clean)}")

    positions, unmatched = [], 0
    for r in clean:
        hits = idx.get(norm(r.get("message")))
        if hits: positions.append(hits[0])
        else: unmatched += 1
    positions.sort()
    print(f"matched to Mendeley: {len(positions)}   unmatched: {unmatched}")
    if not positions:
        raise SystemExit("No matches — text normalization differs; tell me and I'll adjust matching.")

    lo, hi = positions[0], positions[-1]
    span = hi - lo + 1
    is_prefix = positions == list(range(len(positions)))
    is_contig = positions == list(range(lo, lo + len(positions)))
    coverage = len(positions) / len(mrows)
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    print("\n== selection pattern ==")
    print(f"  index range: {lo}..{hi}  (span {span} of {len(mrows)} rows)")
    print(f"  is exactly the FIRST {len(positions)} rows?   {is_prefix}")
    print(f"  is a CONTIGUOUS block?                 {is_contig}")
    print(f"  covers {coverage:.1%} of the file; median gap between picks = {statistics.median(gaps):.1f}")
    if mlabel:
        ham_total = sum(1 for r in mrows if str(r.get(mlabel)).strip().lower() in ("ham","legit","benign","0"))
        print(f"  ham rows in Mendeley: {ham_total}  -> selected {'ALL ham' if len(positions)>=ham_total-3 else 'a SUBSET of ham'}")

    # length filtering?
    sel = set(positions)
    sel_len = [len(str(mrows[i].get(mtext) or '')) for i in positions]
    rest = [len(str(mrows[i].get(mtext) or '')) for i in range(len(mrows)) if i not in sel]
    print(f"\n  msg length  selected: mean={statistics.mean(sel_len):.0f} med={statistics.median(sel_len):.0f} "
          f"min={min(sel_len)} max={max(sel_len)}")
    if rest:
        print(f"  msg length  NOT sel : mean={statistics.mean(rest):.0f} med={statistics.median(rest):.0f} "
              f"min={min(rest)} max={max(rest)}")

    # dedup?
    dup = sum(1 for c in collections.Counter(norm(r.get(mtext)) for r in mrows).values() if c > 1)
    print(f"\n  duplicate texts in Mendeley: {dup}")
    print("\nInterpretation hints:")
    print("  - is_prefix True  -> 'first 1,055 rows' (or first 1,055 ham)")
    print("  - is_contig True  -> a contiguous slice")
    print("  - spread across ~100% with random-looking gaps -> random sample (seed not recoverable)")
    print("  - selected length range much narrower than NOT-sel -> a length filter was applied")

if __name__ == "__main__":
    main()
