#!/usr/bin/env python3
"""
Find clean smishing messages that have a valid variant in ALL FOUR attacks, so we
can show one message perturbed four ways (or pick four tidy per-level examples).
Excludes adult-content spam and non-English messages; prefers short, professional text.

  python pick_cross_attack_examples.py
"""
import argparse, csv, collections

BLOCK = ["sultry","sexy","girls","girl ","explicit","intimate","nude","naked",
         "porn","xxx","horny","hookup","escort","date night","get laid","boobs",
         "hot singles","flirt","released 4 you"]
FRENCH = ["votre","veuillez","colis","vérifier","recevoir","envoyé"]
ATTACKS = ["character","word","sentence","multi"]

def is_english(s):
    if not s: return False
    ascii_ratio = sum(1 for c in s if ord(c) < 128) / len(s)
    low = s.lower()
    return ascii_ratio > 0.95 and not any(f in low for f in FRENCH)

def clean(s):
    low = (s or "").lower()
    return not any(b in low for b in BLOCK)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/perturbed_readable.csv")
    ap.add_argument("--min-len", type=int, default=45)
    ap.add_argument("--max-len", type=int, default=135)
    ap.add_argument("--sim-lo", type=float, default=0.70)
    ap.add_argument("--sim-hi", type=float, default=0.98)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    by_id = collections.defaultdict(dict)   # id -> {attack: row}
    orig = {}
    for r in rows:
        if (r["label"] or "").lower() != "smishing":     continue
        if str(r["valid"]).lower() not in ("true","1"):  continue
        by_id[r["id"]][r["attack_class"]] = r
        orig[r["id"]] = r["original"]

    cands = []
    for mid, d in by_id.items():
        o = orig[mid]
        if not (args.min_len <= len(o) <= args.max_len):  continue
        if not is_english(o) or not clean(o):             continue
        if not all(a in d for a in ATTACKS):              continue
        sims = {a: float(d[a]["sim"]) for a in ATTACKS}
        if not all(args.sim_lo <= sims[a] <= args.sim_hi for a in ATTACKS): continue
        if any(d[a]["perturbed"] == o for a in ATTACKS):  continue
        longest = max(len(t) for t in o.split())
        score = -abs(len(o) - 90) - max(0, longest - 28) * 2
        cands.append((score, mid, o, d, sims))

    cands.sort(key=lambda x: x[0], reverse=True)
    print(f"{len(cands)} clean English smishing messages with all 4 attacks; top {args.topk}:\n")
    for score, mid, o, d, sims in cands[:args.topk]:
        print("=" * 80)
        print(f"{mid}   ORIG: {o}")
        for a in ATTACKS:
            print(f"  {a:9s} sim={sims[a]:.3f}: {d[a]['perturbed']}")
        print()

if __name__ == "__main__":
    main()
