#!/usr/bin/env python3
"""
pick_perturbation_examples.py — choose real smishing perturbation examples,
one (plus alternates) per attack level, for a paper table.

Reads outputs/perturbed_readable.csv (cols: id,attack_class,seed,label,valid,sim,
original,perturbed). Picks short, clearly-perturbed, semantics-preserving smishing
messages so the Original->Perturbed change is easy to see in a table.

  python pick_perturbation_examples.py
  python pick_perturbation_examples.py --min-len 30 --max-len 130 --topk 6
"""
import argparse, csv, json, sys

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows

def fitness(orig, pert, args):
    """higher = better for a table cell."""
    L = len(orig)
    if L < args.min_len or L > args.max_len:
        return -1e9
    score = -abs(L - args.ideal_len)          # prefer ~ideal length
    # penalize messages dominated by a long URL/hash token
    longest_tok = max((len(t) for t in orig.split()), default=0)
    if longest_tok > 30:
        score -= (longest_tok - 30) * 2
    # reward a visible but not huge amount of change
    if orig == pert:
        return -1e9
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/perturbed_readable.csv")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--ideal-len", type=int, default=85)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    rows = load(args.csv)
    attacks = {}
    for r in rows:
        attacks.setdefault(r["attack_class"], 0)
        attacks[r["attack_class"]] += 1
    print("attack_class counts in CSV:", attacks, "\n")

    order = ["character", "word", "sentence", "multi", "multi-level"]
    present = [a for a in order if a in attacks] + \
              [a for a in attacks if a not in order]

    chosen = {}
    for atk in present:
        cands = []
        for r in rows:
            if r["attack_class"] != atk:                 continue
            if (r["label"] or "").lower() != "smishing":  continue
            if str(r["valid"]).lower() not in ("true", "1"): continue
            o, p = r["original"], r["perturbed"]
            s = fitness(o, p, args)
            if s <= -1e8:                                 continue
            cands.append((s, float(r.get("sim", 0) or 0), r["id"], o, p))
        cands.sort(key=lambda x: x[0], reverse=True)
        print("=" * 78)
        print(f"ATTACK: {atk}   (showing top {min(args.topk,len(cands))} of {len(cands)})")
        for i, (s, sim, mid, o, p) in enumerate(cands[:args.topk], 1):
            print(f"\n  [{i}] {mid}  sim={sim:.3f}  len={len(o)}")
            print(f"      ORIG: {o}")
            print(f"      PERT: {p}")
        if cands:
            chosen[atk] = cands[0]
        print()

    with open("perturbation_examples_chosen.json", "w", encoding="utf-8") as fh:
        json.dump({a: {"id": c[2], "sim": c[1], "original": c[3], "perturbed": c[4]}
                   for a, c in chosen.items()}, fh, indent=2, ensure_ascii=False)
    print("Wrote perturbation_examples_chosen.json (the [1] pick per level)")
    if "sentence" not in attacks:
        print("\nNOTE: no 'sentence' rows here — they may live in "
              "outputs/smishtank_perturbed_v065_dedup_sentence.jsonl; tell me and "
              "I'll add that file as a source.")

if __name__ == "__main__":
    main()
