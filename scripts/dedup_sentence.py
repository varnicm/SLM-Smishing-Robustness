"""
dedup_sentence.py
-----------------
Remove the duplicate sentence-level rows created by DETERMINISTIC PEGASUS decoding.

THE PROBLEM
    The paraphraser ran with num_beams=5, do_sample=False. Beam search always returns its
    single best hypothesis, so it CANNOT vary with a random seed. We verified this on the
    real data: all three seeds produced byte-identical output for 2,110 of 2,110 messages
    (100%), and the models — decoding greedily — then returned identical predictions on
    all of them (1,637/1,637 checked).

    Sentence-level therefore carries three rows per message where only ONE is informative.
    The other two are exact copies. Rates are unaffected by this (triplicating an
    identical observation does not move a proportion), but every sentence-level SAMPLE
    SIZE is inflated 3x, and any confidence interval computed on it would be
    correspondingly overconfident.

THE FIX
    Keep ONE sentence-level row per (model, template, message-id) and drop the redundant
    copies. No new inference is required: the predictions being discarded are identical to
    the one being kept. Character-, word- and multi-level rows are untouched — their seeds
    genuinely vary (only 0.9-1.5% collide).

    Before dropping anything we VERIFY the duplicates really are identical. If a
    sentence-level group is found whose predictions differ across seeds, that contradicts
    the determinism finding and the script ABORTS rather than silently discarding a real
    observation.

WHAT THIS SCRIPT DOES NOT DO
    It does not touch multi-level rows. Multi-level chains word -> character -> paraphrase;
    the word and character stages are seeded and do vary, so multi-level variants are
    genuinely distinct across seeds even though their paraphrase stage is deterministic.
    (Verified: only 0.9% of multi-level messages collide across seeds.)

Usage:
    python dedup_sentence.py --preds outputs/preds_norm.jsonl \\
        --out outputs/preds_dedup.jsonl
"""
import argparse, json, sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-seed", type=int, default=0,
                    help="which seed to retain for sentence-level rows")
    ap.add_argument("--allow-differing", action="store_true",
                    help="proceed even if sentence-level seeds DISAGREE (they should not; "
                         "use only to inspect, never to produce paper numbers)")
    args = ap.parse_args()

    rows = []
    for line in open(args.preds, encoding="utf-8"):
        rows.append(json.loads(line))
    print(f"[in]  {len(rows)} predictions", file=sys.stderr)

    # ---- verify the sentence-level duplicates really ARE duplicates ----
    groups = defaultdict(dict)     # (model, template, id) -> {seed: pred}
    for r in rows:
        if r.get("attack_class") == "sentence" and r["condition"] == "perturbed":
            groups[(r["model"], r["template"], r["id"])][r["seed"]] = r["pred"]

    differing = [k for k, s in groups.items() if len(s) > 1 and len(set(s.values())) > 1]
    print(f"[check] sentence-level groups: {len(groups)}", file=sys.stderr)
    print(f"[check] groups whose seeds DISAGREE: {len(differing)}", file=sys.stderr)

    if differing and not args.allow_differing:
        print("\nABORT: some sentence-level seeds produced DIFFERENT predictions.",
              file=sys.stderr)
        print("That contradicts the determinism finding — the seeds are not redundant,",
              file=sys.stderr)
        print("and dropping them would discard real observations. Investigate first.",
              file=sys.stderr)
        for k in differing[:5]:
            print(f"  {k} -> {groups[k]}", file=sys.stderr)
        raise SystemExit(1)

    # ---- keep exactly one sentence-level seed; pass everything else through ----
    kept, dropped = [], 0
    seen = set()
    for r in rows:
        is_sentence = (r.get("attack_class") == "sentence" and r["condition"] == "perturbed")
        if not is_sentence:
            kept.append(r)
            continue
        key = (r["model"], r["template"], r["id"])
        if key in seen:
            dropped += 1                     # a redundant copy
            continue
        # keep the requested seed if it exists in this group, else the first we meet
        if r["seed"] != args.keep_seed and args.keep_seed in groups[key]:
            dropped += 1
            continue
        seen.add(key)
        kept.append(r)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report ----
    def count(rs, attack):
        return sum(1 for r in rs
                   if r["condition"] == "perturbed" and r.get("attack_class") == attack)

    print(f"\n[out] {len(kept)} predictions  ({dropped} redundant sentence rows removed)",
          file=sys.stderr)
    print("\nrows per attack (perturbed only):", file=sys.stderr)
    for a in ("character", "word", "sentence", "multi"):
        print(f"  {a:10s} {count(rows,a):8d} -> {count(kept,a):8d}", file=sys.stderr)
    print(f"\n[write] {args.out}", file=sys.stderr)
    print("\nRates are unchanged by this operation (the removed rows are identical to the",
          file=sys.stderr)
    print("retained one). Sentence-level SAMPLE SIZES are now honest.", file=sys.stderr)


if __name__ == "__main__":
    main()
