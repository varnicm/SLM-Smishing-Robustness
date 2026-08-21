"""
build_nli_goldset.py
--------------------
Step 1 of validating the label-consistency metric properly.

WHY. We found the working recipe (deberta-v3 + rationale+verdict hypothesis) on only SIX
hand-made examples. Tuning framings until 6 examples pass is overfitting to six examples
-- the same mistake that sank the grounding metric. To report a credible number in the
paper we need a real validation set that YOU hand-judge, and then we pick the framing on
it and report the agreement.

WHAT THIS DOES. Samples real (message, rationale, predicted-label) triples from the
actual explanation data, stratified so the set is not all easy cases:
    - correct predictions on smishing        (rationale should be supported)
    - correct predictions on ham             (rationale should be supported)
    - INCORRECT predictions                  (rationale argues for the wrong label ->
                                              often unsupported by the message)
    - short messages                         (the known failure mode: NLI accepted a
                                              fabricated rationale on "Happy new years
                                              melody!")
    - long / verbose rationales
Writes a CSV with an empty `human_label` column for you to fill in.

HOW TO LABEL. For each row ask ONE question:
    "Reading the message, is the rationale's account of it SUPPORTED?"
      supported     -> the rationale describes things that are actually in the message
                       and its verdict follows from them
      unsupported   -> the rationale asserts things the message does not contain,
                       or its verdict does not follow
Put `supported` or `unsupported` in the human_label column. Skip anything genuinely
ambiguous (leave blank) rather than guessing -- ambiguous rows pollute the validation.

Usage:
    python build_nli_goldset.py --preds outputs/expl_all.jsonl \\
        --messages outputs/smishtank_perturbed.jsonl \\
        --out outputs/nli_goldset.csv --n 30
"""
import argparse, csv, json, random, re

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)


def rationale_text(rec):
    raw = rec.get("rationale") or rec.get("raw") or ""
    m = _FINAL_RE.search(raw)
    return (raw[:m.start()] if m else raw).strip()


def norm(x):
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--messages", required=True)
    ap.add_argument("--out", default="outputs/nli_goldset.csv")
    ap.add_argument("--n", type=int, default=30, help="target gold-set size")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)

    # id -> clean message
    clean_msg = {}
    for line in open(args.messages, encoding="utf-8"):
        d = json.loads(line)
        clean_msg.setdefault(d["id"], d["original_message"])

    # collect candidate rows (clean condition only -- message text is unambiguous there)
    buckets = {
        "correct_smishing": [],
        "correct_ham": [],
        "incorrect": [],       # model got the label wrong -> rationale argues wrong case
        "short_message": [],   # known NLI failure mode
        "long_rationale": [],
    }

    for line in open(args.preds, encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] != "clean":
            continue
        if "t3" not in r["template"] and "t4" not in r["template"]:
            continue
        rat = rationale_text(r)
        msg = clean_msg.get(r["id"])
        if not rat or not msg:
            continue

        pred, true = norm(r["pred"]), norm(r["true_label"])
        row = {
            "id": r["id"],
            "model": r["model"].split("/")[-1],
            "template": r["template"],
            "pred": pred,
            "true_label": true,
            "message": msg,
            "rationale": rat,
        }

        if pred != true:
            buckets["incorrect"].append(row)
        elif pred == "smishing":
            buckets["correct_smishing"].append(row)
        else:
            buckets["correct_ham"].append(row)

        if len(msg) < 40:
            buckets["short_message"].append(row)
        if len(rat) > 900:
            buckets["long_rationale"].append(row)

    # stratified sample
    per = max(2, args.n // len(buckets))
    picked, seen = [], set()
    for name, rows in buckets.items():
        random.shuffle(rows)
        take = 0
        for row in rows:
            key = (row["id"], row["model"], row["template"])
            if key in seen:
                continue
            seen.add(key)
            row["bucket"] = name
            picked.append(row)
            take += 1
            if take >= per:
                break
        print(f"[bucket] {name:18s} available={len(rows):6d}  sampled={take}")

    random.shuffle(picked)   # so you don't label bucket-by-bucket and bias yourself

    fields = ["human_label", "bucket", "model", "template", "id", "pred", "true_label",
              "message", "rationale"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in picked:
            row["human_label"] = ""      # <-- you fill this in
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"\n[write] {len(picked)} rows -> {args.out}")
    print("Fill the human_label column with: supported | unsupported   (blank = skip)")
    print("Then run:  python validate_nli_framings.py --gold " + args.out)


if __name__ == "__main__":
    main()
