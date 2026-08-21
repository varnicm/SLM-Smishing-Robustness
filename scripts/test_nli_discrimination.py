"""
test_nli_discrimination.py
--------------------------
HARD test of the label-consistency metric. The easy check (20 correct-prediction
rationales, all ~0.95) proves nothing: of course a good rationale for a correct
prediction is supported by its message.

The metric is only worth anything if it DISCRIMINATES. This script runs four
populations through the deberta config and compares the score distributions.
No manual labeling required -- the populations are defined by data we already have.

  (A) CORRECT predictions   -- model got the label right; rationale should be supported.
  (B) WRONG predictions     -- model got the label WRONG. Its rationale argues for a
                               label the message does not support. These SHOULD score
                               lower. This is the key comparison.
  (C) SHUFFLED rationales   -- a real rationale paired with a DIFFERENT random message.
                               Should score low. If it doesn't, the metric is ignoring
                               the message entirely and is worthless.
  (D) FLIPPED verdict       -- the real rationale, but with the OPPOSITE verdict sentence
                               appended. Should score lower than (A). Tests whether the
                               verdict actually matters or is being ignored.

INTERPRETATION
  * If A >> C, the metric is genuinely reading the message (not just rating fluency).
  * If A >  B, it detects rationales that argue the wrong case.
  * If A >  D, the verdict sentence is doing work.
  * If the distributions overlap, the metric is not measuring what we claim, and
    label-consistency should be dropped like grounding was.

Usage:
    python test_nli_discrimination.py --preds outputs/expl_all.jsonl \\
        --messages outputs/smishtank_perturbed.jsonl --n 150
"""
import argparse, json, random, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/nli-deberta-v3-large"
_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)


def norm(x):
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}: return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:   return "ham"
    return x


def rationale_text(raw):
    m = _FINAL_RE.search(raw or "")
    return ((raw[:m.start()] if m else raw) or "").strip()


def verdict(label):
    if label == "smishing":
        return "This message is classified as a phishing or scam message."
    return "This message is classified as benign. It is not a phishing attempt or a scam."


def flip(label):
    return "ham" if label == "smishing" else "smishing"


def summarize(name, scores):
    if not scores:
        print(f"  {name:26s} (no samples)")
        return None
    scores = sorted(scores)
    n = len(scores)
    mean = sum(scores) / n
    med = scores[n // 2]
    hi = sum(1 for s in scores if s >= 0.5) / n
    print(f"  {name:26s} n={n:4d}  mean={mean:.3f}  median={med:.3f}  "
          f"frac>=0.5={hi:.2f}")
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--messages", required=True)
    ap.add_argument("--n", type=int, default=150, help="samples per population")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    # message lookup
    msgs = {}
    for l in open(args.messages, encoding="utf-8"):
        d = json.loads(l)
        msgs[(d["id"], d["attack_class"], d["seed"])] = d["perturbed_message"]

    correct, wrong = [], []
    for l in open(args.preds, encoding="utf-8"):
        r = json.loads(l)
        if r["condition"] != "perturbed":
            continue
        rat = rationale_text(r.get("raw"))
        msg = msgs.get((r["id"], r["attack_class"], r["seed"]))
        if not rat or not msg:
            continue
        pred, true = norm(r["pred"]), norm(r["true_label"])
        rec = {"msg": msg, "rat": rat, "pred": pred, "model": r["model"].split("/")[-1]}
        (correct if pred == true else wrong).append(rec)

    print(f"[data] correct-prediction rationales: {len(correct)}")
    print(f"[data] WRONG-prediction rationales:   {len(wrong)}")

    random.shuffle(correct); random.shuffle(wrong)
    A = correct[:args.n]
    B = wrong[:args.n]

    # load NLI
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL)
    mdl.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        mdl = mdl.cuda()
    i2l = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    ei = next(i for i, l in i2l.items() if "entail" in l)
    print(f"[nli] {MODEL} id2label={i2l} entail_idx={ei}\n")

    def p_entail(premise, hypothesis):
        x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        if cuda:
            x = {k: v.cuda() for k, v in x.items()}
        with torch.no_grad():
            return float(F.softmax(mdl(**x).logits, -1)[0][ei])

    # (A) correct predictions
    sA = [p_entail(r["msg"], f"{r['rat']} {verdict(r['pred'])}") for r in A]

    # (B) wrong predictions
    sB = [p_entail(r["msg"], f"{r['rat']} {verdict(r['pred'])}") for r in B]

    # (C) shuffled: rationale paired with a DIFFERENT message
    others = [r["msg"] for r in correct]
    sC = []
    for r in A:
        other = random.choice(others)
        if other == r["msg"]:
            continue
        sC.append(p_entail(other, f"{r['rat']} {verdict(r['pred'])}"))

    # (D) flipped verdict on a correct rationale
    sD = [p_entail(r["msg"], f"{r['rat']} {verdict(flip(r['pred']))}") for r in A]

    print("=" * 70)
    print("SCORE DISTRIBUTIONS (higher = message supports the rationale+verdict)")
    print("=" * 70)
    mA = summarize("A correct predictions", sA)
    mB = summarize("B WRONG predictions", sB)
    mC = summarize("C shuffled message", sC)
    mD = summarize("D flipped verdict", sD)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if mA is None:
        return
    checks = [
        ("metric reads the MESSAGE (A > C)", mC is not None and mA - mC > 0.15),
        ("metric detects wrong-label rationales (A > B)", mB is not None and mA - mB > 0.10),
        ("verdict sentence matters (A > D)", mD is not None and mA - mD > 0.10),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if all(ok for _, ok in checks):
        print("\n  -> metric discriminates. Label-consistency is reportable.")
    else:
        print("\n  -> metric does NOT discriminate on at least one axis.")
        print("     Treat label-consistency with the same skepticism as grounding.")


if __name__ == "__main__":
    main()
