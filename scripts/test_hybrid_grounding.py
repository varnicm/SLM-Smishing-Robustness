"""
test_hybrid_grounding.py
------------------------
HYBRID unsupported-cue (grounding) detection.

Why hybrid: neither pure method worked.
  * Pure lexical  -> over-counted: "no urgency here" was read as CLAIMING urgency.
  * Pure NLI      -> over-counted worse: the message rarely ENTAILS an interpretive
                     hypothesis like "contains a SUSPICIOUS link", so real links were
                     marked unsupported.

Hybrid uses each method where it actually works:
  STEP 1 - "Did the model CLAIM this cue?"      -> NLI on the RATIONALE.
           premise=rationale, hypothesis="The message contains <cue>."
           This handles negation correctly (validated: ham rationales that deny cues
           produce NO claims).
  STEP 2 - "Is that cue's FACTUAL CORE present?" -> OBJECTIVE CHECK on the MESSAGE.
           Not an interpretive NLI hypothesis. We check the verifiable fact underneath
           each cue:
             suspicious link  -> does the message contain a URL at all?      (regex)
             brand impersonation -> does it name a brand/institution?        (lookup)
             request for personal info -> does it ask to confirm/verify/etc? (patterns)
             request for payment -> does it mention money/payment/prize?     (patterns)
             urgency          -> does it contain urgency markers?            (patterns)

  A cue is UNSUPPORTED only if: the model CLAIMED it (step 1) AND its factual core is
  ABSENT from the message (step 2). That is genuine fabrication -- citing evidence that
  is not there.

LIMITATION (state in paper): we verify that the cited evidence EXISTS in the message,
not that the model's INTERPRETATION of it is correct. A legitimate URL called
"suspicious" counts as grounded (a link is present); we do not adjudicate suspicion.

Usage:
    python test_hybrid_grounding.py --preds outputs/expl_all.jsonl \
        --messages outputs/smishtank_perturbed.jsonl --limit 40
"""
import argparse, json, re

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)

# NLI hypotheses used ONLY for step 1 (did the rationale claim the cue?)
CUE_CLAIM_HYP = {
    "link":          "a suspicious or unexpected web link",
    "impersonation": "impersonation of a company or brand",
    "personal_info": "a request for personal or sensitive information",
    "payment":       "a request for payment or money",
    "urgency":       "a sense of urgency or a demand for immediate action",
}

# ---- STEP 2: objective factual-core checks on the MESSAGE --------------------
_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9\-]*\.(com|net|org|info|co|io|in|link|pw|app|fi|ly|me|us|biz|xyz|site|club)\b\S*)",
    re.IGNORECASE)

_BRANDS = [
    "amazon","paypal","usps","ups","fedex","dhl","irs","netflix","apple","google",
    "microsoft","bank","wells fargo","chase","citi","navyfed","navy federal","t mobile",
    "t-mobile","verizon","at&t","att","walmart","target","lowe's","lowes","costco",
    "ebay","venmo","zelle","cash app","coinbase","edd","dmv","hmrc","nhs","visa",
    "mastercard","amex","american express","whatsapp","instagram","facebook","spotify",
]

_PERSONAL_PATTERNS = [
    "confirm your","verify your","update your","your account","personal information",
    "your details","your address","your information","ssn","social security","password",
    "pin","login","log in","sign in","credential","identity","card number","account number",
    "billing information","fill in","complete the form","provide your",
]

_PAYMENT_PATTERNS = [
    "$", "payment","pay ","paid","fee","refund","prize","won","win ","winner","gift card",
    "claim your","deposit","balance","invoice","billing","charge","withdraw","transaction",
    "money","cash","reward","bonus","debt","owe",
]

_URGENCY_PATTERNS = [
    "urgent","immediately","asap","right away","act now","act fast","hurry","expire",
    "expiring","within 24","within 12","within 48","today","deadline","final notice",
    "last chance","limited time","suspend","suspended","locked","restricted","on hold",
    "cannot be delivered","pending","failed","overdue","don't delay","dont delay","now",
    "quickly","soon","before ",
]


def _has_url(msg):
    return bool(_URL_RE.search(msg))

def _has_brand(msg):
    m = msg.lower()
    return any(b in m for b in _BRANDS)

def _has_any(msg, patterns):
    m = msg.lower()
    return any(p in m for p in patterns)


def factual_core_present(cue, message):
    """Objectively: is the FACT the cue refers to actually in the message?"""
    if cue == "link":
        return _has_url(message)
    if cue == "impersonation":
        return _has_brand(message)
    if cue == "personal_info":
        return _has_any(message, _PERSONAL_PATTERNS)
    if cue == "payment":
        return _has_any(message, _PAYMENT_PATTERNS)
    if cue == "urgency":
        return _has_any(message, _URGENCY_PATTERNS)
    return True


def rationale_text(rec):
    raw = rec.get("rationale") or rec.get("raw") or ""
    m = _FINAL_RE.search(raw)
    return (raw[:m.start()] if m else raw).strip()


class NLI:
    def __init__(self, model_name="roberta-large-mnli", device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device, self.torch = device, torch
        self.ENTAIL = 2   # 0=contradiction, 1=neutral, 2=entailment

    def entails(self, premise, hypothesis, thresh=0.5):
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=256).to(self.device)
        with self.torch.no_grad():
            probs = self.model(**x).logits[0].softmax(-1)
        return (int(probs.argmax()) == self.ENTAIL) or (float(probs[self.ENTAIL]) >= thresh)


def iter_records(path, limit=None):
    n = 0
    for line in open(path, encoding="utf-8"):
        yield json.loads(line)
        n += 1
        if limit and n >= limit:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--messages", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    msg_text = {}
    for r in iter_records(args.messages):
        msg_text[(r["id"], r["attack_class"], r["seed"])] = r["perturbed_message"]
        msg_text[(r["id"], None, None)] = r["original_message"]

    nli = NLI(device=args.device)

    seen = tot_claimed = tot_unsup = 0
    for r in iter_records(args.preds):
        if "t3" not in r["template"] and "t4" not in r["template"]:
            continue
        rat = rationale_text(r)
        msg = msg_text.get((r["id"], r.get("attack_class"), r.get("seed")))
        if not rat or msg is None:
            continue

        print("=" * 78)
        print(f"id={r['id']} cond={r['condition']} pred={r['pred']} true={r['true_label']}")
        print(f"MSG: {msg[:150]}")
        print(f"RAT: {rat[:170]}")
        for cue, hyp_phrase in CUE_CLAIM_HYP.items():
            # STEP 1 (NLI): did the rationale CLAIM this cue?
            if not nli.entails(rat, f"The message contains {hyp_phrase}."):
                continue
            tot_claimed += 1
            # STEP 2 (objective): is the cue's factual core in the message?
            grounded = factual_core_present(cue, msg)
            if not grounded:
                tot_unsup += 1
            tag = "GROUNDED" if grounded else ">>> UNSUPPORTED (fabricated)"
            print(f"   cue={cue:14s} claimed=YES  fact_present={grounded}   {tag}")
        seen += 1
        if seen >= args.limit:
            break

    print("=" * 78)
    if tot_claimed:
        print(f"[summary] {seen} rationales | claimed cues={tot_claimed} | "
              f"unsupported={tot_unsup} ({100*tot_unsup/tot_claimed:.1f}% of claimed)")
    else:
        print(f"[summary] {seen} rationales | no cues claimed")


if __name__ == "__main__":
    main()
