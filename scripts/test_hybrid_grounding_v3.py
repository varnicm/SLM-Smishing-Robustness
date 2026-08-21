"""
test_hybrid_grounding_v3.py
---------------------------
HYBRID grounding, v3 = v2 + DE-OBFUSCATION of the checker.

Why v3. Real smishing in the corpus is already obfuscated BY THE ATTACKER, e.g.
  msg-00047: "The USPS pack-age ... cannot be deliv-ered due to incom-plete address
              ... https://usp-s.searchtd.top"
The SLM reads through this fine (it cites the urgency cue). Our literal string matcher
does not: "cannot be deliv-ered" != "cannot be delivered", "usp-s" != "usps". So the
CHECKER failed where the MODEL succeeded, and the model was wrongly scored as
fabricating. That is a measurement failure, not a model failure.

Fix: normalize the message INSIDE THE CHECKER before pattern matching --
strip intra-word hyphens and stray spaces so obfuscated forms collapse to their
canonical forms. The MODEL never sees this; it only ever saw the raw text. We are
fixing the ruler, not the experiment.

Grounding ground-truth remains the ORIGINAL (clean) message: perturbations are
semantics-preserving, so the cues in the clean text are the cues in the perturbed text.

Method:
  STEP 1  "Did the model CLAIM this cue?"   -> NLI on the RATIONALE (handles negation).
  STEP 2  "Is the cue actually present?"    -> factual check on the DE-OBFUSCATED clean
                                               message (objective, not interpretive).
  UNSUPPORTED = claimed AND absent.

Usage:
    python test_hybrid_grounding_v3.py --preds outputs/expl_all.jsonl \
        --messages outputs/smishtank_perturbed.jsonl --limit 40
"""
import argparse, json, re

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)

CUE_CLAIM_HYP = {
    "link":          "a suspicious or unexpected web link",
    "impersonation": "impersonation of a company or brand",
    "personal_info": "a request for personal or sensitive information",
    "payment":       "a request for payment or money",
    "urgency":       "a sense of urgency or a demand for immediate action",
}


# ---------------------------------------------------------------------------
# DE-OBFUSCATION (checker-side only)
# ---------------------------------------------------------------------------
def deobfuscate(text):
    """Collapse attacker/perturbation surface obfuscation so literal matching can work.
      "pack-age"      -> "package"
      "deliv-ered"    -> "delivered"
      "in- formation" -> "information"
      "usp-s"         -> "usps"
      "u r g e n t"   -> "urgent"
    Only used to establish ground truth for the metric. The model never sees this."""
    t = text.lower()
    # join a hyphen (or stray space after it) that sits between word characters
    t = re.sub(r"(?<=\w)[-\u2010-\u2015]\s*(?=\w)", "", t)
    # join single letters separated by spaces (spaced-out obfuscation)
    t = re.sub(r"\b(?:\w\s){2,}\w\b", lambda m: m.group(0).replace(" ", ""), t)
    # collapse repeated whitespace
    t = re.sub(r"\s+", " ", t)
    return t


_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9\-]*\.(com|net|org|info|co|io|in|link|pw|app|fi|ly|me|us|biz|xyz|site|club|gov|top)\b\S*)",
    re.IGNORECASE)

_BRANDS = [
    "amazon","paypal","usps","ups","fedex","dhl","irs","netflix","apple","google",
    "microsoft","bank","wellsfargo","wells fargo","chase","citi","navyfed","navy federal",
    "tmobile","t mobile","verizon","at&t","walmart","target","lowes","lowe's","costco",
    "ebay","venmo","zelle","cashapp","coinbase","edd","dmv","hmrc","nhs","visa",
    "mastercard","amex","american express","whatsapp","instagram","facebook","spotify",
    "postal",
]

_PERSONAL_PATTERNS = [
    "confirm your","verify your","update your","your account","personal information",
    "your details","your address","your information","ssn","social security","password",
    "pin","login","log in","sign in","credential","identity","card number","account number",
    "billing information","fill in","complete the form","provide your","missing details",
    "incomplete address","incomplete personal","postal address","send me your",
]

_PAYMENT_PATTERNS = [
    "$","payment","pay ","paid","fee","refund","prize","won","winner","gift card",
    "claim your","deposit","balance","invoice","billing","charge","withdraw","withdrawn",
    "transaction","money","cash","reward","bonus","debt","owe","earn","no cost","free",
]

_URGENCY_PATTERNS = [
    "urgent","urgently","immediately","immediate","asap","right away","act now","act fast",
    "hurry","quickly","promptly","without delay","dont delay",
    "expire","expiring","expires","within 12","within 24","within 48","within 72",
    "24 hours","12 hours","48 hours","today","tonight","deadline","final notice",
    "last chance","limited time","limited","by tomorrow","time sensitive",
    "suspend","suspended","suspension","locked","restricted","restrict","on hold",
    "will be closed","terminated","deactivat","blocked","disabled",
    "if not verified","if you do not","failure to","avoid interruption","to avoid",
    "cannot be delivered","could not be delivered","cannot process","unable to",
    "pending","incomplete","failed","overdue","needs attention","requires action",
    "action required","needs urgent","tried to deliver","attempted delivery",
]


def _has_url(m):     return bool(_URL_RE.search(m))
def _has_brand(m):   return any(b in m for b in _BRANDS)
def _has_any(m, ps): return any(p in m for p in ps)


def factual_core_present(cue, message):
    """Objective check on the DE-OBFUSCATED clean message."""
    m = deobfuscate(message)
    if cue == "link":          return _has_url(m)
    if cue == "impersonation": return _has_brand(m)
    if cue == "personal_info": return _has_any(m, _PERSONAL_PATTERNS)
    if cue == "payment":       return _has_any(m, _PAYMENT_PATTERNS)
    if cue == "urgency":       return _has_any(m, _URGENCY_PATTERNS)
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
        self.ENTAIL = 2

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

    clean_msg = {}
    for r in iter_records(args.messages):
        clean_msg.setdefault(r["id"], r["original_message"])

    nli = NLI(device=args.device)

    seen = tot_claimed = tot_unsup = 0
    for r in iter_records(args.preds):
        if "t3" not in r["template"] and "t4" not in r["template"]:
            continue
        rat = rationale_text(r)
        orig = clean_msg.get(r["id"])
        if not rat or orig is None:
            continue

        print("=" * 78)
        print(f"id={r['id']} cond={r['condition']} pred={r['pred']} true={r['true_label']}")
        print(f"CLEAN: {orig[:130]}")
        print(f"DEOBF: {deobfuscate(orig)[:130]}")
        print(f"RAT:   {rat[:140]}")
        for cue, hyp in CUE_CLAIM_HYP.items():
            if not nli.entails(rat, f"The message contains {hyp}."):
                continue
            tot_claimed += 1
            grounded = factual_core_present(cue, orig)
            if not grounded:
                tot_unsup += 1
            tag = "GROUNDED" if grounded else ">>> UNSUPPORTED (fabricated)"
            print(f"   cue={cue:14s} claimed=YES  present={grounded}   {tag}")
        seen += 1
        if seen >= args.limit:
            break

    print("=" * 78)
    if tot_claimed:
        print(f"[summary] {seen} rationales | claimed={tot_claimed} | "
              f"unsupported={tot_unsup} ({100*tot_unsup/tot_claimed:.1f}% of claimed)")
    else:
        print(f"[summary] {seen} rationales | no cues claimed")


if __name__ == "__main__":
    main()
