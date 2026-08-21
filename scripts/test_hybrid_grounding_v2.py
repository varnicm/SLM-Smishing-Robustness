"""
test_hybrid_grounding_v2.py
---------------------------
HYBRID grounding, v2. Two changes from v1:

  (1) GROUND TRUTH = the ORIGINAL (clean) message, not the perturbed one.
      Rationale: perturbations are semantics-preserving by construction (similarity
      gate >= 0.65; URLs/code tokens preserved). So the cues in the clean message ARE
      the cues in the perturbed message -- only the surface form changed. Checking cue
      presence on clean text avoids the character-attack problem where "deliv-ered" or
      "urgnet" defeats literal matching and the checker (not the model) fails.
      IMPORTANT: the MODEL still saw only the perturbed text. We never feed clean text
      to the model; the clean message is used solely as the measurement ground truth.

  (2) URGENCY is kept, with a literature-grounded lexicon.
      Urgency is an established persuasion principle in the phishing literature and IS
      a detectable linguistic attribute. v1's list was too shallow. We expand it to the
      standard markers: explicit urgency words, deadlines/time pressure, consequence
      framing (account will be suspended/restricted), and delivery/action failure
      framing -- all of which are the textual signatures used in prior work.

Method:
  STEP 1  "Did the model CLAIM this cue?"  -> NLI on the RATIONALE.
          premise=rationale, hypothesis="The message contains <cue>."
          (Validated: correctly ignores denials like "there is no urgency here.")
  STEP 2  "Is that cue actually present?"  -> objective check on the CLEAN message.
          Not an interpretive NLI hypothesis -- the verifiable factual core of the cue.

  UNSUPPORTED (fabrication) = claimed in the rationale AND absent from the clean message.

LIMITATION for the paper: we verify that the cited evidence EXISTS, not that the
model's interpretation of it is correct (a benign URL described as "suspicious" counts
as grounded -- a link is present; we do not adjudicate suspicion).

Usage:
    python test_hybrid_grounding_v2.py --preds outputs/expl_all.jsonl \
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

_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9\-]*\.(com|net|org|info|co|io|in|link|pw|app|fi|ly|me|us|biz|xyz|site|club|gov)\b\S*)",
    re.IGNORECASE)

_BRANDS = [
    "amazon","paypal","usps","ups","fedex","dhl","irs","i.r.s","netflix","apple","google",
    "microsoft","bank","wells fargo","wells farg","chase","citi","navyfed","navy federal",
    "t mobile","t-mobile","verizon","at&t","walmart","target","lowe's","lowes","costco",
    "ebay","venmo","zelle","cash app","coinbase","edd","dmv","hmrc","nhs","visa",
    "mastercard","amex","american express","whatsapp","instagram","facebook","spotify",
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

# Literature-grounded urgency markers (persuasion / scarcity / time-pressure signatures):
_URGENCY_PATTERNS = [
    # explicit urgency lexicon
    "urgent","urgently","immediately","immediate","asap","right away","act now","act fast",
    "hurry","quickly","promptly","without delay","don't delay","dont delay",
    # deadlines / time pressure
    "expire","expiring","expires","within 12","within 24","within 48","within 72",
    "24 hours","12 hours","48 hours","today","tonight","deadline","final notice",
    "last chance","limited time","limited","before ","by tomorrow","time-sensitive",
    # consequence framing / account threat (standard phishing urgency signature)
    "suspend","suspended","suspension","locked","lock","restricted","restrict","on hold",
    "will be closed","terminated","deactivat","blocked","disabled","cancel",
    "if not verified","if you do not","failure to","avoid interruption","to avoid",
    # action-failure / pending framing (delivery scams)
    "cannot be delivered","could not be delivered","cannot process","unable to",
    "pending","incomplete","failed","overdue","needs attention","requires action",
    "action required","needs urgent","tried to deliver","attempted delivery",
]


def _has_url(msg):     return bool(_URL_RE.search(msg))
def _has_brand(msg):   m = msg.lower(); return any(b in m for b in _BRANDS)
def _has_any(msg, ps): m = msg.lower(); return any(p in m for p in ps)


def factual_core_present(cue, message):
    """Objectively: is the cue's verifiable factual core in the (CLEAN) message?"""
    if cue == "link":          return _has_url(message)
    if cue == "impersonation": return _has_brand(message)
    if cue == "personal_info": return _has_any(message, _PERSONAL_PATTERNS)
    if cue == "payment":       return _has_any(message, _PAYMENT_PATTERNS)
    if cue == "urgency":       return _has_any(message, _URGENCY_PATTERNS)
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

    # id -> ORIGINAL (clean) message. This is the grounding ground-truth.
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
        print(f"CLEAN MSG (ground truth): {orig[:140]}")
        print(f"RAT: {rat[:160]}")
        for cue, hyp in CUE_CLAIM_HYP.items():
            if not nli.entails(rat, f"The message contains {hyp}."):
                continue
            tot_claimed += 1
            grounded = factual_core_present(cue, orig)     # <-- checked on CLEAN message
            if not grounded:
                tot_unsup += 1
            tag = "GROUNDED" if grounded else ">>> UNSUPPORTED (fabricated)"
            print(f"   cue={cue:14s} claimed=YES  present_in_clean={grounded}   {tag}")
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
