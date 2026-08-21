"""
test_nli_deberta.py
-------------------
Diagnose and fix the label-consistency metric.

WHAT WE LEARNED (from the user's earlier working notebook):
  On the SAME premise/hypothesis pair:
      roberta-large-mnli            -> NEUTRAL      0.993   (wrong)
      cross-encoder/nli-deberta-v3-large -> ENTAILMENT  0.990   (right)
  We have been using roberta-large-mnli the whole time. That is very likely why
  label-consistency (and grounding) failed: WRONG MODEL, not wrong method.

  The notebook also framed the premise as:
      "The message says\n<SMS>"
  i.e. it wrapped the raw SMS in a declarative sentence, because NLI models are trained
  on well-formed prose and a bare SMS fragment is out of distribution.

THIS SCRIPT tests, on cases where we ALREADY KNOW the right answer:
  * roberta vs deberta-v3
  * raw premise vs "The message says ..." framing
  * raw hypothesis vs framed hypothesis
so we pick a configuration on evidence rather than by guessing.

IMPORTANT: label index order DIFFERS between the two models. We read id2label from
the model config instead of hardcoding (hardcoding would silently produce garbage).

Usage:
    python test_nli_deberta.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODELS = [
    "roberta-large-mnli",                    # what we used (suspected broken)
    "cross-encoder/nli-deberta-v3-large",    # what the user's notebook validated
]


# ---------------------------------------------------------------------------
# GOLD CASES — we know the correct answer for every one of these.
# (message, rationale, expected)  expected in {"entailment", "not-entailment"}
# ---------------------------------------------------------------------------
GOLD = [
    # --- Qwen rationales we inspected by hand: all CLEARLY support their labels ---
    ("Notif: From: Usps CONTENT: A parcel needs to be delivered for you but is pending an "
     "update. To update your preferences head to 83hm2uaj.kzra.in/?/OVA013",
     "This message exhibits multiple strong indicators of smishing. First, it uses a deceptive "
     "'Notif:' header to mimic an official notification while the sender ID is generic rather "
     "than the official USPS number. Second, the content creates a false sense of urgency by "
     "claiming a parcel is pending an update. Third, it includes a suspicious link.",
     "entailment"),

    ("See the intimate pictures Kelly released 4 you! Unlock below...http://dtwxzf.com/003u52r",
     "This message exhibits multiple strong indicators of smishing. It begins with a sensational "
     "hook designed to trigger curiosity. The sender attempts to impersonate a specific individual "
     "to establish false credibility. Most critically, it includes a suspicious, non-branded URL.",
     "entailment"),

    ("Sorry, I'll came to ur place later..",
     "This message exhibits several strong indicators of legitimacy. The grammar is slightly "
     "informal, which is common in casual text conversations between friends. There is no request "
     "for personal information, payment, or sensitive data. The sender is not impersonating a brand.",
     "entailment"),

    # --- a rationale that CONTRADICTS its message (should NOT entail) ---
    ("Happy new years melody!",
     "This message contains a suspicious link, a request for payment, and impersonates Amazon "
     "to steal the recipient's banking credentials.",
     "not-entailment"),

    ("Was doing my test earlier. I appreciate you. Will call you tomorrow.",
     "This message urgently demands that the recipient confirm their social security number "
     "through an official IRS portal within 24 hours or their account will be suspended.",
     "not-entailment"),

    # --- the user's original notebook case (known: deberta says ENTAILMENT 0.99) ---
    ("#Ants were bathing & elephant jumped in2 pool All ants got out of water,1 ant climbed on "
     "d back of d elephant. All ants cried DUBA DUBA KE MAAR SAALE KO :-)#",
     "This message is classified as benign because it provides light entertainment and "
     "educational content. It is not a phishing attempt or a scam.",
     "entailment"),
]


# ---- framing variants -------------------------------------------------------
def prem_raw(msg):    return msg
def prem_framed(msg): return f"The message says:\n{msg}"

def hyp_raw(rat):     return rat
def hyp_framed(rat):  return f"This message is described as follows: {rat}"

PREMISE_FRAMINGS   = [("raw", prem_raw), ("framed", prem_framed)]
HYPOTHESIS_FRAMINGS = [("raw", hyp_raw), ("framed", hyp_framed)]


class NLI:
    def __init__(self, model_name):
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # READ the label order from config — it differs between models!
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in self.id2label.items() if "entail" in l)

    def entail_prob(self, premise, hypothesis):
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probs = F.softmax(self.model(**x).logits, dim=-1)[0]
        return float(probs[self.entail_idx])


def main():
    for model_name in MODELS:
        print("=" * 78)
        print(f"MODEL: {model_name}")
        try:
            nli = NLI(model_name)
        except Exception as e:
            print(f"  could not load: {e}")
            continue
        print(f"  label order from config: {nli.id2label}  (entailment index = {nli.entail_idx})")

        for pf_name, pf in PREMISE_FRAMINGS:
            for hf_name, hf in HYPOTHESIS_FRAMINGS:
                correct = 0
                print(f"\n  --- premise={pf_name}, hypothesis={hf_name} ---")
                for msg, rat, expected in GOLD:
                    p = nli.entail_prob(pf(msg), hf(rat))
                    got = "entailment" if p >= 0.5 else "not-entailment"
                    ok = "OK " if got == expected else "BAD"
                    if got == expected:
                        correct += 1
                    print(f"    {ok}  p_entail={p:.3f}  expect={expected:15s} {msg[:44]}")
                print(f"    >>> {correct}/{len(GOLD)} correct")


if __name__ == "__main__":
    main()
