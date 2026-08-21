"""
test_nli_exact.py
-----------------
Reproduce the user's EXACT working notebook configuration, then apply that same
format to our real rationales.

The notebook used:
    premise    = "The message says\\n#<SMS>\\n#"        <-- no colon, SMS wrapped in #
    hypothesis = "This message is classified as benign because ... It is not a phishing
                  attempt or a scam."                   <-- the explanation, ALREADY a
                                                            well-formed declarative
    model      = cross-encoder/nli-deberta-v3-large
    -> entailment 0.9895

Two things I got wrong in the previous test:
  * I used "The message says:" (with a colon) and no # markers.
  * I wrapped the hypothesis in "This message is described as follows: ..." — but the
    notebook's hypothesis was already declarative on its own ("This message is
    classified as benign because...").

KEY INSIGHT to test: our model rationales mostly BEGIN declaratively too ("This message
exhibits multiple strong indicators of smishing..."). So maybe they need NO wrapper --
but they may need a LABEL SENTENCE appended, since the notebook's hypothesis explicitly
states the verdict ("classified as benign", "not a phishing attempt").

We test several hypothesis constructions to find which the NLI model can actually judge.

Usage:  python test_nli_exact.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cross-encoder/nli-deberta-v3-large"


# ---- premise formats --------------------------------------------------------
def prem_notebook(msg):
    """EXACT notebook format: no colon, SMS wrapped in # markers."""
    return f"The message says\n#{msg}\n#"

def prem_plain(msg):
    return msg


# ---- hypothesis formats -----------------------------------------------------
def hyp_raw(rat, label):
    """The rationale as-is."""
    return rat

def hyp_with_verdict(rat, label):
    """Rationale + an explicit verdict sentence, mirroring the notebook's hypothesis
    which explicitly stated the classification ('classified as benign ... not a phishing
    attempt or a scam')."""
    verdict = ("This message is classified as a phishing or scam message."
               if label == "smishing" else
               "This message is classified as benign. It is not a phishing attempt or a scam.")
    return f"{rat} {verdict}"

def hyp_verdict_only(rat, label):
    """Only the verdict — no rationale. Baseline to isolate the rationale's contribution."""
    return ("This message is classified as a phishing or scam message."
            if label == "smishing" else
            "This message is classified as benign. It is not a phishing attempt or a scam.")


# ---- GOLD: (message, rationale, model_label, expected) ----------------------
GOLD = [
    # the notebook's own case — deberta gave 0.9895 here
    ("#Ants were bathing & elephant jumped in2 pool All ants got out of water,1 ant climbed "
     "on d back of d elephant. All ants cried DUBA DUBA KE MAAR SAALE KO :-)",
     "This message is classified as benign because it provides light entertainment and "
     "educational content. It is not a phishing attempt or a scam.",
     "ham", "entailment"),

    # real Qwen rationales — all clearly support their labels
    ("Notif: From: Usps CONTENT: A parcel needs to be delivered for you but is pending an "
     "update. To update your preferences head to 83hm2uaj.kzra.in/?/OVA013",
     "This message exhibits multiple strong indicators of smishing. First, it uses a deceptive "
     "'Notif:' header to mimic an official notification while the sender ID is generic rather "
     "than the official USPS number. Second, the content creates a false sense of urgency by "
     "claiming a parcel is pending an update. Third, it includes a suspicious link.",
     "smishing", "entailment"),

    ("See the intimate pictures Kelly released 4 you! Unlock below...http://dtwxzf.com/003u52r",
     "This message exhibits multiple strong indicators of smishing. It begins with a sensational "
     "hook designed to trigger curiosity. The sender attempts to impersonate a specific individual "
     "to establish false credibility. Most critically, it includes a suspicious, non-branded URL.",
     "smishing", "entailment"),

    ("Sorry, I'll came to ur place later..",
     "This message exhibits several strong indicators of legitimacy. The grammar is slightly "
     "informal, which is common in casual text conversations between friends. There is no request "
     "for personal information, payment, or sensitive data. The sender is not impersonating a brand.",
     "ham", "entailment"),

    # rationales that CONTRADICT their message -> must NOT entail
    ("Happy new years melody!",
     "This message contains a suspicious link, a request for payment, and impersonates Amazon "
     "to steal the recipient's banking credentials.",
     "smishing", "not-entailment"),

    ("Was doing my test earlier. I appreciate you. Will call you tomorrow.",
     "This message urgently demands that the recipient confirm their social security number "
     "through an official IRS portal within 24 hours or their account will be suspended.",
     "smishing", "not-entailment"),
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    e_idx = next(i for i, l in id2label.items() if "entail" in l)
    print(f"MODEL: {MODEL}")
    print(f"label order: {id2label}  (entailment index = {e_idx})\n")

    def p_entail(premise, hypothesis):
        x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probs = F.softmax(model(**x).logits, dim=-1)[0]
        return float(probs[e_idx])

    configs = [
        ("notebook-premise + raw-rationale",        prem_notebook, hyp_raw),
        ("notebook-premise + rationale+verdict",    prem_notebook, hyp_with_verdict),
        ("notebook-premise + verdict-only",         prem_notebook, hyp_verdict_only),
        ("plain-premise    + rationale+verdict",    prem_plain,    hyp_with_verdict),
    ]

    for cname, pf, hf in configs:
        print("=" * 78)
        print(f"CONFIG: {cname}")
        correct = 0
        for msg, rat, label, expected in GOLD:
            p = p_entail(pf(msg), hf(rat, label))
            got = "entailment" if p >= 0.5 else "not-entailment"
            ok = "OK " if got == expected else "BAD"
            if got == expected:
                correct += 1
            print(f"  {ok}  p={p:.3f}  expect={expected:15s} {msg[:44]}")
        print(f"  >>> {correct}/{len(GOLD)} correct\n")


if __name__ == "__main__":
    main()
