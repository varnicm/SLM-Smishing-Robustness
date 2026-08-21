"""
test_urgency_nli.py
-------------------
FOCUSED TEST: can NLI reliably verify URGENCY presence in a message?

Motivation. Urgency is an established persuasion technique in the phishing literature
and should be extractable from text. Our lexical list keeps missing paraphrased forms
("hard time delivering", "to dispute", "you've won ... claim"), which over-flags models
as fabricating. Urgency, unlike a URL or a brand name, is a PRAGMATIC property with
many surface realizations -- so a semantic check (entailment) is the linguistically
appropriate verification method, not a keyword list.

This script does NOT patch anything. It answers one question:
    Does NLI (premise=message, hypothesis="this message conveys urgency/time pressure")
    fire on genuinely urgent messages and stay silent on benign ones?

If YES -> urgency gets semantic verification; lexical stays for the surface cues
          (link/brand/personal-info/payment). One principle: verify each cue by the
          method matching its linguistic nature.
If NO  -> urgency resists automatic verification, and THAT is an honest finding to
          report, not something to paper over.

We test several hypothesis phrasings, since NLI is sensitive to wording.

Usage:
    python test_urgency_nli.py
"""
import re

# ---- gold set: messages we can judge by hand ------------------------------------
# (text, human_judgment_urgent?)  -- drawn from the real corpus + the cases our
# lexical list got wrong.
GOLD = [
    # --- genuinely urgent smishing (lexical list CAUGHT these) ---
    ("[I.R.S] You will receive a tax refund of $573.00, but cannot process it due to "
     "incomplete personal information. Please confirm your information in the link.", True),
    ("The USPS package has arrived at the warehouse and cannot be delivered due to "
     "incomplete address information. Please confirm your address in the link within 12 hours.", True),
    ("Your EDD (UI online) account needs urgent attention due to some irregular activities. "
     "Your account will be restricted if not verified.", True),
    ("Amazon.com We temporarily placed your Amazon suspended to Restore follow instruction below.", True),
    ("WELLS FARG O: Account temporarily locked. Visit use.app.link/wf", True),

    # --- genuinely urgent smishing (lexical list MISSED these) ---
    ("Hey, it's John here, I'm having a hard time delivering your parcel. "
     "Please fill in the missing details here: zskuyyd.com/aj4BdNxG", True),
    ("#NavyFed: $870 withdrawn at ATMSW-2. To dispute, go to bit.ly/navyauth", True),
    ("Verizon You've won a prize! Go to bit.ly/yourprize001 to claim your $500 Amazon gift card", True),
    ("We regret to inform you that your Netflix membership has been placed on hold due to "
     "issues with your billing information. We kindly request to update your billing.", True),

    # --- NOT urgent: smishing but no time pressure ---
    ("This is CJ with POR. Please take a moment to complete this survey about politics "
     "in San Diego: https://www.surveymonkey.com/r/F8VLMMV", False),
    ("Dennis, July Tournament Pass $500 in Prizes Fortnite, LoL, SSBU Signup at "
     "ggsimplicity.com/tournaments Reply STOP", False),
    ("See the intimate pictures Kelly released 4 you! Unlock below...http://dtwxzf.com/003u52r", False),
    ("Lowe's retail company would like your help on this official questionnaire, Anybody "
     "that enters this are to receive a no cost product surveypurpose.com/wv95u", False),

    # --- benign ham (must NOT fire) ---
    ("Happy new years melody!", False),
    ("Sorry, I'll came to ur place later..", False),
    ("What time you thinkin of goin?", False),
    ("Was doing my test earlier. I appreciate you. Will call you tomorrow.", False),
    ("Does cinema plus drink appeal tomo? * Is a fr thriller by director i like on at mac at 8.30.", False),
    ("I tagged MY friends that you seemed to count as YOUR friends.", False),

    # --- benign but literally says "urgent" (hard case: should fire) ---
    ("Hello which the site to download songs its urgent pls", True),
    ("K.:)do it at evening da:)urgent:)", True),
]

HYPOTHESES = [
    "This message creates a sense of urgency.",
    "This message pressures the recipient to act quickly.",
    "The message demands immediate action or has a deadline.",
    "This message conveys urgency or time pressure.",
]

# the lexical list currently in use, for side-by-side comparison
_URGENCY_PATTERNS = [
    "urgent","urgently","immediately","immediate","asap","right away","act now","act fast",
    "hurry","quickly","promptly","expire","expiring","expires","within 12","within 24",
    "24 hours","12 hours","today","deadline","final notice","last chance","limited time",
    "suspend","suspended","locked","restricted","on hold","terminated","deactivat",
    "if not verified","failure to","cannot be delivered","cannot process","unable to",
    "pending","incomplete","failed","overdue","needs attention","action required",
]

def lexical_urgent(msg):
    m = msg.lower()
    return any(p in m for p in _URGENCY_PATTERNS)


class NLI:
    def __init__(self, model_name="roberta-large-mnli", device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device, self.torch = device, torch
        self.ENTAIL = 2

    def entail_prob(self, premise, hypothesis):
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=256).to(self.device)
        with self.torch.no_grad():
            probs = self.model(**x).logits[0].softmax(-1)
        return float(probs[self.ENTAIL])


def evaluate(pred_fn, name):
    tp = fp = tn = fn = 0
    wrong = []
    for msg, gold in GOLD:
        p = pred_fn(msg)
        if gold and p:       tp += 1
        elif gold and not p: fn += 1; wrong.append(("MISSED  ", msg))
        elif not gold and p: fp += 1; wrong.append(("FALSE+  ", msg))
        else:                tn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"\n--- {name} ---")
    print(f"  accuracy={acc:.2f}  precision={prec:.2f}  recall={rec:.2f}  "
          f"(tp={tp} fp={fp} tn={tn} fn={fn})")
    for tag, m in wrong:
        print(f"    {tag} {m[:80]}")
    return acc


def main():
    print("GOLD SET:", len(GOLD), "messages "
          f"({sum(1 for _, g in GOLD if g)} urgent, {sum(1 for _, g in GOLD if not g)} not)")

    # baseline: the current lexical list
    evaluate(lexical_urgent, "LEXICAL (current keyword list)")

    nli = NLI()
    for hyp in HYPOTHESES:
        for thresh in (0.5, 0.7):
            evaluate(lambda m, h=hyp, t=thresh: nli.entail_prob(m, h) >= t,
                     f"NLI  thresh={thresh}  hyp='{hyp}'")

    # show raw entailment probabilities for the best-looking hypothesis
    print("\n--- raw entailment probabilities (hyp: 'This message creates a sense of urgency.') ---")
    for msg, gold in GOLD:
        p = nli.entail_prob(msg, "This message creates a sense of urgency.")
        mark = "URGENT" if gold else "not   "
        print(f"  gold={mark}  p_entail={p:.3f}  {msg[:66]}")


if __name__ == "__main__":
    main()