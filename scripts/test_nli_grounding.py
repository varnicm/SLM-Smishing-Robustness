"""
test_nli_grounding.py
---------------------
Small-batch test of NLI-BASED unsupported-cue (grounding) detection.

The OLD lexical method flags a cue as "claimed" whenever its keyword appears in the
rationale — so "no urgency here" wrongly counts 'urgency' as claimed, then unsupported.

This NLI version does two things differently:
  (1) AFFIRMATION check: only treat a cue as CLAIMED if the rationale asserts it is
      PRESENT. We test with NLI:
          premise    = rationale
          hypothesis = "The message contains <cue>."
      claimed  <=>  entailment (rationale asserts the cue), NOT contradiction/neutral.
      This drops "no urgency" because that yields contradiction, not entailment.
  (2) GROUNDING check: for a claimed cue, test whether the MESSAGE supports it:
          premise    = message
          hypothesis = "This message contains <cue>."
      grounded <=>  entailment. If NOT entailed, the claimed cue is UNSUPPORTED.

Runs on a small --limit sample and PRINTS each decision so we can eyeball whether it
behaves (especially on benign/ham messages that deny cues).

Usage:
    python test_nli_grounding.py --preds outputs/expl_all.jsonl \
        --messages outputs/smishtank_perturbed.jsonl --limit 40
"""
import argparse, json, re

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)

CUE_HYPOTHESES = {
    "urgency":       "a sense of urgency or a demand for immediate action",
    "suspicious link": "a suspicious or unexpected web link",
    "brand impersonation": "impersonation of a company or brand",
    "request for personal information": "a request for personal or sensitive information",
    "request for payment": "a request for payment or money",
}


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
        self.device = device
        self.torch = torch
        # roberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
        self.ENTAIL = 2

    def label(self, premise, hypothesis):
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=256).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**x).logits[0]
        probs = logits.softmax(-1)
        idx = int(probs.argmax())
        return idx, float(probs[self.ENTAIL])  # (argmax_label, entailment_prob)

    def entails(self, premise, hypothesis, thresh=0.5):
        idx, p_entail = self.label(premise, hypothesis)
        return (idx == self.ENTAIL) or (p_entail >= thresh)


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

    # message lookup: id + perturbed variant -> text. For clean we need originals too.
    msg_text = {}
    for r in iter_records(args.messages):
        msg_text[(r["id"], r["attack_class"], r["seed"])] = r["perturbed_message"]
        msg_text[(r["id"], None, None)] = r["original_message"]

    nli = NLI(device=args.device)

    # only look at perturbed explanation rows for this quick test
    seen = 0
    tot_claimed = tot_unsup = 0
    for r in iter_records(args.preds):
        if "t3" not in r["template"] and "t4" not in r["template"]:
            continue
        rat = rationale_text(r)
        if not rat:
            continue
        msg = msg_text.get((r["id"], r.get("attack_class"), r.get("seed")))
        if msg is None:
            continue

        print("=" * 80)
        print(f"id={r['id']} cond={r['condition']} pred={r['pred']} true={r['true_label']}")
        print(f"MSG: {msg[:160]}")
        print(f"RAT: {rat[:200]}")
        for cue, hyp_phrase in CUE_HYPOTHESES.items():
            # (1) does the rationale AFFIRM this cue is present?
            claim_hyp = f"The message contains {hyp_phrase}."
            claimed = nli.entails(rat, claim_hyp)
            if not claimed:
                continue
            tot_claimed += 1
            # (2) does the MESSAGE actually support it?
            grounded = nli.entails(msg, claim_hyp)
            status = "GROUNDED" if grounded else ">>> UNSUPPORTED"
            if not grounded:
                tot_unsup += 1
            print(f"   cue='{cue}': claimed=YES  message_supports={grounded}  {status}")
        seen += 1
        if seen >= args.limit:
            break

    print("=" * 80)
    print(f"[summary] over {seen} rationales: claimed={tot_claimed}, "
          f"unsupported={tot_unsup} "
          f"({100*tot_unsup/tot_claimed:.1f}% of claimed)" if tot_claimed else "no claims")


if __name__ == "__main__":
    main()
