"""
validate_nli_framings.py
------------------------
Step 2: pick the label-consistency framing ON EVIDENCE, using the gold set you labeled.

Runs several premise/hypothesis constructions through deberta-v3 (and roberta as a
baseline, to document why we switched) and reports accuracy / precision / recall against
your human labels. The winning configuration -- and its validated accuracy -- is what
goes in the paper.

What we already know from a 6-case probe (this script tests it properly):
  * roberta-large-mnli is unusable here (2/6; near-zero entailment on correct rationales)
  * deberta-v3-large + "rationale + explicit verdict sentence" scored 5/6
  * the "The message says #...#" premise framing made NO difference
  * verdict-ONLY (no rationale) scored 2/6 -- so the model is genuinely reading the
    rationale, not just the label. That is the key sanity check.

Usage:
    python validate_nli_framings.py --gold outputs/nli_goldset.csv
"""
import argparse, csv, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = [
    "cross-encoder/nli-deberta-v3-large",   # the candidate
    "roberta-large-mnli",                   # baseline, to document why we switched
]


# ---------- premise constructions -------------------------------------------
def prem_plain(msg):
    return msg

def prem_says(msg):
    return f"The message says\n#{msg}\n#"


# ---------- hypothesis constructions ----------------------------------------
def _verdict(label):
    if label == "smishing":
        return "This message is classified as a phishing or scam message."
    return "This message is classified as benign. It is not a phishing attempt or a scam."

def _verdict_contrastive(label):
    if label == "smishing":
        return ("This message is a phishing or scam message. "
                "It is not a legitimate message.")
    return ("This message is a legitimate message. "
            "It is not a phishing attempt or a scam.")

def _strip_markdown(rat):
    rat = re.sub(r"\*\*(.*?)\*\*", r"\1", rat)      # **bold** -> bold
    rat = re.sub(r"^\s*\d+[\.\)]\s*", "", rat, flags=re.M)   # "1. " list markers
    rat = re.sub(r"^\s*[-*]\s*", "", rat, flags=re.M)        # bullets
    rat = re.sub(r"\s+", " ", rat)
    return rat.strip()

def hyp_rat_only(rat, label):
    return rat

def hyp_rat_verdict(rat, label):
    return f"{rat} {_verdict(label)}"

def hyp_rat_verdict_contrastive(rat, label):
    return f"{rat} {_verdict_contrastive(label)}"

def hyp_clean_rat_verdict(rat, label):
    return f"{_strip_markdown(rat)} {_verdict(label)}"

def hyp_verdict_only(rat, label):
    return _verdict(label)


CONFIGS = [
    # (name, premise_fn, hypothesis_fn)
    ("plain + rationale only",              prem_plain, hyp_rat_only),
    ("plain + rationale+verdict",           prem_plain, hyp_rat_verdict),
    ("plain + rationale+verdict(contrast)", prem_plain, hyp_rat_verdict_contrastive),
    ("plain + clean-rationale+verdict",     prem_plain, hyp_clean_rat_verdict),
    ("says  + rationale+verdict",           prem_says,  hyp_rat_verdict),
    ("plain + VERDICT ONLY (sanity)",       prem_plain, hyp_verdict_only),
]


class NLI:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.cuda = torch.cuda.is_available()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        # READ the entailment index -- it differs across models (roberta=2, deberta=1)
        self.e_idx = next(i for i, l in self.id2label.items() if "entail" in l)

    def p_entail(self, premise, hypothesis):
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        if self.cuda:
            x = {k: v.cuda() for k, v in x.items()}
        with torch.no_grad():
            probs = F.softmax(self.model(**x).logits, dim=-1)[0]
        return float(probs[self.e_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    gold = []
    with open(args.gold, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("human_label") or "").strip().lower()
            if lab not in {"supported", "unsupported"}:
                continue          # blank / ambiguous -> skipped, as instructed
            gold.append({
                "msg": row["message"],
                "rat": row["rationale"],
                "pred": row["pred"].strip().lower(),
                "gold": lab,
                "bucket": row.get("bucket", ""),
            })

    if not gold:
        print("No labeled rows found. Fill the human_label column with "
              "'supported' / 'unsupported' first.")
        return

    n_sup = sum(1 for g in gold if g["gold"] == "supported")
    print(f"GOLD SET: {len(gold)} labeled rows "
          f"({n_sup} supported, {len(gold)-n_sup} unsupported)\n")

    for model_name in MODELS:
        print("=" * 80)
        print(f"MODEL: {model_name}")
        try:
            nli = NLI(model_name)
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        print(f"  id2label={nli.id2label}  entailment index={nli.e_idx}")

        for cname, pf, hf in CONFIGS:
            tp = fp = tn = fn = 0
            errors = []
            for g in gold:
                p = nli.p_entail(pf(g["msg"]), hf(g["rat"], g["pred"]))
                got = "supported" if p >= args.thresh else "unsupported"
                if g["gold"] == "supported" and got == "supported":   tp += 1
                elif g["gold"] == "supported" and got == "unsupported":
                    fn += 1; errors.append(("MISSED", p, g))
                elif g["gold"] == "unsupported" and got == "supported":
                    fp += 1; errors.append(("FALSE+", p, g))
                else: tn += 1
            tot = tp + fp + tn + fn
            acc = (tp + tn) / tot if tot else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            print(f"\n  {cname:38s} acc={acc:.2f} prec={prec:.2f} rec={rec:.2f} "
                  f"(tp={tp} fp={fp} tn={tn} fn={fn})")
            for tag, p, g in errors[:4]:
                print(f"      {tag} p={p:.3f} [{g['bucket']}] {g['msg'][:52]}")
        print()


if __name__ == "__main__":
    main()
