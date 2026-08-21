"""
analyze_explanation.py
----------------------
DIMENSION 2 — Conditional Explanation Reliability.

Subset (the conditioning):
    instances the model classifies CORRECTLY on BOTH the clean and the perturbed input.
    This isolates explanation-level failure from prediction-level failure: if the label
    itself flipped, a changed rationale would be expected, so those cases are excluded.

Two metrics, both computed within that subset:

  (1) STABILITY      token-Jaccard(clean rationale, perturbed rationale).
                     Higher = the justification survives perturbation.
                     Deterministic, model-free.

  (2) LABEL-CONSISTENCY   is the model's rationale SUPPORTED BY THE MESSAGE, given the
                     verdict the model itself reached?
                        premise    = the message the model saw
                        hypothesis = rationale + explicit verdict sentence
                     Model: cross-encoder/nli-deberta-v3-large.
                     roberta-large-mnli was tried first and is UNUSABLE here: on identical
                     inputs it returns NEUTRAL ~0.99 where deberta returns ENTAILMENT ~0.99,
                     and it scored near-zero entailment on rationales that plainly support
                     their labels. Entailment index is read from config (roberta=2,
                     deberta=1) — hardcoding it silently corrupts the metric.
                     An automatic PROXY, not a ground-truth faithfulness measure.

REMOVED — grounding / unsupported-cue:
    We attempted to verify automatically whether a cited cue is actually present in the
    message. Two approaches were tested and both proved unreliable:
      * lexical matching  — depends entirely on hand-constructed keyword lists; the
                            reported rate moves with arbitrary vocabulary choices.
      * NLI entailment    — fails on short SMS: on a hand-judged urgency gold set it
                            reached recall 0.27 (vs 0.73 lexical), detecting only the
                            literal token "urgent" and missing account-suspension /
                            deadline framing entirely.
    Robust automatic grounding verification of free-text rationales is an open problem;
    we report this as a negative result and leave it to future work. Dimension 2 is
    therefore measured by stability and label-consistency.

Legacy note: template names containing "cot" are EXPLANATION-REQUIRED prompts, not
chain-of-thought reasoning.

Usage:
    python analyze_explanation.py \
        --preds outputs/expl_all.jsonl \
        --out-dir outputs/analysis_expl \
        --nli
"""
import argparse, json, os, re, sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# rationale extraction & normalization
# ---------------------------------------------------------------------------

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)

def rationale_text(rec):
    """Return the explanation portion (strip the FINAL: tag line)."""
    raw = rec.get("rationale") or rec.get("raw") or ""
    # cut everything from the FINAL: tag onward -> the rationale is what precedes it
    m = _FINAL_RE.search(raw)
    return (raw[:m.start()] if m else raw).strip()


def normalize(txt):
    return re.sub(r"\s+", " ", txt.lower()).strip()


def norm_label(x):
    """Canonicalize labels so 0/1, 'benign', 'phishing', case variants all map to
    'smishing'/'ham'. 'spam' is deliberately NOT mapped (it can include non-phishing
    promotional messages)."""
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x


# ---------------------------------------------------------------------------
# (1) STABILITY — similarity between clean and perturbed rationales
# ---------------------------------------------------------------------------

def jaccard(a, b):
    """Cheap lexical stability: token Jaccard. (A sentence-embedding cosine can be
    swapped in for a stronger measure; Jaccard needs no model and is deterministic.)"""
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# (2) LABEL-CONSISTENCY — NLI entailment
#
# MODEL CHOICE MATTERS ENORMOUSLY HERE.
#   roberta-large-mnli is UNUSABLE on this text. On identical inputs it returns
#   NEUTRAL with ~0.99 confidence where deberta-v3 returns ENTAILMENT with ~0.99.
#   Against hand-checked rationales that plainly support their labels, roberta gave
#   entailment probabilities of 0.001-0.05 — i.e. it says "not entailed" for
#   everything. Its scores tracked rhetorical style, not rationale quality.
#   We therefore use cross-encoder/nli-deberta-v3-large.
#
# LABEL ORDER DIFFERS BETWEEN MODELS — never hardcode the index:
#   roberta-large-mnli : {0: contradiction, 1: neutral,     2: entailment}
#   deberta-v3-large   : {0: contradiction, 1: entailment,  2: neutral}
#   We read id2label from the model config.
#
# HYPOTHESIS CONSTRUCTION. The rationale ALONE is not a judgeable hypothesis: it
# states evidence, not a conclusion, and NLI will not make the leap to a categorical
# verdict (rationale-only scored 2/6 on probe cases). Appending an explicit verdict
# sentence makes the hypothesis a complete proposition (evidence + conclusion) and
# scored 5/6 with p≈0.98-0.99. A verdict-ONLY hypothesis (no rationale) scored 2/6
# with p≈0.000 — confirming the model is genuinely reading the rationale and not just
# the label. This remains an automatic PROXY, not a ground-truth faithfulness measure.
# ---------------------------------------------------------------------------

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"


def verdict_sentence(label):
    """The explicit conclusion appended to the rationale to form the hypothesis."""
    if label == "smishing":
        return "This message is classified as a phishing or scam message."
    return "This message is classified as benign. It is not a phishing attempt or a scam."


class NLI:
    def __init__(self, model_name=NLI_MODEL, device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval(); self.device = device
        # read the entailment index from config — do NOT hardcode
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in self.id2label.items() if "entail" in l)
        print(f"[nli] {model_name} id2label={self.id2label} "
              f"entail_idx={self.entail_idx}", file=sys.stderr)

    def entail_prob(self, premise, hypothesis):
        import torch
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=512).to(self.device)   # 512: rationales run 500-800 chars
        with torch.no_grad():
            probs = self.model(**x).logits[0].softmax(-1)
        return float(probs[self.entail_idx])

    def entails(self, premise, hypothesis, thresh=0.5):
        return self.entail_prob(premise, hypothesis) >= thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", help="combined jsonl with clean+perturbed explanation rows")
    ap.add_argument("--messages", help="jsonl mapping id->message text (perturbed variants)")
    ap.add_argument("--out-dir", default="outputs/analysis_expl")
    ap.add_argument("--nli", action="store_true", help="run NLI label-consistency (needs GPU + model)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.preds, encoding="utf-8")]
    # keep only explanation templates
    rows = [r for r in rows if "cot" in r["template"] or "expl" in r["template"]]
    print(f"[data] {len(rows)} explanation-template rows", file=sys.stderr)

    # message lookup: perturbed variant text keyed by (id, attack, seed) if provided
    msg_lookup = {}
    if args.messages:
        for l in open(args.messages, encoding="utf-8"):
            d = json.loads(l)
            msg_lookup[(d["id"], d["attack_class"], d["seed"])] = d["perturbed_message"]

    # index clean rationales by (model, template, id)
    clean = {}
    for r in rows:
        if r["condition"] == "clean":
            clean[(r["model"], r["template"], r["id"])] = r

    if args.nli and not args.messages:
        sys.exit("--nli requires --messages: the NLI premise is the message text.")
    nli = NLI(device=args.device) if args.nli else None

    # Accumulate per (model, template) within the both-correct subset.
    # NOTE: the unsupported-cue / grounding sub-metric was REMOVED. Automatic
    # verification of whether a free-text rationale is grounded in message evidence
    # proved unreliable: lexical matching depends on hand-constructed word lists, and
    # NLI-based verification failed badly on short SMS (recall 0.27 against a
    # hand-judged urgency set, vs 0.73 for lexical). We report that as a negative
    # result and leave robust grounding verification to future work. Dimension 2 is
    # therefore measured by STABILITY and LABEL-CONSISTENCY, both of which are
    # model-free / standard and do not depend on tuned vocabularies.
    agg = defaultdict(lambda: {"n": 0, "stability": 0.0,
                               "labelcons": 0.0, "labelcons_n": 0})

    for r in rows:
        if r["condition"] != "perturbed":
            continue
        ck = (r["model"], r["template"], r["id"])
        c = clean.get(ck)
        if c is None:
            continue
        # ---- THE SUBSET: both clean and perturbed correct ----
        if not (norm_label(c["pred"]) == norm_label(c["true_label"]) and
                norm_label(r["pred"]) == norm_label(r["true_label"])):
            continue

        key = (r["model"], r["template"])      # per adviser: not model alone
        a = agg[key]
        a["n"] += 1

        rc, rp = rationale_text(c), rationale_text(r)

        # (1) STABILITY — does the rationale survive perturbation?
        a["stability"] += jaccard(rc, rp)

        # (2) LABEL-CONSISTENCY — is the rationale supported by the message,
        #     given the verdict the model itself reached?
        #     premise    = the message the model actually saw (perturbed)
        #     hypothesis = rationale + explicit verdict sentence
        #     This is the direction that works: message -> (evidence + conclusion).
        if nli is not None:
            msg = msg_lookup.get((r["id"], r["attack_class"], r["seed"]))
            if msg:
                hyp = f"{rp} {verdict_sentence(norm_label(r['pred']))}"
                a["labelcons"] += nli.entails(msg, hyp)
                a["labelcons_n"] += 1

    # ---- report ----
    out = os.path.join(args.out_dir, "explanation_reliability.txt")
    with open(out, "w") as f:
        f.write("=== Conditional Explanation Reliability ===\n")
        f.write("Subset: instances the model classifies CORRECTLY on BOTH the clean and\n")
        f.write("the perturbed input. This isolates explanation-level failure from\n")
        f.write("prediction-level failure.\n\n")
        f.write("stability  = token-Jaccard(clean rationale, perturbed rationale); higher=more stable\n")
        f.write("labelcons  = fraction of cases where the MESSAGE entails (rationale + the model's\n")
        f.write("             own verdict), via deberta-v3 NLI. Automatic proxy, not ground truth.\n")
        f.write("Grounding/unsupported-cue is NOT reported: automatic verification proved\n")
        f.write("unreliable (see paper limitations).\n\n")

        f.write(f"{'model':32s} {'template':26s} {'n':>7s} {'stability':>10s} {'labelcons':>10s}\n")
        f.write("-" * 90 + "\n")
        for (model, tmpl), a in sorted(agg.items()):
            if a["n"] == 0:
                continue
            stab = a["stability"] / a["n"]
            lc = (a["labelcons"] / a["labelcons_n"]) if a["labelcons_n"] else float("nan")
            f.write(f"{model.split('/')[-1]:32s} {tmpl:26s} {a['n']:>7d} "
                    f"{stab:>10.3f} {lc:>10.3f}\n")

        # pooled per model (across templates)
        f.write("\n=== pooled per model (over templates) ===\n")
        f.write(f"{'model':32s} {'n':>8s} {'stability':>10s} {'labelcons':>10s}\n")
        f.write("-" * 64 + "\n")
        pooled = defaultdict(lambda: {"n": 0, "stab": 0.0, "lc": 0.0, "lc_n": 0})
        for (model, tmpl), a in agg.items():
            p = pooled[model]
            p["n"] += a["n"]
            p["stab"] += a["stability"]
            p["lc"] += a["labelcons"]
            p["lc_n"] += a["labelcons_n"]
        for model, p in sorted(pooled.items()):
            if p["n"] == 0:
                continue
            stab = p["stab"] / p["n"]
            lc = (p["lc"] / p["lc_n"]) if p["lc_n"] else float("nan")
            f.write(f"{model.split('/')[-1]:32s} {p['n']:>8d} {stab:>10.3f} {lc:>10.3f}\n")

    print(open(out).read())
    print(f"[write] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
