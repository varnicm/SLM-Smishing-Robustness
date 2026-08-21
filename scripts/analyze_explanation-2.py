"""
analyze_explanation.py  (Dimension 2 — Conditional Explanation Reliability)
---------------------------------------------------------------------------
Implements the adviser-approved method:

  Subset:  clean_pred == true_label  AND  perturbed_pred == true_label
  Within that subset, measure whether the generated rationale:
    (1) STABILITY        — changes between clean and perturbed (it shouldn't much)
    (2) LABEL-CONSISTENCY — the rationale entails / supports its own label  (NLI)
    (3) UNSUPPORTED-CUE   — cites cues NOT present in the message           (grounding)

  This does NOT claim to recover internal reasoning; it evaluates observable
  justification behavior under perturbation.

Requires: predictions that store the FULL rationale text (from the CoT/explanation
re-run with an enlarged cap). Each record needs at least:
    model, template, condition (clean|perturbed), id, attack_class, seed,
    true_label, pred, rationale (or raw), and for perturbed rows the message text.

Usage:
    python analyze_explanation.py \
        --clean   outputs/expl_clean.jsonl \
        --perturbed outputs/expl_perturbed.jsonl \
        --messages outputs/smishtank_perturbed.jsonl \
        --out-dir outputs/analysis_expl
    # (or a single combined --preds file that has both conditions)

NLI is optional: with --nli it loads an entailment model; without it, label-
consistency is skipped and the other two metrics still run.
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
# (3) UNSUPPORTED-CUE — do rationale cue-terms actually appear in the message?
# ---------------------------------------------------------------------------

# canonical cue vocabulary the prompt asked about, with surface synonyms
CUE_TERMS = {
    "urgency":       ["urgent", "immediately", "now", "asap", "expire", "within", "hurry", "act fast", "24 hours", "limited"],
    "link":          ["link", "url", "http", "click", "visit", "website", "bit.ly", ".com", ".in", "tap"],
    "impersonation": ["bank", "amazon", "paypal", "usps", "irs", "netflix", "apple", "official", "account", "verify", "brand"],
    "personal_info": ["ssn", "password", "pin", "login", "credential", "personal", "confirm your", "identity", "card number"],
    "payment":       ["payment", "pay", "fee", "$", "refund", "prize", "gift card", "won", "claim", "deposit", "balance"],
}

def unsupported_cues(rationale, message):
    """Fraction of cue-categories the rationale CLAIMS that are NOT evidenced in the
    message. Returns (n_claimed, n_unsupported)."""
    r, m = normalize(rationale), normalize(message)
    claimed = []
    for cue, terms in CUE_TERMS.items():
        # rationale claims this cue if it names the category or a strong synonym
        if cue.replace("_", " ") in r or any(t in r for t in terms[:3]):
            claimed.append(cue)
    unsupported = 0
    for cue in claimed:
        terms = CUE_TERMS[cue]
        if not any(t in m for t in terms):   # message shows no evidence of the cue
            unsupported += 1
    return len(claimed), unsupported


# ---------------------------------------------------------------------------
# (2) LABEL-CONSISTENCY — NLI entailment (optional)
# ---------------------------------------------------------------------------

class NLI:
    def __init__(self, model_name="roberta-large-mnli", device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval(); self.device = device
        # roberta-large-mnli labels: 0=contradiction,1=neutral,2=entailment
        self.entail_idx = 2

    def entails(self, premise, hypothesis):
        import torch
        x = self.tok(premise, hypothesis, return_tensors="pt", truncation=True,
                     max_length=256).to(self.device)
        with torch.no_grad():
            logits = self.model(**x).logits[0]
            probs = logits.softmax(-1)
        return float(probs[self.entail_idx])


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

    nli = NLI(device=args.device) if args.nli else None

    # accumulate metrics per model (within the both-correct subset)
    agg = defaultdict(lambda: {"n": 0, "stability": 0.0,
                               "labelcons": 0.0, "labelcons_n": 0,
                               "claimed": 0, "unsupported": 0})

    for r in rows:
        if r["condition"] != "perturbed":
            continue
        ck = (r["model"], r["template"], r["id"])
        c = clean.get(ck)
        if c is None:
            continue
        # ---- THE SUBSET: both clean and perturbed correct ----
        if not (c["pred"] == c["true_label"] and r["pred"] == r["true_label"]):
            continue

        model = r["model"]
        a = agg[model]
        a["n"] += 1

        rc, rp = rationale_text(c), rationale_text(r)

        # (1) stability
        a["stability"] += jaccard(rc, rp)

        # (3) unsupported cues — evaluated on the PERTURBED rationale vs perturbed msg
        msg = msg_lookup.get((r["id"], r["attack_class"], r["seed"]), "")
        if msg:
            nc, nu = unsupported_cues(rp, msg)
            a["claimed"] += nc
            a["unsupported"] += nu

        # (2) label-consistency via NLI (rationale entails label?)
        if nli is not None:
            hyp = f"This message is {r['true_label']}."
            a["labelcons"] += nli.entails(rp, hyp)
            a["labelcons_n"] += 1

    # ---- report ----
    out = os.path.join(args.out_dir, "explanation_reliability.txt")
    with open(out, "w") as f:
        f.write("=== Conditional Explanation Reliability (both clean & perturbed correct) ===\n\n")
        f.write(f"{'model':40s} {'n':>7s} {'stability':>10s} {'labelcons':>10s} {'unsup_cue%':>10s}\n")
        f.write("-" * 82 + "\n")
        for model, a in sorted(agg.items()):
            if a["n"] == 0:
                f.write(f"{model.split('/')[-1]:40s} {'--- no both-correct instances ---':>40s}\n")
                continue
            stab = a["stability"] / a["n"]
            lc = (a["labelcons"] / a["labelcons_n"]) if a["labelcons_n"] else float("nan")
            uc = (100 * a["unsupported"] / a["claimed"]) if a["claimed"] else 0.0
            f.write(f"{model.split('/')[-1]:40s} {a['n']:>7d} {stab:>10.3f} "
                    f"{lc:>10.3f} {uc:>9.1f}%\n")
    print(open(out).read())
    print(f"[write] {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
