"""
analyze_explanation.py
----------------------
DIMENSION 2 — EXPLANATION STABILITY.

NAMING (adviser, accepted). Previously called "explanation reliability" — an overclaim.
We measure the TEXTUAL SIMILARITY of generated rationales, not their faithfulness,
correctness, or grounding. A rationale can be perfectly stable and still be wrong.
The dimension is therefore EXPLANATION STABILITY.

WHAT IS MEASURED
    For each (model, template, message, attack, seed) we compare the rationale produced
    on the CLEAN message with the rationale produced on the PERTURBED message, and report
    their textual similarity.

    We do this ONLY on pairs the model classifies correctly in BOTH forms. Phrasing
    matters: this CONTROLS FOR prediction correctness — it does not "separate"
    explanation failure from prediction failure. If the label itself flipped, a changed
    rationale would be expected, so those pairs are not comparable.

    We explicitly do NOT measure faithfulness, correctness, or grounding. Two attempts to
    do so (lexical cue-matching; NLI entailment) both failed validation and are reported
    as negative results. Most decisively: pairing a rationale with a RANDOM UNRELATED
    message produced the same NLI entailment score (0.890) as pairing it with its true
    message (0.892) — the NLI model never reads the premise.

METRICS
    PRIMARY    BERTScore F1  — semantic similarity, robust to paraphrase.
    SECONDARY  ROUGE-L       — longest-common-subsequence F-measure (ordered overlap).
    SECONDARY  token-Jaccard — lexical set overlap; model-free, deterministic.

    Reporting a semantic AND a lexical measure is deliberate: a model may restate the
    same justification in different words. BERTScore calls that stable; Jaccard does not.
    The gap between them is itself informative.

ELIGIBILITY (adviser: non-negotiable)
    We report, per model AND per attack class, how many clean/perturbed pairs were
    correctly classified in BOTH forms — as a count and as a RATE. Without this, a weak
    model can look explanation-stable simply because its eligible subset is small and
    consists of easy cases.

Legacy note: template names containing "cot" are EXPLANATION-REQUIRED prompts, not
chain-of-thought reasoning.

Usage:
    python analyze_explanation.py --preds outputs/expl_all.jsonl \
        --out-dir outputs/analysis_expl --bertscore
"""
import argparse, json, os, re, sys
from collections import defaultdict

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)\b", re.IGNORECASE)


def rationale_text(rec):
    """Rationale = everything BEFORE the FINAL: tag (the tag is the label, not the
    justification)."""
    raw = rec.get("rationale") or rec.get("raw") or ""
    m = _FINAL_RE.search(raw)
    return (raw[:m.start()] if m else raw).strip()


def normalize(txt):
    return re.sub(r"\s+", " ", txt.lower()).strip()


def norm_label(x):
    """'spam' deliberately NOT mapped to smishing — it can include non-phishing promo."""
    x = str(x).strip().lower()
    if x in {"1", "smish", "smishing", "phishing", "phish"}:
        return "smishing"
    if x in {"0", "ham", "benign", "legitimate", "legit"}:
        return "ham"
    return x


def jaccard(a, b):
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def rouge_l(a, b):
    """ROUGE-L F-measure via LCS. Captures ordered overlap Jaccard (a bag) misses."""
    ta, tb = normalize(a).split(), normalize(b).split()
    if not ta or not tb:
        return 0.0
    prev = [0] * (len(tb) + 1)
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1)
        ai = ta[i-1]
        for j in range(1, len(tb) + 1):
            cur[j] = prev[j-1] + 1 if ai == tb[j-1] else max(prev[j], cur[j-1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(tb), lcs / len(ta)
    return 2 * p * r / (p + r)



# ---------------------------------------------------------------------------
# BERTScore — implemented directly.
#
# The `bert-score` pip package (0.3.13) is INCOMPATIBLE with current transformers:
#   "RobertaTokenizer has no attribute build_inputs_with_special_tokens"
# Rather than downgrade transformers (which the whole inference pipeline depends on),
# we compute BERTScore ourselves. It is not complicated:
#   1. embed every token of both texts with a contextual encoder (roberta-large)
#   2. cosine-similarity matrix between the two token sets
#   3. greedy matching: recall = mean over reference tokens of their best match;
#      precision = mean over candidate tokens of their best match
#   4. F1 = harmonic mean
# Special tokens are excluded; padding is masked out.
# ---------------------------------------------------------------------------

def compute_bertscore(refs, cands, model_name="roberta-large", device="cuda",
                      batch_size=64, layer=17):
    """BERTScore F1 for aligned lists of texts. `layer` 17 of roberta-large is the
    layer the original BERTScore paper found best-correlated with human judgment."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    mdl.eval()
    print(f"[bertscore] {len(refs)} pairs | {model_name} layer={layer}", file=sys.stderr)

    def embed(texts):
        """-> list of (n_tokens, dim) tensors, special tokens & padding removed."""
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(device)
            with torch.no_grad():
                hs = mdl(**enc).hidden_states[layer]        # (B, T, H)
            hs = torch.nn.functional.normalize(hs, dim=-1)  # cosine == dot product
            mask = enc["attention_mask"].bool()
            special = torch.zeros_like(mask)
            for sid in tok.all_special_ids:
                special |= (enc["input_ids"] == sid)
            keep = mask & ~special
            for b in range(hs.size(0)):
                out.append(hs[b][keep[b]])
            if (i // batch_size) % 20 == 0:
                print(f"  [bertscore] embedded {min(i+batch_size, len(texts))}/{len(texts)}",
                      file=sys.stderr)
        return out

    R = embed(refs)
    C = embed(cands)

    f1s = []
    for r, c in zip(R, C):
        if r.numel() == 0 or c.numel() == 0:
            f1s.append(0.0)
            continue
        sim = c @ r.T                       # (n_cand, n_ref) cosine matrix
        precision = sim.max(dim=1).values.mean().item()   # each cand token -> best ref
        recall = sim.max(dim=0).values.mean().item()      # each ref token  -> best cand
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return f1s


# ---------------------------------------------------------------------------
# Sentence-embedding cosine — a second semantic measure, using all-MiniLM-L6-v2,
# the SAME encoder as the perturbation similarity gate. Cheap, already cached,
# and keeps the semantic-similarity notion consistent across the pipeline.
# ---------------------------------------------------------------------------

def compute_sentence_cosine(a_list, b_list, device="cuda", batch_size=64,
                            model_name="sentence-transformers/all-MiniLM-L6-v2"):
    import torch
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    print(f"[sentcos] {len(a_list)} pairs | {model_name}", file=sys.stderr)

    def encode(texts):
        vecs = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(device)
            with torch.no_grad():
                hs = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)      # mean pooling
            vecs.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(vecs)

    A, B = encode(a_list), encode(b_list)
    return (A * B).sum(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--messages", help="(optional) not needed for stability")
    ap.add_argument("--out-dir", default="outputs/analysis_expl")
    ap.add_argument("--bertscore", action="store_true",
                    help="compute BERTScore F1 (PRIMARY). Needs GPU + `pip install bert-score`.")
    ap.add_argument("--bertscore-model", default="roberta-large")
    ap.add_argument("--sentcos", action="store_true",
                    help="also compute sentence-embedding cosine (all-MiniLM-L6-v2, the "
                         "same encoder as the perturbation similarity gate)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.preds, encoding="utf-8")]
    rows = [r for r in rows if "cot" in r["template"] or "expl" in r["template"]]
    print(f"[data] {len(rows)} explanation-template rows", file=sys.stderr)

    clean = {}
    for r in rows:
        if r["condition"] == "clean":
            clean[(r["model"], r["template"], r["id"])] = r

    pairs = []
    elig = defaultdict(lambda: [0, 0])        # (model, attack) -> [eligible, total]
    elig_t = defaultdict(lambda: [0, 0])      # (model, template) -> [eligible, total]

    for r in rows:
        if r["condition"] != "perturbed":
            continue
        c = clean.get((r["model"], r["template"], r["id"]))
        if c is None:
            continue
        model, tmpl, attack = r["model"], r["template"], r["attack_class"]
        elig[(model, attack)][1] += 1
        elig_t[(model, tmpl)][1] += 1
        if not (norm_label(c["pred"]) == norm_label(c["true_label"]) and
                norm_label(r["pred"]) == norm_label(r["true_label"])):
            continue
        elig[(model, attack)][0] += 1
        elig_t[(model, tmpl)][0] += 1
        pairs.append({"model": model, "template": tmpl, "attack": attack,
                      "clean_rat": rationale_text(c), "pert_rat": rationale_text(r)})

    print(f"[eligible] {len(pairs)} both-correct rationale pairs", file=sys.stderr)

    for p in pairs:
        p["jaccard"] = jaccard(p["clean_rat"], p["pert_rat"])
        p["rougeL"] = rouge_l(p["clean_rat"], p["pert_rat"])

    if args.bertscore and pairs:
        try:
            f1s = compute_bertscore([p["clean_rat"] for p in pairs],
                                    [p["pert_rat"] for p in pairs],
                                    model_name=args.bertscore_model,
                                    device=args.device, batch_size=args.batch_size)
            for p, f in zip(pairs, f1s):
                p["bertscore"] = f
        except Exception as e:
            print(f"[bertscore] failed: {e}", file=sys.stderr)

    if args.sentcos and pairs:
        try:
            cos = compute_sentence_cosine([p["clean_rat"] for p in pairs],
                                          [p["pert_rat"] for p in pairs],
                                          device=args.device, batch_size=args.batch_size)
            for p, c in zip(pairs, cos):
                p["sentcos"] = c
        except Exception as e:
            print(f"[sentcos] failed: {e}", file=sys.stderr)

    has_bs = any("bertscore" in p for p in pairs)
    has_sc = any("sentcos" in p for p in pairs)

    def agg(keyfn):
        d = defaultdict(lambda: defaultdict(list))
        for p in pairs:
            k = keyfn(p)
            for m in ("jaccard", "rougeL", "bertscore", "sentcos"):
                if m in p:
                    d[k][m].append(p[m])
        return d

    by_model = agg(lambda p: p["model"])
    by_mt = agg(lambda p: (p["model"], p["template"]))
    by_ma = agg(lambda p: (p["model"], p["attack"]))

    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    models = sorted({p["model"] for p in pairs})
    attacks = sorted({p["attack"] for p in pairs})
    templates = sorted({p["template"] for p in pairs})
    primary = "bertscore" if has_bs else "jaccard"

    out = os.path.join(args.out_dir, "explanation_stability.txt")
    with open(out, "w") as f:
        f.write("=" * 92 + "\n")
        f.write("DIMENSION 2 - EXPLANATION STABILITY\n")
        f.write("=" * 92 + "\n\n")
        f.write("Textual similarity between the rationale on the CLEAN message and the rationale\n")
        f.write("on the PERTURBED message. Computed ONLY on pairs the model classifies correctly\n")
        f.write("in BOTH forms, which CONTROLS FOR prediction correctness.\n\n")
        f.write("This measures STABILITY - not faithfulness, correctness, or grounding.\n")
        f.write("A rationale can be perfectly stable and still be wrong.\n\n")
        f.write("PRIMARY  : BERTScore F1 (semantic, paraphrase-robust)\n")
        f.write("SECONDARY: ROUGE-L (ordered overlap); token-Jaccard (lexical set overlap)\n\n")

        f.write("-" * 92 + "\n")
        f.write("(A) ELIGIBILITY - pairs correctly classified in BOTH clean and perturbed form.\n")
        f.write("    A model with a SMALL eligible set may look stable only because its eligible\n")
        f.write("    cases are the easy ones. Read table (B) against this.\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'model':28s} {'attack':12s} {'eligible':>10s} {'total':>10s} {'rate':>8s}\n")
        for m in models:
            for a in attacks:
                e, t = elig[(m, a)]
                f.write(f"{m.split('/')[-1]:28s} {str(a):12s} {e:>10d} {t:>10d} "
                        f"{(f'{100*e/t:.1f}%' if t else 'n/a'):>8s}\n")
            e = sum(elig[(m, a)][0] for a in attacks)
            t = sum(elig[(m, a)][1] for a in attacks)
            f.write(f"{m.split('/')[-1]:28s} {'ALL':12s} {e:>10d} {t:>10d} "
                    f"{(f'{100*e/t:.1f}%' if t else 'n/a'):>8s}\n\n")

        f.write("-" * 92 + "\n")
        f.write("(B) STABILITY by model (pooled over templates and attacks)\n")
        f.write("-" * 92 + "\n")
        h = f"{'model':28s} {'n':>8s}"
        if has_bs:
            h += f" {'BERTScore':>10s}"
        if has_sc:
            h += f" {'SentCos':>9s}"
        h += f" {'ROUGE-L':>9s} {'Jaccard':>9s}"
        f.write(h + "\n")
        for m in models:
            d = by_model[m]
            line = f"{m.split('/')[-1]:28s} {len(d['jaccard']):>8d}"
            if has_bs:
                line += f" {mean(d['bertscore']):>10.3f}"
            if has_sc:
                line += f" {mean(d['sentcos']):>9.3f}"
            line += f" {mean(d['rougeL']):>9.3f} {mean(d['jaccard']):>9.3f}"
            f.write(line + "\n")

        f.write("\n" + "-" * 92 + "\n")
        f.write(f"(C) STABILITY by model x attack   (values = "
                f"{'BERTScore F1' if has_bs else 'token-Jaccard'})\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'model':28s}" + "".join(f"{a:>12s}" for a in attacks) + "\n")
        for m in models:
            line = f"{m.split('/')[-1]:28s}"
            for a in attacks:
                v = by_ma[(m, a)][primary]
                line += f"{mean(v):>12.3f}" if v else f"{'n/a':>12s}"
            f.write(line + "\n")

        f.write("\n" + "-" * 92 + "\n")
        f.write(f"(D) STABILITY by model x template (does role framing shift the rationale?)\n")
        f.write("-" * 92 + "\n")
        f.write(f"{'model':28s}" + "".join(f"{t:>26s}" for t in templates) + "\n")
        for m in models:
            line = f"{m.split('/')[-1]:28s}"
            for t in templates:
                v = by_mt[(m, t)][primary]
                line += f"{mean(v):>26.3f}" if v else f"{'n/a':>26s}"
            f.write(line + "\n")

    print(open(out).read())
    print(f"[write] {out}", file=sys.stderr)

    with open(os.path.join(args.out_dir, "stability_metrics.json"), "w") as f:
        json.dump({
            "primary_metric": "bertscore_f1" if has_bs else "token_jaccard",
            "eligibility": {f"{m}|{a}": elig[(m, a)] for (m, a) in elig},
            "by_model": {m: {k: mean(v) for k, v in d.items()} for m, d in by_model.items()},
            "by_model_attack": {f"{m}|{a}": {k: mean(v) for k, v in d.items()}
                                for (m, a), d in by_ma.items()},
            "by_model_template": {f"{m}|{t}": {k: mean(v) for k, v in d.items()}
                                  for (m, t), d in by_mt.items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()
