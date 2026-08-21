"""Same-label baseline control for the rationale-similarity (Dimension 2) metric.

Question it answers
-------------------
Is the high clean-vs-perturbed rationale similarity evidence of message-specific
justification, or just label-conditioned boilerplate? If *unrelated* rationales
that merely share a predicted label score nearly as high, the similarity is
boilerplate and must not be read as explanation robustness.

What this script computes (the baseline population, B)
------------------------------------------------------
Using ONLY nonempty rationales, it pairs each rationale with one from a
DIFFERENT message that shares the SAME model, SAME template, and SAME predicted
label, then scores the pair with BERTScore F1. It reports the mean per
(model, template) and pooled, with the number of pairs.

Compare the pooled mean here against the true-pair mean from
analyze_explanation.py (clean vs its own perturbed rationale). In our run the two
were ~0.77 (this baseline) vs ~0.81 (true pairs) — a gap of only ~0.04, i.e. the
similarity is largely label-conditioned boilerplate.

Why nonempty matters
--------------------
Empty rationales (a bare `FINAL:` label with no explanation) otherwise score a
spurious BERTScore ~1.0 against each other and inflate everything — this is the
empty-rationale artifact that contaminated one model's metrics. We drop any
rationale shorter than --min-chars (default 10, matching the paper's near-empty
exclusion) before pairing.

Input
-----
The explanations JSONL (default outputs/explanations/expl_all.jsonl), one row per
(model, template, message, condition) with at least:
    model, template, id, condition, pred, raw
`raw` is the full generation; the rationale is the text BEFORE the `FINAL:` tag.

Usage
-----
    python tests/test_bertscore_baseline.py \
        --preds outputs/explanations/expl_all.jsonl \
        --condition clean --min-chars 10 --seed 0 \
        --out outputs/analysis/bertscore_baseline.json

Notes
-----
* BERTScore is implemented in-house (roberta-large, hidden layer 17, greedy
  token matching, F1) to stay consistent with analyze_explanation.py and to
  avoid the broken `bert-score` PyPI package. Factor this into a shared module
  if you prefer a single source of truth.
* Runs on GPU if available, else CPU (slower). Token embeddings for the unique
  rationales are cached once, then pairs are scored from the cache.
"""

import argparse
import json
import random
import re
from collections import defaultdict

import torch

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# I/O + rationale extraction
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_rationale(raw: str) -> str:
    """Return the rationale text: everything before the FINAL: tag, stripped.

    A bare label ('FINAL: smishing') yields an empty string.
    """
    if not raw:
        return ""
    m = _FINAL_RE.search(raw)
    text = raw[: m.start()] if m else raw
    return text.strip()


def load_rationales(path, condition, min_chars):
    """Return {(model, template, pred_label): {msg_id: rationale}}.

    Keeps one rationale per (model, template, pred_label, message). Only
    condition-matching rows (default 'clean') and rationales with at least
    `min_chars` characters are retained.
    """
    groups = defaultdict(dict)
    kept = dropped_empty = skipped_cond = 0
    for row in read_jsonl(path):
        if condition and row.get("condition") != condition:
            skipped_cond += 1
            continue
        pred = (row.get("pred") or "").strip().lower()
        if pred not in ("smishing", "ham"):
            continue  # unparsed / missing label -> not usable for a same-label control
        rat = extract_rationale(row.get("raw", ""))
        if len(rat) < min_chars:
            dropped_empty += 1
            continue
        key = (row["model"], row["template"], pred)
        # first rationale wins if a message somehow appears twice for the key
        groups[key].setdefault(row["id"], rat)
        kept += 1
    print(f"[load] kept={kept}  dropped_empty(<{min_chars} chars)={dropped_empty}"
          f"  skipped_wrong_condition={skipped_cond}")
    return groups


# --------------------------------------------------------------------------- #
# In-house BERTScore (roberta-large, layer 17, greedy matching, F1)
# --------------------------------------------------------------------------- #
class BERTScorer:
    def __init__(self, model_name="roberta-large", layer=17, device=None,
                 max_length=256, batch_size=64):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.max_length = max_length
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts):
        """Return {text: (T, H) float tensor of L2-normalized token embeddings}."""
        cache = {}
        uniq = list(dict.fromkeys(texts))  # de-dup, preserve order
        for i in range(0, len(uniq), self.batch_size):
            chunk = uniq[i : i + self.batch_size]
            enc = self.tok(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            hs = self.model(**enc).hidden_states[self.layer]  # (B, T, H)
            hs = torch.nn.functional.normalize(hs, dim=-1)
            mask = enc["attention_mask"].bool()
            special = self._special_mask(enc["input_ids"])
            keep = mask & ~special
            for j, text in enumerate(chunk):
                cache[text] = hs[j][keep[j]].to("cpu")  # (T_j, H) content tokens only
        return cache

    def _special_mask(self, input_ids):
        special_ids = torch.tensor(self.tok.all_special_ids, device=input_ids.device)
        return torch.isin(input_ids, special_ids)

    @staticmethod
    def f1(cand, ref):
        """Greedy-matched BERTScore F1 between two (T, H) normalized tensors."""
        if cand.numel() == 0 or ref.numel() == 0:
            return float("nan")
        sim = cand @ ref.T                       # (Tc, Tr) cosine (already normalized)
        precision = sim.max(dim=1).values.mean() # each cand token -> best ref token
        recall = sim.max(dim=0).values.mean()    # each ref token -> best cand token
        if precision + recall == 0:
            return 0.0
        return float((2 * precision * recall) / (precision + recall))


# --------------------------------------------------------------------------- #
# Pairing + scoring
# --------------------------------------------------------------------------- #
def build_pairs(groups, rng):
    """For each rationale, sample a partner from a DIFFERENT message in the same
    (model, template, pred_label) group. Returns a list of dicts.
    """
    pairs = []
    for (model, template, pred), by_msg in groups.items():
        msg_ids = list(by_msg.keys())
        if len(msg_ids) < 2:
            continue  # cannot pair within this group
        for mid in msg_ids:
            others = [o for o in msg_ids if o != mid]
            partner = rng.choice(others)
            pairs.append({
                "model": model, "template": template, "pred": pred,
                "id_a": mid, "id_b": partner,
                "text_a": by_msg[mid], "text_b": by_msg[partner],
            })
    return pairs


def summarize(scored):
    """Mean BERTScore F1 per (model, template) and pooled."""
    per_key = defaultdict(list)
    pooled = []
    for p in scored:
        if p["bertscore_f1"] == p["bertscore_f1"]:  # not NaN
            per_key[(p["model"], p["template"])].append(p["bertscore_f1"])
            pooled.append(p["bertscore_f1"])
    out = {"per_model_template": {}, "pooled": None, "n_pairs": len(pooled)}
    for (model, template), vals in sorted(per_key.items()):
        out["per_model_template"][f"{model}|{template}"] = {
            "mean_bertscore_f1": round(sum(vals) / len(vals), 4),
            "n_pairs": len(vals),
        }
    if pooled:
        out["pooled"] = round(sum(pooled) / len(pooled), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", default="outputs/explanations/expl_all.jsonl",
                    help="explanations JSONL with model, template, id, condition, pred, raw")
    ap.add_argument("--condition", default="clean",
                    help="which rows to pool rationales from (default: clean); "
                         "pass '' to use all conditions")
    ap.add_argument("--min-chars", type=int, default=10,
                    help="drop rationales shorter than this (default 10 = paper's "
                         "near-empty threshold; use 1 for strictly nonempty)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for partner sampling")
    ap.add_argument("--model-name", default="roberta-large")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="outputs/analysis/bertscore_baseline.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    groups = load_rationales(args.preds, args.condition or None, args.min_chars)
    if not groups:
        raise SystemExit("No usable rationales found — check --preds/--condition/--min-chars.")

    pairs = build_pairs(groups, rng)
    print(f"[pairs] built {len(pairs)} same-label / different-message pairs")

    scorer = BERTScorer(args.model_name, args.layer, batch_size=args.batch_size)
    all_texts = [p["text_a"] for p in pairs] + [p["text_b"] for p in pairs]
    cache = scorer.embed(all_texts)
    for p in pairs:
        p["bertscore_f1"] = BERTScorer.f1(cache[p["text_a"]], cache[p["text_b"]])

    summary = summarize(pairs)
    summary["config"] = {
        "preds": args.preds, "condition": args.condition, "min_chars": args.min_chars,
        "seed": args.seed, "model_name": args.model_name, "layer": args.layer,
        "pairing": "same model + same template + same predicted label, different message",
    }

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== Same-label baseline (BERTScore F1) ===")
    for k, v in summary["per_model_template"].items():
        print(f"  {k:45s}  {v['mean_bertscore_f1']:.4f}  (n={v['n_pairs']})")
    print(f"\n  POOLED: {summary['pooled']}  over {summary['n_pairs']} pairs")
    print(f"  Compare to the true-pair (clean vs perturbed) mean from "
          f"analyze_explanation.py; a small gap => label-conditioned boilerplate.")
    print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
