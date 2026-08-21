"""Shared rationale-similarity metrics (single source of truth for Dimension 2).

Imported by:
  * scripts/analyze_explanation.py   (true-pair stability + same-label baseline)
  * tests/test_bertscore_baseline.py (the baseline control, standalone)

Keeping one implementation here guarantees the integrated report and the
standalone control score rationales identically.

Metrics
-------
  BERTScore F1  : roberta-large, hidden layer 17, greedy token matching (primary).
                  In-house because the PyPI `bert-score` breaks on recent
                  transformers.
  SentCos       : cosine of all-MiniLM-L6-v2 sentence embeddings (same encoder as
                  the perturbation similarity gate).
  ROUGE-L F1    : lexical, order-sensitive.
  token-Jaccard : lexical, order-free. NOTE: two empty token sets are treated as
                  similarity 0.0 here (NOT 1.0) — the 1.0 convention is exactly
                  the empty-rationale artifact, and callers must exclude empty
                  rationales before scoring anyway.
"""

import json
import re

_FINAL_RE = re.compile(r"FINAL:\s*(smishing|ham)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


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
    """The rationale = text before the FINAL: tag, stripped. Bare label -> ''."""
    if not raw:
        return ""
    m = _FINAL_RE.search(raw)
    return (raw[: m.start()] if m else raw).strip()


# --------------------------------------------------------------------------- #
# Lexical metrics (no model)
# --------------------------------------------------------------------------- #
def _tokens(text):
    return _TOKEN_RE.findall((text or "").lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa and not sb:
        return 0.0  # deliberately NOT 1.0 — empties must be excluded upstream
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class RougeL:
    def __init__(self):
        from rouge_score import rouge_scorer
        self._s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def f1(self, a: str, b: str) -> float:
        return float(self._s.score(a or "", b or "")["rougeL"].fmeasure)


# --------------------------------------------------------------------------- #
# BERTScore (roberta-large, layer 17, greedy matching, F1)
# --------------------------------------------------------------------------- #
class BERTScorer:
    def __init__(self, model_name="roberta-large", layer=17, device=None,
                 max_length=256, batch_size=64):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.max_length = max_length
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(self.device).eval()
        self._cache = {}

    def warm(self, texts):
        """Embed and cache token vectors (content tokens only) for `texts`."""
        torch = self._torch
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        specials = torch.tensor(self.tok.all_special_ids)
        with torch.no_grad():
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i : i + self.batch_size]
                enc = self.tok(chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_length).to(self.device)
                hs = self.model(**enc).hidden_states[self.layer]
                hs = torch.nn.functional.normalize(hs, dim=-1)
                mask = enc["attention_mask"].bool()
                special = torch.isin(enc["input_ids"], specials.to(self.device))
                keep = mask & ~special
                for j, text in enumerate(chunk):
                    self._cache[text] = hs[j][keep[j]].to("cpu")

    def f1(self, a: str, b: str) -> float:
        ca, cb = self._cache.get(a), self._cache.get(b)
        if ca is None or cb is None:
            self.warm([a, b])
            ca, cb = self._cache[a], self._cache[b]
        if ca.numel() == 0 or cb.numel() == 0:
            return float("nan")
        sim = ca @ cb.T
        p = sim.max(dim=1).values.mean()
        r = sim.max(dim=0).values.mean()
        if float(p + r) == 0.0:
            return 0.0
        return float((2 * p * r) / (p + r))


# --------------------------------------------------------------------------- #
# SentCos (all-MiniLM-L6-v2 sentence embeddings)
# --------------------------------------------------------------------------- #
class SentCos:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self._cache = {}

    def warm(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if not todo:
            return
        embs = self.model.encode(todo, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        for t, e in zip(todo, embs):
            self._cache[t] = e

    def cos(self, a: str, b: str) -> float:
        if a not in self._cache or b not in self._cache:
            self.warm([a, b])
        return float((self._cache[a] * self._cache[b]).sum())


# --------------------------------------------------------------------------- #
# Convenience: score a list of {text_a, text_b, ...} pairs with selected metrics
# --------------------------------------------------------------------------- #
def score_pairs(pairs, bertscorer=None, sentcos=None, rougel=None, jaccard_on=True):
    """Annotate each pair dict in place with the requested metric values."""
    if bertscorer is not None:
        bertscorer.warm([p["text_a"] for p in pairs] + [p["text_b"] for p in pairs])
    if sentcos is not None:
        sentcos.warm([p["text_a"] for p in pairs] + [p["text_b"] for p in pairs])
    for p in pairs:
        if bertscorer is not None:
            p["bertscore_f1"] = bertscorer.f1(p["text_a"], p["text_b"])
        if sentcos is not None:
            p["sentcos"] = sentcos.cos(p["text_a"], p["text_b"])
        if rougel is not None:
            p["rougel_f1"] = rougel.f1(p["text_a"], p["text_b"])
        if jaccard_on:
            p["jaccard"] = jaccard(p["text_a"], p["text_b"])
    return pairs
