"""
perturbations.py
----------------
Model-agnostic, corpus-grounded perturbation transforms for smishing robustness eval.

Design decisions (see manuscript Section 5.3):
  - URLs are MASKED before perturbation and RESTORED after -> never altered.
  - Word-level: random selection of non-URL tokens + counter-fitted synonym substitution.
  - Character-level: homoglyph / injected-spacing / deliberate-misspelling on non-URL,
    non-numeric surface tokens (brand & content-adjacent forms).
  - Sentence-level & paraphrase (multi-level): PEGASUS paraphrase model (loaded lazily).
  - Every edit is logged; a semantic-similarity gate is applied downstream (not here).

This module contains ONLY the transforms + logging. Orchestration (seeds, batching,
similarity gate, IO) lives in run_perturbations.py so the two concerns stay separate.
"""

import re
import random
import string
from dataclasses import dataclass, field, asdict
from typing import Callable

# ----------------------------------------------------------------------------
# URL handling: mask -> perturb -> restore. This is the single most important
# invariant in the whole pipeline, so it is deliberately conservative/greedy.
# ----------------------------------------------------------------------------

# Matches http(s)://..., www...., bare domains with a path, app-link/whatsapp schemes,
# and shortener-style host/path tokens seen in the corpus (e.g. gt33.pw/snlXXnncQ3).
_URL_RE = re.compile(
    r"""(
        (?:https?://|www\.)\S+                # explicit http(s) or www
        | (?:[a-z0-9\-]+\.)+[a-z]{2,}/\S+       # domain.tld/path  (catches shorteners w/ path)
        | (?:app\.link|use\.app\.link)/\S+      # app links
        | whatsapp://\S+                        # app schemes
    )""",
    re.IGNORECASE | re.VERBOSE,
)

_URL_PLACEHOLDER = "\u0001URL{}\u0001"  # non-printing sentinel unlikely to appear in SMS

# Alphanumeric identifier / code tokens (tracking numbers, ref codes, order IDs).
# Rationale (manuscript 5.3): these carry no natural-language semantic content, so
# perturbing them spends budget without affecting classification-relevant cues.
# Matches: #-prefixed codes (#US91177J), 'word#digits' (Shipment#JYTH...), and
# standalone mixed alphanumeric tokens with >=1 digit and length >=6 (JYTHII1Z8381N4G4).
# NOTE: applied AFTER url masking so it never touches URL internals. Deliberately
# conservative so it does not swallow ordinary words (which have no digits).
_CODE_RE = re.compile(
    r"""(
        \#[A-Za-z0-9]{4,}                       # #NavyFed, #US91177J, voucher#... handled below
        | [A-Za-z]+\#[A-Za-z0-9]{4,}            # Shipment#JYTH..., voucher#xxnlrispw
        | \b(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\b # bare mixed alnum w/ a digit, len>=6
    )""",
    re.VERBOSE,
)


def mask_protected(text: str):
    """Mask URLs first, then code/identifier tokens. Returns (masked, protected_list).
    A single placeholder namespace covers both, restored in one pass."""
    protected = []

    def _sub_url(m):
        protected.append(m.group(0))
        return _URL_PLACEHOLDER.format(len(protected) - 1)

    masked = _URL_RE.sub(_sub_url, text)

    def _sub_code(m):
        tok = m.group(0)
        # don't re-mask a placeholder we just inserted
        if "\u0001URL" in tok:
            return tok
        protected.append(tok)
        return _URL_PLACEHOLDER.format(len(protected) - 1)

    masked = _CODE_RE.sub(_sub_code, masked)
    return masked, protected


# Backwards-compatible alias (tests / older calls) — now also protects codes.
def mask_urls(text: str):
    return mask_protected(text)


def restore_urls(text: str, urls: list) -> str:
    for i, u in enumerate(urls):
        text = text.replace(_URL_PLACEHOLDER.format(i), u)
    return text


def _is_placeholder(tok: str) -> bool:
    return "\u0001URL" in tok


# ----------------------------------------------------------------------------
# Edit logging
# ----------------------------------------------------------------------------

@dataclass
class Edit:
    op: str            # 'homoglyph' | 'space' | 'misspell' | 'synonym' | 'paraphrase' | 'insert'
    original: str
    replacement: str
    index: int = -1    # token index where applicable


# ----------------------------------------------------------------------------
# Tokenization helpers (whitespace-based; punctuation kept attached so we don't
# have to reassemble spacing perfectly — SMS is noisy anyway).
# ----------------------------------------------------------------------------

def _tokens(text: str):
    return text.split(" ")


def _eligible_word_indices(toks):
    """Indices of tokens eligible for perturbation: not URL placeholders, not pure
    numbers/codes, has at least 2 alphabetic chars."""
    idx = []
    for i, t in enumerate(toks):
        if _is_placeholder(t):
            continue
        alpha = sum(c.isalpha() for c in t)
        if alpha >= 2:
            idx.append(i)
    return idx


def _pick(indices, frac, rng):
    """Pick ~frac of the given indices (at least 1 if any exist)."""
    if not indices:
        return []
    k = max(1, int(round(len(indices) * frac)))
    return rng.sample(indices, min(k, len(indices)))


# ----------------------------------------------------------------------------
# CHARACTER-LEVEL
# ----------------------------------------------------------------------------

_HOMOGLYPH = {
    "o": "0", "O": "0", "l": "1", "i": "1", "e": "3", "a": "@",
    "s": "5", "S": "5", "t": "7", "g": "9", "B": "8",
}


def _homoglyph_word(w, rng):
    chars = list(w)
    cand = [i for i, c in enumerate(chars) if c in _HOMOGLYPH]
    if not cand:
        return w
    i = rng.choice(cand)
    chars[i] = _HOMOGLYPH[chars[i]]
    return "".join(chars)


def _space_word(w, rng):
    """Inject a space into a word: 'WELLS' -> 'WEL LS' style brand-token evasion."""
    if len(w) < 3:
        return w
    i = rng.randint(1, len(w) - 1)
    return w[:i] + " " + w[i:]


def _misspell_word(w, rng):
    """DeepWordBug-style single edit: swap/delete/insert one char."""
    if len(w) < 3:
        return w
    op = rng.choice(["swap", "delete", "insert"])
    i = rng.randint(0, len(w) - 2)
    chars = list(w)
    if op == "swap":
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "delete":
        del chars[i]
    else:  # insert a random lowercase letter
        chars.insert(i, rng.choice(string.ascii_lowercase))
    return "".join(chars)


def character_level(text, frac, rng):
    masked, urls = mask_urls(text)
    toks = _tokens(masked)
    edits = []
    for i in _pick(_eligible_word_indices(toks), frac, rng):
        orig = toks[i]
        style = rng.choice(["homoglyph", "space", "misspell"])
        if style == "homoglyph":
            new = _homoglyph_word(orig, rng)
        elif style == "space":
            new = _space_word(orig, rng)
        else:
            new = _misspell_word(orig, rng)
        if new != orig:
            toks[i] = new
            edits.append(Edit(op=style, original=orig, replacement=new, index=i))
    return restore_urls(" ".join(toks), urls), edits


# ----------------------------------------------------------------------------
# WORD-LEVEL  (counter-fitted synonym substitution; synonyms injected at runtime)
# ----------------------------------------------------------------------------

class SynonymSource:
    """Wraps a counter-fitted embedding synonym lookup.
    In production this is backed by counter-fitted-vectors nearest neighbours.
    A get(word)->list[str] interface keeps the transform decoupled from the source,
    so WordNet or another source could be swapped without touching this file."""

    def __init__(self, lookup: dict | None = None, neighbour_fn: Callable | None = None):
        self._lookup = lookup or {}
        self._neighbour_fn = neighbour_fn

    def get(self, word):
        w = word.lower()
        if self._neighbour_fn is not None:
            return self._neighbour_fn(w)
        return self._lookup.get(w, [])


def _match_case(src, repl):
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl.capitalize()
    return repl


def word_level(text, frac, rng, syn: SynonymSource):
    masked, urls = mask_urls(text)
    toks = _tokens(masked)
    edits = []
    for i in _pick(_eligible_word_indices(toks), frac, rng):
        orig = toks[i]
        # strip trailing punctuation for lookup, reattach after
        core = orig.strip(string.punctuation)
        prefix = orig[: len(orig) - len(orig.lstrip(string.punctuation))]
        suffix = orig[len(prefix) + len(core):]
        cands = syn.get(core)
        cands = [c for c in cands if c.lower() != core.lower()]
        if not cands:
            continue
        repl = _match_case(core, rng.choice(cands))
        new = prefix + repl + suffix
        toks[i] = new
        edits.append(Edit(op="synonym", original=orig, replacement=new, index=i))
    return restore_urls(" ".join(toks), urls), edits


# ----------------------------------------------------------------------------
# SENTENCE-LEVEL & PARAPHRASE  (PEGASUS, loaded lazily so char/word attacks
# don't pay the model-load cost, and so this file imports with no torch present)
# ----------------------------------------------------------------------------

class Paraphraser:
    """Lazy PEGASUS wrapper. .load() pulls the model onto GPU once; .paraphrase()
    returns a single deterministic rewrite (num_beams, no sampling) for reproducibility."""

    def __init__(self, model_name="tuner007/pegasus_paraphrase", device="cuda"):
        # Load safetensors weights directly from the Hub (this model ships a
        # model.safetensors), which avoids the torch<2.6 torch.load restriction
        # (CVE-2025-32434) without upgrading torch or converting anything.
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None

    def load(self):
        if self._model is not None:
            return
        import torch  # noqa
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, use_safetensors=True
        ).to(self.device)
        self._model.eval()

    def paraphrase(self, text, max_length=80):
        self.load()
        import torch
        batch = self._tok([text], truncation=True, padding="longest",
                          max_length=max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**batch, max_length=max_length,
                                       num_beams=5, num_return_sequences=1,
                                       do_sample=False)  # deterministic
        return self._tok.batch_decode(out, skip_special_tokens=True)[0]


def sentence_level(text, rng, paraphraser: Paraphraser):
    """Paraphrase the non-URL body while keeping URLs verbatim. This realizes the
    'benign-looking rewrite' sentence-level attack without a hand-built template bank."""
    masked, urls = mask_urls(text)
    # PEGASUS can choke on the sentinel; temporarily strip placeholders for the
    # paraphrase then re-append URLs at the end (position-agnostic, matches how
    # smishing appends links).
    body = _URL_RE_PLACEHOLDER.sub("", masked).strip()
    para = paraphraser.paraphrase(body) if body else body
    restored = para
    for u in urls:
        restored = restored + " " + u
    edits = [Edit(op="paraphrase", original=text, replacement=restored)]
    return restored.strip(), edits


_URL_RE_PLACEHOLDER = re.compile(r"\u0001URL\d+\u0001")


# ----------------------------------------------------------------------------
# MULTI-LEVEL  (chain: word -> char -> paraphrase; order logged)
# ----------------------------------------------------------------------------

def multi_level(text, frac, rng, syn, paraphraser):
    edits_all = []
    t, e = word_level(text, frac, rng, syn); edits_all += e
    t, e = character_level(t, frac, rng);    edits_all += e
    t, e = sentence_level(t, rng, paraphraser); edits_all += e
    return t, edits_all


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

ATTACKS = {
    "character": "character_level",
    "word": "word_level",
    "sentence": "sentence_level",
    "multi": "multi_level",
}
