"""
Shared rationale extraction and matching helpers for Dimension-2 analysis.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

_SMISHING_ALIASES = {"1", "smish", "smishing", "phishing", "phish"}
_HAM_ALIASES = {"0", "ham", "benign", "legitimate", "legit"}

_STANDALONE_FINAL_RE = re.compile(
    r"""(?imx)
    ^\s*(?:[*_`~]+\s*)?
    FINAL\s*:\s*
    (?:smishing|smish|phishing|phish|ham|benign|legitimate|legit|1|0)
    \s*[.!]?(?:\s*[*_`~]+)?\s*$
    """
)

_INLINE_FINAL_RE = re.compile(
    r"""(?ix)
    (?:[*_`~]+\s*)?\bFINAL\s*:\s*
    (?:smishing|smish|phishing|phish|ham|benign|legitimate|legit|1|0)
    \b\s*[.!]?(?:\s*[*_`~]+)?
    """
)

_LEADING_LABEL_TOKEN_RE = re.compile(
    r"""(?ix)
    ^\s*(?:[*_`~]+\s*)?
    (?:
        (?:the\s+)?(?:message|sms|text)\s+
        (?:is|was|appears\s+to\s+be|should\s+be\s+classified\s+as)\s+
      | (?:classification|label|answer)\s*:\s*
      | classified\s+as\s+
    )?
    (?:smishing|smish|phishing|phish|ham|benign|legitimate|legit)\b
    """
)


def normalize_label(label: object) -> Optional[str]:
    if label is None or isinstance(label, bool):
        return None
    if isinstance(label, (int, float)):
        if label == 1:
            return "smishing"
        if label == 0:
            return "ham"
    normalized = str(label).strip().lower()
    if normalized in _SMISHING_ALIASES:
        return "smishing"
    if normalized in _HAM_ALIASES:
        return "ham"
    return None


def extract_rationale_anywhere(
    output: object,
    min_chars: int = 10,
) -> Optional[str]:
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")
    if not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return None

    rationale = _STANDALONE_FINAL_RE.sub(" ", text)
    rationale = _INLINE_FINAL_RE.sub(" ", rationale)
    rationale = re.sub(r"\s+", " ", rationale).strip()

    non_whitespace_length = len(re.sub(r"\s+", "", rationale))
    return rationale if non_whitespace_length >= min_chars else None


def is_eligible_pair(
    clean_pred: object,
    perturbed_pred: object,
    true_label: object,
    clean_output: object,
    perturbed_output: object,
    min_chars: int = 10,
) -> Tuple[bool, Optional[str], Optional[str]]:
    clean_rationale = extract_rationale_anywhere(clean_output, min_chars=min_chars)
    perturbed_rationale = extract_rationale_anywhere(
        perturbed_output, min_chars=min_chars
    )
    gold = normalize_label(true_label)
    clean_label = normalize_label(clean_pred)
    perturbed_label = normalize_label(perturbed_pred)
    eligible = (
        gold is not None
        and clean_label == gold
        and perturbed_label == gold
        and clean_rationale is not None
        and perturbed_rationale is not None
    )
    return eligible, clean_rationale, perturbed_rationale


def _normalize_attack(attack: object) -> Optional[str]:
    if attack is None:
        return None
    value = str(attack).strip().lower().replace("_", "-")
    aliases = {
        "char": "character",
        "character-level": "character",
        "word-level": "word",
        "sentence-level": "sentence",
        "paraphrase": "sentence",
        "multi": "multi-level",
        "multilevel": "multi-level",
    }
    return aliases.get(value, value)


def normalized_seed(attack: object, seed: object) -> Optional[int]:
    attack_name = _normalize_attack(attack)
    if attack_name == "sentence":
        return None
    if seed is None or (isinstance(seed, str) and not seed.strip()):
        return None
    if isinstance(seed, bool):
        raise ValueError(f"Invalid seed {seed!r} for attack {attack!r}")
    try:
        numeric_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid seed {seed!r} for attack {attack!r}"
        ) from exc
    if isinstance(seed, float) and not seed.is_integer():
        raise ValueError(f"Invalid non-integer seed {seed!r}")
    if isinstance(seed, str) and not re.fullmatch(r"[+-]?\d+", seed.strip()):
        raise ValueError(f"Invalid seed {seed!r}")
    return numeric_seed


def baseline_key(
    model: object,
    template: object,
    pred_label: object,
    attack: object,
    seed: object,
) -> tuple:
    canonical_label = normalize_label(pred_label)
    if canonical_label is None:
        raise ValueError(f"Unrecognized predicted label: {pred_label!r}")
    canonical_attack = _normalize_attack(attack)
    canonical_seed = normalized_seed(canonical_attack, seed)
    return (
        str(model),
        str(template),
        canonical_label,
        canonical_attack,
        canonical_seed,
    )


def has_leading_label_token(rationale: object) -> bool:
    if not isinstance(rationale, str):
        return False
    return bool(_LEADING_LABEL_TOKEN_RE.search(rationale.strip()))


def _run_self_tests() -> None:
    assert normalize_label("phishing") == "smishing"
    assert normalize_label("1") == "smishing"
    assert normalize_label(1) == "smishing"
    assert normalize_label("legitimate") == "ham"
    assert normalize_label("0") == "ham"
    assert normalize_label(0) == "ham"
    assert normalize_label("unparsed") is None

    assert extract_rationale_anywhere(
        "Evidence appears first.\nFINAL: smishing", 10
    ) == "Evidence appears first."
    assert extract_rationale_anywhere(
        "**FINAL: ham** This is an expected personal conversation.", 10
    ) == "This is an expected personal conversation."
    assert extract_rationale_anywhere("FINAL: ham", 10) is None

    eligible, clean_rat, pert_rat = is_eligible_pair(
        "phishing",
        "1",
        "smishing",
        "FINAL: smishing. The unknown sender requests account verification.",
        "Urgency and a suspicious link are present.\nFINAL: phishing",
        10,
    )
    assert eligible and clean_rat and pert_rat

    assert normalized_seed("sentence", 2) is None
    assert normalized_seed("character", "2") == 2
    assert baseline_key("m", "t3", "phishing", "sentence-level", 2) == (
        "m", "t3", "smishing", "sentence", None
    )
    assert baseline_key("m", "t3", "0", "multi", "1") == (
        "m", "t3", "ham", "multi-level", 1
    )


if __name__ == "__main__":
    _run_self_tests()
    print("All rationale-extraction self-tests passed.")
