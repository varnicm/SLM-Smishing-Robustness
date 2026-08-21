"""Shared rationale extraction helpers for Dimension-2 analysis.

The same extractor must be used for:
  1. rationale provision,
  2. true-pair explanation stability, and
  3. the matched same-label baseline.

A rationale is the generated text that remains after removing every valid
``FINAL: smishing`` or ``FINAL: ham`` label expression, regardless of where
that expression appears. A rationale is usable when the remaining text has at
least ``min_chars`` non-whitespace characters.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Handles standalone or inline labels, optional Markdown emphasis/backticks,
# and optional sentence punctuation. It intentionally removes only the required
# FINAL label expression and leaves all other generated text unchanged.
_FINAL_LABEL_RE = re.compile(
    r"(?ix)"
    r"(?:[`*_~]{0,3})"          # optional Markdown wrappers before FINAL
    r"\bFINAL\s*:\s*"
    r"(?:smishing|ham)"
    r"\b"
    r"(?:[`*_~]{0,3})"          # optional Markdown wrappers after the label
    r"[.!]?"
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_label(label: object) -> Optional[str]:
    """Return ``smishing`` or ``ham`` for a valid label; otherwise ``None``."""
    if not isinstance(label, str):
        return None
    normalized = label.strip().lower()
    return normalized if normalized in {"smishing", "ham"} else None


def extract_rationale_anywhere(
    output: object,
    min_chars: int = 10,
) -> Optional[str]:
    """Extract a usable rationale from a raw model output.

    All valid FINAL label expressions are removed. Text before and after those
    expressions is retained and joined after whitespace normalization.
    """
    if min_chars < 0:
        raise ValueError("min_chars must be non-negative")
    if not isinstance(output, str):
        return None

    text = output.strip()
    if not text:
        return None

    rationale = _FINAL_LABEL_RE.sub(" ", text)
    rationale = _WHITESPACE_RE.sub(" ", rationale).strip(" \t\r\n-–—:;,.!?")

    non_whitespace_length = len(_WHITESPACE_RE.sub("", rationale))
    if non_whitespace_length < min_chars:
        return None
    return rationale


def is_eligible_pair(
    clean_pred: object,
    perturbed_pred: object,
    true_label: object,
    clean_output: object,
    perturbed_output: object,
    min_chars: int = 10,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return eligibility and the exact rationales used for similarity scoring.

    A pair is eligible only when both predictions match the gold label and both
    outputs contain usable rationales. Callers must reuse the returned strings;
    they should not extract the rationales again downstream.
    """
    clean_rationale = extract_rationale_anywhere(clean_output, min_chars=min_chars)
    perturbed_rationale = extract_rationale_anywhere(
        perturbed_output, min_chars=min_chars
    )

    gold = normalize_label(true_label)
    clean = normalize_label(clean_pred)
    perturbed = normalize_label(perturbed_pred)

    eligible = (
        gold is not None
        and clean == gold
        and perturbed == gold
        and clean_rationale is not None
        and perturbed_rationale is not None
    )
    return eligible, clean_rationale, perturbed_rationale


def normalized_seed(attack: object, seed: object) -> Optional[int]:
    """Normalize seed values for matching baseline candidates.

    Sentence paraphrasing is deterministic and contributes one deduplicated
    variant per source message, so its seed is always ``None``. Character,
    word, and multi-level conditions retain their actual integer seed.
    """
    attack_name = str(attack).strip().lower().replace("_", "-")
    if attack_name in {"sentence", "sentence-level"}:
        return None
    if seed is None or seed == "":
        return None
    try:
        return int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid seed value for attack {attack!r}: {seed!r}") from exc


def baseline_key(
    model: object,
    template: object,
    pred_label: object,
    attack: object,
    seed: object,
) -> tuple[str, str, str, str, Optional[int]]:
    """Build the exact key for matched same-label baseline candidates.

    The caller must additionally exclude candidates with the same source-message
    identifier. This function normalizes sentence seeds to ``None`` and preserves
    actual seeds for character, word, and multi-level perturbations.
    """
    label = normalize_label(pred_label)
    if label is None:
        raise ValueError(f"Invalid predicted label: {pred_label!r}")

    attack_name = str(attack).strip().lower().replace("_", "-")
    return (
        str(model),
        str(template),
        label,
        attack_name,
        normalized_seed(attack_name, seed),
    )


def _run_self_test() -> None:
    cases = [
        (
            "before FINAL",
            "The message asks for account verification through a link.\nFINAL: smishing",
            "The message asks for account verification through a link",
        ),
        (
            "after inline FINAL",
            "FINAL: smishing. The sender is unknown and the link is shortened.",
            "The sender is unknown and the link is shortened",
        ),
        (
            "both sides",
            "Urgency cues are present.\nFINAL: smishing\nThe domain also differs from the brand.",
            "Urgency cues are present. The domain also differs from the brand",
        ),
        (
            "Markdown label",
            "**FINAL: ham** Ordinary conversation with no suspicious request.",
            "Ordinary conversation with no suspicious request",
        ),
        ("label only", "FINAL: ham", None),
        ("too short", "FINAL: ham\nlooks ok", None),
        ("empty", "   ", None),
        ("non-string", None, None),
    ]

    for name, output, expected in cases:
        actual = extract_rationale_anywhere(output, min_chars=10)
        assert actual == expected, f"{name}: expected {expected!r}, got {actual!r}"

    eligible, clean_rationale, perturbed_rationale = is_eligible_pair(
        clean_pred="Smishing",
        perturbed_pred="smishing",
        true_label="smishing",
        clean_output="FINAL: smishing. Suspicious link and urgent request.",
        perturbed_output="Urgent account-verification request. FINAL: smishing",
    )
    assert eligible and clean_rationale and perturbed_rationale

    assert baseline_key("m", "t3", "ham", "sentence", 0)[-1] is None
    assert baseline_key("m", "t3", "ham", "character", "2")[-1] == 2
    print("All rationale-extraction self-tests passed.")


if __name__ == "__main__":
    _run_self_test()
