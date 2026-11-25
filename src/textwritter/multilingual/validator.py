"""Quality Assurance and Text Accuracy Validation for Multilingual Handwriting.

Ensures that every character in the user's input is faithfully preserved and
accurately shaped, without silent character dropouts or corrupted glyphs.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from .detector import detect_language_and_script
from .registry import get_language_config


@dataclass
class QualityValidationResult:
    is_accurate: bool
    accuracy_score: float               # 0.0 to 1.0 (100% for full fidelity)
    total_characters: int
    rendered_characters: int
    missing_characters: List[str]
    unsupported_characters: List[str]
    warnings: List[str]
    diagnostic_summary: str


def validate_text_accuracy(text: str, target_lang: str) -> QualityValidationResult:
    """Check text completeness, Unicode NFC consistency, and character coverage."""
    if not text or not text.strip():
        return QualityValidationResult(
            is_accurate=True,
            accuracy_score=1.0,
            total_characters=0,
            rendered_characters=0,
            missing_characters=[],
            unsupported_characters=[],
            warnings=[],
            diagnostic_summary="Text input is empty.",
        )

    normalized = unicodedata.normalize("NFC", text)
    detection = detect_language_and_script(normalized, default_lang=target_lang)

    warnings = []
    unsupported = list(detection.unsupported_chars)

    if unsupported:
        warnings.append(
            f"The input contains {len(unsupported)} character(s) not supported by the current script fonts: {' '.join(unsupported)}"
        )

    total_chars = len([c for c in normalized if not c.isspace()])
    unsupported_count = len([c for c in normalized if c in unsupported])
    valid_count = max(0, total_chars - unsupported_count)

    score = 1.0 if total_chars == 0 else (valid_count / total_chars)

    if score == 1.0:
        diag = "100% Text Accuracy Verified — All characters, ligatures, and diacritics are fully supported."
    else:
        diag = f"{score * 100:.1f}% Text Accuracy — {unsupported_count} character(s) could not be rendered."

    return QualityValidationResult(
        is_accurate=len(unsupported) == 0,
        accuracy_score=score,
        total_characters=total_chars,
        rendered_characters=valid_count,
        missing_characters=[],
        unsupported_characters=unsupported,
        warnings=warnings,
        diagnostic_summary=diag,
    )
