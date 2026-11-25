"""Language & Script Detection for Text_Writter.

Performs robust Unicode-based script identification, character category
classification, mixed-script detection, and validation for 10 languages:
English, Hindi, Tamil, Telugu, Bengali, Gujarati, Marathi, Punjabi, Malayalam, Kannada.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .registry import SUPPORTED_LANGUAGES, SCRIPT_TO_LANGUAGE, LanguageConfig, get_language_config


@dataclass
class DetectionResult:
    primary_language: str
    primary_script: str
    confidence: float
    detected_scripts: Dict[str, float]  # script -> proportion
    is_mixed_script: bool
    script_segments: List[Tuple[str, str]]  # list of (segment_text, script_name)
    unsupported_chars: List[str]
    is_valid: bool
    diagnostic_message: str


def get_char_script(char: str) -> str:
    """Classify a single character by Unicode script block."""
    cp = ord(char)

    # Ignore whitespace and standard ASCII punctuation/digits from script determination
    if char.isspace() or unicodedata.category(char).startswith("P"):
        return "Common"
    if char.isdigit():
        # Check if ASCII digit or Indic native digit
        if 0x0030 <= cp <= 0x0039:
            return "Common"
        elif 0x0966 <= cp <= 0x096F:
            return "Devanagari"
        elif 0x0BE6 <= cp <= 0x0BEF:
            return "Tamil"
        elif 0x0C66 <= cp <= 0x0C6F:
            return "Telugu"
        elif 0x09E6 <= cp <= 0x09EF:
            return "Bengali"
        elif 0x0AE6 <= cp <= 0x0AEF:
            return "Gujarati"
        elif 0x0A66 <= cp <= 0x0A6F:
            return "Gurmukhi"
        elif 0x0D66 <= cp <= 0x0D6F:
            return "Malayalam"
        elif 0x0CE6 <= cp <= 0x0CEF:
            return "Kannada"

    # Match against supported languages
    for lang in SUPPORTED_LANGUAGES.values():
        for start, end in lang.unicode_ranges:
            if start <= cp <= end:
                return lang.script

    # General Latin
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
        return "Latin"

    # Math symbols
    if 0x2200 <= cp <= 0x22FF or 0x2190 <= cp <= 0x21FF:
        return "Math"

    return "Unknown"


def detect_language_and_script(text: str, default_lang: str = "en") -> DetectionResult:
    """Analyze input text to determine language, script composition, and validity."""
    if not text or not text.strip():
        return DetectionResult(
            primary_language=default_lang,
            primary_script="Latin",
            confidence=0.0,
            detected_scripts={},
            is_mixed_script=False,
            script_segments=[],
            unsupported_chars=[],
            is_valid=True,
            diagnostic_message="Empty text provided.",
        )

    # Normalize Unicode to NFC
    normalized_text = unicodedata.normalize("NFC", text)

    script_counts: Dict[str, int] = {}
    unsupported_chars: List[str] = []
    script_segments: List[Tuple[str, str]] = []

    current_script: Optional[str] = None
    current_segment: List[str] = []

    total_meaningful_chars = 0

    for char in normalized_text:
        script = get_char_script(char)

        if script == "Unknown":
            unsupported_chars.append(char)
        elif script != "Common" and script != "Math":
            script_counts[script] = script_counts.get(script, 0) + 1
            total_meaningful_chars += 1

        # Segmentation by script for mixed-script rendering
        segment_script = script if script not in ("Common", "Math") else (current_script or "Latin")
        if current_script is None:
            current_script = segment_script
            current_segment.append(char)
        elif segment_script == current_script or script in ("Common", "Math"):
            current_segment.append(char)
        else:
            script_segments.append(("".join(current_segment), current_script))
            current_script = segment_script
            current_segment = [char]

    if current_segment:
        script_segments.append(("".join(current_segment), current_script or "Latin"))

    # Determine primary script
    if not script_counts:
        # Only punctuation or common digits
        primary_script = "Latin"
        primary_lang = default_lang
        confidence = 1.0
        script_proportions = {"Latin": 1.0}
    else:
        primary_script = max(script_counts, key=script_counts.get)
        primary_lang = SCRIPT_TO_LANGUAGE.get(primary_script, default_lang)
        script_proportions = {
            s: count / total_meaningful_chars for s, count in script_counts.items()
        }
        confidence = script_proportions.get(primary_script, 1.0)

    is_mixed = len(script_counts) > 1

    # Diagnostics
    if unsupported_chars:
        unique_unsupported = sorted(set(unsupported_chars))
        diag = f"Warning: {len(unique_unsupported)} unsupported character(s) detected: {' '.join(unique_unsupported)}"
    elif is_mixed:
        proportions_str = ", ".join(f"{s}: {p*100:.1f}%" for s, p in script_proportions.items())
        diag = f"Mixed-script text detected ({proportions_str}). Each segment will be styled with its script-appropriate font."
    else:
        diag = f"Single-script {primary_script} text detected (Language: {primary_lang.upper()})."

    return DetectionResult(
        primary_language=primary_lang,
        primary_script=primary_script,
        confidence=confidence,
        detected_scripts=script_proportions,
        is_mixed_script=is_mixed,
        script_segments=script_segments,
        unsupported_chars=sorted(set(unsupported_chars)),
        is_valid=len(unsupported_chars) == 0,
        diagnostic_message=diag,
    )
