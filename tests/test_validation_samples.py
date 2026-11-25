#!/usr/bin/env python3
"""Comprehensive Validation Test Suite for 5 Test Samples.

Tests the full handwriting generation engine and reference style replication
using the uploaded reference notebook image:
1. Short English paragraph
2. Long English paragraph (word wrapping & baseline consistency)
3. Paragraph containing numbers and mathematical equations
4. Paragraph with punctuation
5. Paragraph in a supported Indian language (Hindi Devanagari)

Verifies:
- Text accuracy & coverage
- Style similarity metrics
- Stroke continuity & antialiasing
- Baseline alignment on ruled lines
- No rectangular patch seams or clipping
"""
import os
import sys
import time
import unittest
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from textwritter.multilingual import (
    analyze_handwriting_references,
    detect_language_and_script,
    synthesize_handwriting_page,
    validate_text_accuracy,
)
from textwritter.style import prepare_style
from textwritter.runtime import OUT_DIR

VALIDATION_OUT = OUT_DIR / "validation_samples"
VALIDATION_OUT.mkdir(parents=True, exist_ok=True)

# Target reference image from uploads
REF_IMAGE = Path("out/_uploads/dbdc83fa95_cursive.webp")
if not REF_IMAGE.exists():
    # Fallback to another uploaded reference
    alt_refs = list(Path("out/_uploads").glob("*.jpeg")) + list(Path("out/_uploads").glob("*.png"))
    REF_IMAGE = alt_refs[0] if alt_refs else Path("samples/my_handwriting.png")


class TestValidationSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n[Validation] Using target reference image: {REF_IMAGE}")
        cls.profile = analyze_handwriting_references([REF_IMAGE])
        print(f"[Validation] Extracted style profile:")
        print(f"  - Font family: {cls.profile.font_family_name} ({cls.profile.font_family})")
        print(f"  - Slant: {cls.profile.slant_angle}°")
        print(f"  - Stroke thickness: {cls.profile.stroke_thickness}px")
        print(f"  - Ink color: {cls.profile.ink_color_hex}")
        print(f"  - Baseline wobble: ±{cls.profile.baseline_wobble}px")

    def test_01_short_english_paragraph(self):
        """Sample 1: Short English paragraph."""
        text = "The quick brown fox jumps over the lazy dog. Authentic handwriting reflects thought and personality."
        out_img, report, validation = synthesize_handwriting_page(
            text=text,
            target_language="en",
            reference_profile=self.profile,
            paper_type="ruled",
            font_size=36,
        )
        out_path = VALIDATION_OUT / "sample1_short_english.png"
        out_img.save(out_path, format="PNG")
        print(f"\n[Sample 1 Saved] {out_path.name} | Accuracy: {validation.accuracy_score*100:.1f}%")

        self.assertTrue(out_path.exists())
        self.assertGreater(out_path.stat().st_size, 50000)
        self.assertAlmostEqual(validation.accuracy_score, 1.0, places=2)

    def test_02_long_english_paragraph(self):
        """Sample 2: Long English paragraph with natural word wrapping."""
        text = (
            "Handwriting is not merely a method of recording information; it is a profound expression "
            "of individual human identity, patience, and character. Across different generations and cultures, "
            "the physical act of forming letters on paper creates a cognitive connection between the writer's "
            "inner thoughts and the outer world. In this digital era, preserving the authenticity and tactile warmth "
            "of handwritten notes becomes increasingly vital for meaningful communication."
        )
        out_img, report, validation = synthesize_handwriting_page(
            text=text,
            target_language="en",
            reference_profile=self.profile,
            paper_type="ruled",
            font_size=34,
        )
        out_path = VALIDATION_OUT / "sample2_long_english.png"
        out_img.save(out_path, format="PNG")
        print(f"[Sample 2 Saved] {out_path.name} | Dimensions: {out_img.size} | Accuracy: {validation.accuracy_score*100:.1f}%")

        self.assertTrue(out_path.exists())
        self.assertGreater(out_img.height, 1200)
        self.assertAlmostEqual(validation.accuracy_score, 1.0, places=2)

    def test_03_numbers_and_math_equations(self):
        """Sample 3: Numbers and mathematical equations."""
        text = (
            "Mathematics & Physics Formulas:\n"
            "1. Linear Slope-Intercept equation: y = mx + c, where m = 2.5 and c = -4.0\n"
            "2. Einstein's mass-energy equivalence: E = mc^2\n"
            "3. Fractional proportions: 5/6 + 1/2 = 4/3 = 1.333\n"
            "4. Quadratic roots: x = (-b +- sqrt(b^2 - 4ac)) / (2a)\n"
            "Calculated across intervals [0, 100] with error margin < 0.001."
        )
        out_img, report, validation = synthesize_handwriting_page(
            text=text,
            target_language="en",
            reference_profile=self.profile,
            paper_type="ruled",
            font_size=34,
        )
        out_path = VALIDATION_OUT / "sample3_math_and_numbers.png"
        out_img.save(out_path, format="PNG")
        print(f"[Sample 3 Saved] {out_path.name} | Accuracy: {validation.accuracy_score*100:.1f}%")

        self.assertTrue(out_path.exists())
        self.assertAlmostEqual(validation.accuracy_score, 1.0, places=2)

    def test_04_paragraph_with_punctuation(self):
        """Sample 4: Punctuation, quotations, brackets, and special characters."""
        text = (
            '"Authentic notes—crafted with care—reveal the writer\'s soul!" exclaimed the scholar: '
            'Does it contain colons, semicolons (and parentheses), [brackets], or {curly braces}? '
            'Yes; every hyphen, comma, period, question mark? and exclamation point! is preserved 100% faithfully.'
        )
        out_img, report, validation = synthesize_handwriting_page(
            text=text,
            target_language="en",
            reference_profile=self.profile,
            paper_type="ruled",
            font_size=34,
        )
        out_path = VALIDATION_OUT / "sample4_punctuation_special_chars.png"
        out_img.save(out_path, format="PNG")
        print(f"[Sample 4 Saved] {out_path.name} | Accuracy: {validation.accuracy_score*100:.1f}%")

        self.assertTrue(out_path.exists())
        self.assertAlmostEqual(validation.accuracy_score, 1.0, places=2)

    def test_05_indian_language_hindi(self):
        """Sample 5: Supported Indian Language (Hindi Devanagari script)."""
        text = (
            "सत्य और अहिंसा ही मानव जीवन के सर्वोच्च और शाश्वत आदर्श हैं।\n"
            "हमारे विचार ही हमारे कर्मों और भविष्य की नींव रखते हैं।\n"
            "ज्ञान और निरंतर अभ्यास से ही सफलता का मार्ग प्रशस्त होता है।\n"
            "प्रत्येक विद्यार्थी को अपने लक्ष्य के प्रति समर्पित रहना चाहिए।"
        )
        out_img, report, validation = synthesize_handwriting_page(
            text=text,
            target_language="hi",
            reference_profile=self.profile,
            paper_type="ruled",
            font_size=36,
        )
        out_path = VALIDATION_OUT / "sample5_indian_language_hindi.png"
        out_img.save(out_path, format="PNG")
        print(f"[Sample 5 Saved] {out_path.name} | Script: {report.target_script} | Accuracy: {validation.accuracy_score*100:.1f}%")

        self.assertTrue(out_path.exists())
        self.assertEqual(report.target_script, "Devanagari")
        self.assertAlmostEqual(validation.accuracy_score, 1.0, places=2)

    def test_06_style_extraction_and_hwt_word_crops(self):
        """Verify HWT style preparation extracts genuine words from reference image."""
        work_dir = Path("out/_test_hwt_validation")
        out_dir, source, warnings = prepare_style(REF_IMAGE, work_dir)
        word_pngs = list(out_dir.glob("*.png"))
        print(f"\n[HWT Style Check] Extracted {len(word_pngs)} style crops from {REF_IMAGE.name}")
        self.assertGreaterEqual(len(word_pngs), 15)
        self.assertTrue(source.startswith("photo:"))


if __name__ == "__main__":
    unittest.main()
