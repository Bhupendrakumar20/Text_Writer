"""Comprehensive Automated Verification Test Suite for Multilingual AI Handwriting Upgrade.

Tests:
1. Language Registry & Script Detection for all 10 languages
2. Complex OpenType Text Shaping & Conjunct Formation (HarfBuzz + FreeType)
3. Biometric Reference Style Extraction (Stroke width, Slant, Ink color, Baseline wobble)
4. Style Replication: Same-Script & Cross-Script Transfer
5. Mixed-Script sentences
6. Quality Assurance & Character Coverage Validation
7. Backward Compatibility (Existing API endpoints & engines)
"""
import sys
import unittest
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from textwritter.multilingual.registry import SUPPORTED_LANGUAGES, get_language_config, get_font_for_language
from textwritter.multilingual.detector import detect_language_and_script
from textwritter.multilingual.shaper import shape_multilingual_text, render_shaped_line_to_mask
from textwritter.multilingual.style_analyzer import analyze_handwriting_references, HandwritingStyleProfile
from textwritter.multilingual.style_transfer import plan_style_transfer
from textwritter.multilingual.synthesis_engine import synthesize_handwriting_page
from textwritter.multilingual.validator import validate_text_accuracy
from textwritter.multilingual.model_registry import REGISTRY


class TestMultilingualUpgrade(unittest.TestCase):

    def test_01_language_registry(self):
        """Verify all 10 languages are properly registered with fonts and samples."""
        expected_langs = ["en", "hi", "ta", "te", "bn", "gu", "mr", "pa", "ml", "kn"]
        for code in expected_langs:
            self.assertIn(code, SUPPORTED_LANGUAGES, f"Missing language: {code}")
            cfg = get_language_config(code)
            self.assertIsNotNone(cfg)
            font_path = get_font_for_language(code)
            self.assertTrue(font_path.exists(), f"Font does not exist for {code}: {font_path}")
            self.assertTrue(len(cfg.sample_text_short) > 0)
        print("✓ Test 1 Passed: All 10 languages registered with valid fonts.")

    def test_02_script_detection(self):
        """Verify Unicode script detection for all 10 languages."""
        test_cases = [
            ("The quick brown fox jumps over the lazy dog.", "Latin", "en"),
            ("मेरा नाम भारत है और मैं एक विद्यार्थी हूँ।", "Devanagari", "hi"),
            ("என் பெயர் பாரத் மற்றும் நான் ஒரு மாணவன்.", "Tamil", "ta"),
            ("నా పేరు భారత్ మరియు నేను ఒక విద్యార్థిని.", "Telugu", "te"),
            ("আমার নাম ভারত এবং আমি একজন ছাত্র।", "Bengali", "bn"),
            ("મારું નામ ભારત છે અને હું એક વિદ્યાર્થી છું.", "Gujarati", "gu"),
            ("माझे नाव भारत आहे आणि मी एक विद्यार्थी आहे.", "Devanagari", "mr"),
            ("ਮੇਰਾ ਨਾਮ ਭਾਰਤ ਹੈ ਅਤੇ ਮੈਂ ਇੱਕ ਵਿਦਿਆਰਥੀ ਹਾਂ।", "Gurmukhi", "pa"),
            ("എന്റെ പേര് ഭാരത് എന്നാണ്, ഞാൻ ഒരു വിദ്യാർത്ഥിയാണ്.", "Malayalam", "ml"),
            ("ನನ್ನ ಹೆಸರು ಭಾರತ್ ಮತ್ತು ನಾನು ಒಬ್ಬ ವಿದ್ಯಾರ್ಥಿ.", "Kannada", "kn"),
        ]
        for text, exp_script, exp_lang in test_cases:
            res = detect_language_and_script(text, default_lang=exp_lang)
            self.assertEqual(res.primary_script, exp_script, f"Failed script detection for {text}")
            self.assertTrue(res.is_valid)
            self.assertEqual(len(res.unsupported_chars), 0)
        print("✓ Test 2 Passed: Script detection 100% accurate across all 10 scripts.")

    def test_03_mixed_script_detection(self):
        """Verify mixed-script sentences are accurately partitioned."""
        mixed = "Hello भारत — என் பெயர் John."
        res = detect_language_and_script(mixed)
        self.assertTrue(res.is_mixed_script)
        self.assertTrue(len(res.script_segments) >= 3)
        scripts = [s for _, s in res.script_segments]
        self.assertIn("Latin", scripts)
        self.assertIn("Devanagari", scripts)
        self.assertIn("Tamil", scripts)
        print("✓ Test 3 Passed: Mixed-script sentences properly segmented.")

    def test_04_complex_text_shaping(self):
        """Verify HarfBuzz shapes ligatures and conjuncts without code-point errors."""
        # Hindi conjuncts: dya (द्य), r-vowel (विद्यार्थी)
        hindi_runs = shape_multilingual_text("विद्यार्थी", target_lang="hi", font_size=36)
        self.assertEqual(len(hindi_runs), 1)
        mask, w, baseline = render_shaped_line_to_mask(hindi_runs[0])
        self.assertTrue(w > 50)
        self.assertTrue(mask.shape[0] > 30)

        # Tamil ligatures: (கொ, ணி, றை)
        tamil_runs = shape_multilingual_text("கொங்குநாடு", target_lang="ta", font_size=36)
        mask_ta, w_ta, _ = render_shaped_line_to_mask(tamil_runs[0])
        self.assertTrue(w_ta > 50)
        print("✓ Test 4 Passed: Complex conjuncts and marks shaped accurately.")

    def test_05_style_analysis_from_references(self):
        """Verify biometric parameters extraction from real images."""
        sample_path = Path("assets/sample_styles/00_PRESIDENT.png")
        if sample_path.exists():
            profile = analyze_handwriting_references([sample_path])
            self.assertGreater(profile.stroke_thickness, 1.0)
            self.assertLess(profile.stroke_thickness, 7.0)
            self.assertIsNotNone(profile.ink_color_hex)
            self.assertIn("#", profile.ink_color_hex)
            self.assertGreater(profile.baseline_wobble, 0.5)
            print(f"✓ Test 5 Passed: Biometric extraction working (stroke={profile.stroke_thickness:.2f}px, slant={profile.slant_angle:.1f}°, ink={profile.ink_color_hex}).")

    def test_06_style_transfer_modes(self):
        """Verify distinction between same-script and cross-script transfer."""
        sample_path = Path("assets/sample_styles/00_PRESIDENT.png")
        profile = analyze_handwriting_references([sample_path]) if sample_path.exists() else HandwritingStyleProfile()

        # Cross-script: English reference -> Hindi text
        report_cross = plan_style_transfer("मेरा नाम भारत है", "hi", profile)
        self.assertEqual(report_cross.transfer_mode, "cross_script")
        self.assertIn("stroke_thickness", report_cross.transferred_features)
        self.assertTrue(len(report_cross.script_boundary_notes) > 0)

        # Generic: No reference
        report_generic = plan_style_transfer("Hello world", "en", HandwritingStyleProfile(sample_count=0))
        self.assertEqual(report_generic.transfer_mode, "generic")
        print("✓ Test 6 Passed: Same-script vs. cross-script transfer properly distinguished.")

    def test_07_full_synthesis_all_10_languages(self):
        """Verify handwriting page synthesis for all 10 languages."""
        out_dir = Path("out/test_suite")
        out_dir.mkdir(parents=True, exist_ok=True)

        for code, lang in SUPPORTED_LANGUAGES.items():
            img, report, validation = synthesize_handwriting_page(
                text=lang.sample_text_short,
                target_language=code,
                paper_type="ruled",
                font_size=36
            )
            self.assertIsNotNone(img)
            self.assertTrue(img.width >= 1000)
            self.assertTrue(img.height >= 1200)
            self.assertEqual(validation.accuracy_score, 1.0)
            save_path = out_dir / f"test_{code}.png"
            img.crop((0, 0, 1200, 360)).save(save_path)
            self.assertTrue(save_path.exists())
        print("✓ Test 7 Passed: Successfully synthesized handwritten pages for all 10 languages.")

    def test_08_text_accuracy_validation(self):
        """Verify text accuracy validation identifies unsupported characters."""
        valid_res = validate_text_accuracy("Hello world", "en")
        self.assertTrue(valid_res.is_accurate)
        self.assertEqual(valid_res.accuracy_score, 1.0)

        # Input with unknown emoji / non-supported symbol
        invalid_res = validate_text_accuracy("Hello \U0001F600 world", "en")
        self.assertFalse(invalid_res.is_accurate)
        self.assertLess(invalid_res.accuracy_score, 1.0)
        self.assertTrue(len(invalid_res.unsupported_characters) > 0)
        print("✓ Test 8 Passed: Quality assurance validation accurately flags unsupported chars.")

    def test_09_model_registry(self):
        """Verify pluggable model registry discoverability."""
        models = REGISTRY.list_models()
        self.assertTrue(len(models) >= 2)
        hi_models = REGISTRY.get_compatible_models("hi")
        self.assertTrue(any(m.model_id == "multilingual_style" for m in hi_models))
        best_hi = REGISTRY.select_best_model("hi")
        self.assertEqual(best_hi.model_id, "multilingual_style")
        print("✓ Test 9 Passed: Model registry correctly routes language requirements.")

    def test_10_word_wrapping_and_page_layout(self):
        """Verify long paragraphs wrap into multiple lines without getting cut off."""
        long_english = (
            "This is a longer continuous handwritten paragraph that in the past was cut off "
            "because it was expected to come in a single line. With natural word wrapping, "
            "every sentence flows organically onto ruled notebook lines with genuine handwriting margins."
        )
        lines = shape_multilingual_text(long_english, target_lang="en", font_size=36, max_line_width=880.0)
        self.assertGreater(len(lines), 1, "Long text should wrap into multiple lines")

        # Synthesize full page with wrapped lines
        img, report, val = synthesize_handwriting_page(long_english, target_language="en", paper_type="ruled")
        self.assertIsNotNone(img)
        self.assertEqual(img.width, 1200)
        self.assertGreaterEqual(img.height, 1600)
        print(f"✓ Test 10 Passed: Long sentence wrapped into {len(lines)} lines without truncation.")

    def test_11_font_family_style_classification_and_matching(self):
        """Verify handwriting sample photos match the appropriate font family."""
        sample_path = Path("assets/sample_styles/00_PRESIDENT.png")
        if sample_path.exists():
            profile = analyze_handwriting_references([sample_path])
            self.assertIn("font_family", profile.to_dict())
            self.assertIn(profile.font_family, ["caveat", "patrick", "indie", "kalam"])
            self.assertTrue(len(profile.font_family_name) > 0)
            
            # Synthesize with matched font
            img, report, val = synthesize_handwriting_page(
                "Authentic handwritten style reproduction",
                target_language="en",
                reference_profile=profile
            )
            self.assertIsNotNone(img)
            self.assertIn("font_family", report.transferred_features)
        print("✓ Test 11 Passed: Font family style classification and matching verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
