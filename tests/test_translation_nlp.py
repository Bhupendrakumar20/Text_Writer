"""Unit and Integration Tests for Multilingual Translation & NLP System."""
import unittest
from textwritter.multilingual.translation import (
    TRANSLATION_LANGUAGES,
    get_grouped_languages,
    get_translation_language,
    list_available_models,
    route_translation,
    get_translation_engine,
)


class TestMultilingualTranslation(unittest.TestCase):

    def setUp(self):
        self.engine = get_translation_engine()

    def test_language_registry_completeness(self):
        """Verify presence and categorization of key required languages."""
        req_codes = ["en", "hi", "ta", "te", "ml", "kn", "bn", "gu", "mr", "pa", "sa", "cy", "ga", "mi", "qu", "am", "yo", "sw", "bo", "tcy"]
        for code in req_codes:
            lang = get_translation_language(code)
            self.assertIsNotNone(lang, f"Language {code} must be in the registry.")
            self.assertTrue(len(lang.name) > 0)
            self.assertTrue(len(lang.native_name) > 0)

        # Check Tulu is explicitly marked unsupported in standard checkpoints
        tulu = get_translation_language("tcy")
        self.assertFalse(tulu.is_supported)
        self.assertTrue(tulu.is_low_resource)

    def test_model_listing(self):
        """Verify models include MADLAD-400, IndicTrans2, and NLLB-200."""
        models = list_available_models()
        model_ids = [m["id"] for m in models]
        self.assertIn("madlad400", model_ids)
        self.assertIn("indictrans2", model_ids)
        self.assertIn("nllb200", model_ids)

    def test_intelligent_routing(self):
        """Verify intelligent model routing decisions."""
        # 1. English to Tamil -> IndicTrans2
        dec_en_ta = route_translation("en", "ta")
        self.assertEqual(dec_en_ta.model_id, "indictrans2")
        self.assertTrue(dec_en_ta.is_supported)

        # 2. Hindi to Tamil (Indic-to-Indic) -> IndicTrans2
        dec_hi_ta = route_translation("hi", "ta")
        self.assertEqual(dec_hi_ta.model_id, "indictrans2")

        # 3. English to Welsh (Low-resource European) -> NLLB-200
        dec_en_cy = route_translation("en", "cy")
        self.assertEqual(dec_en_cy.model_id, "nllb200")
        self.assertTrue(dec_en_cy.is_low_resource)

        # 4. English to Tulu (Unsupported) -> Honest unsupported status
        dec_en_tcy = route_translation("en", "tcy")
        self.assertFalse(dec_en_tcy.is_supported)
        self.assertEqual(dec_en_tcy.confidence_tier, "Unsupported")

    def test_english_to_tamil(self):
        res = self.engine.translate(
            text="Education is the most powerful weapon which you can use to change the world.",
            source_lang="en",
            target_lang="ta"
        )
        self.assertEqual(res.target_language, "ta")
        self.assertTrue(len(res.translated_text) > 5)
        self.assertEqual(res.translation_status, "Success")
        print("\n[TEST] EN -> TA:", res.translated_text)

    def test_english_to_telugu(self):
        res = self.engine.translate(
            text="Knowledge is power.",
            source_lang="en",
            target_lang="te"
        )
        self.assertEqual(res.target_language, "te")
        self.assertTrue(len(res.translated_text) > 3)
        print("[TEST] EN -> TE:", res.translated_text)

    def test_english_to_malayalam(self):
        res = self.engine.translate(
            text="Hard work is the key to success.",
            source_lang="en",
            target_lang="ml"
        )
        self.assertEqual(res.target_language, "ml")
        self.assertTrue(len(res.translated_text) > 3)
        print("[TEST] EN -> ML:", res.translated_text)

    def test_english_to_kannada(self):
        res = self.engine.translate(
            text="Truth and non-violence are the supreme ideals of life.",
            source_lang="en",
            target_lang="kn"
        )
        self.assertEqual(res.target_language, "kn")
        self.assertTrue(len(res.translated_text) > 3)
        print("[TEST] EN -> KN:", res.translated_text)

    def test_english_to_hindi(self):
        res = self.engine.translate(
            text="Education brings freedom and peace.",
            source_lang="en",
            target_lang="hi"
        )
        self.assertEqual(res.target_language, "hi")
        self.assertTrue(len(res.translated_text) > 3)
        print("[TEST] EN -> HI:", res.translated_text)

    def test_hindi_to_english(self):
        res = self.engine.translate(
            text="सत्य और अहिंसा ही जीवन के आदर्श हैं।",
            source_lang="hi",
            target_lang="en"
        )
        self.assertEqual(res.target_language, "en")
        self.assertTrue(len(res.translated_text) > 3)
        print("[TEST] HI -> EN:", res.translated_text)

    def test_english_to_welsh(self):
        res = self.engine.translate(
            text="Welcome to our school.",
            source_lang="en",
            target_lang="cy"
        )
        self.assertEqual(res.target_language, "cy")
        self.assertTrue(len(res.translated_text) > 3)
        print("[TEST] EN -> CY (Welsh):", res.translated_text)

    def test_low_resource_sanskrit(self):
        res = self.engine.translate(
            text="Truth always triumphs.",
            source_lang="en",
            target_lang="sa"
        )
        self.assertEqual(res.target_language, "sa")
        self.assertTrue(res.is_low_resource)
        print("[TEST] EN -> SA (Sanskrit):", res.translated_text)

    def test_unsupported_language_honesty(self):
        res = self.engine.translate(
            text="Hello world",
            source_lang="en",
            target_lang="tcy"
        )
        self.assertEqual(res.confidence_level, "Unsupported")
        self.assertIn("unmodeled", res.translation_status.lower())
        # Original text preserved, not fake generated
        self.assertEqual(res.translated_text, "Hello world")
        print("[TEST] EN -> TCY (Tulu Unsupported):", res.translation_status[:60])

    def test_empty_input(self):
        res = self.engine.translate(text="", source_lang="en", target_lang="hi")
        self.assertEqual(res.translated_text, "")

    def test_paragraph_structure_preservation(self):
        para_text = "First paragraph about education.\n\nSecond paragraph about discipline."
        res = self.engine.translate(text=para_text, source_lang="en", target_lang="hi")
        self.assertIn("\n", res.translated_text)
        print("[TEST] Multi-paragraph preservation verified.")


if __name__ == "__main__":
    unittest.main()
