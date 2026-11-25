"""Tests for Multilingual OCR Feature in Text_Writter Studio."""
import io
import unittest
from PIL import Image, ImageDraw

from textwritter.multilingual.ocr import (
    OCRResult,
    extract_text_from_image,
    preprocess_image,
    reconstruct_paragraphs_from_boxes,
)
from textwritter.multilingual.translation import get_translation_engine


class TestMultilingualOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic test image with two paragraphs
        cls.test_img = Image.new("RGB", (600, 240), color=(255, 255, 255))
        d = ImageDraw.Draw(cls.test_img)
        d.text((20, 20), "Education is the most powerful weapon.", fill=(0, 0, 0))
        d.text((20, 55), "Knowledge brings freedom and peace.", fill=(0, 0, 0))
        # Paragraph gap
        d.text((20, 140), "Work hard in silence and let success make the noise.", fill=(0, 0, 0))

        cls.img_bytes_io = io.BytesIO()
        cls.test_img.save(cls.img_bytes_io, format="PNG")
        cls.img_bytes = cls.img_bytes_io.getvalue()

    def test_preprocess_image(self):
        """Preprocess function should return a valid 3-channel numpy array."""
        proc = preprocess_image(self.img_bytes)
        self.assertEqual(len(proc.shape), 3)
        self.assertEqual(proc.shape[2], 3)

    def test_reconstruct_paragraphs_spacing(self):
        """Bounding box clustering should accurately reconstruct lines and separate paragraphs."""
        boxes = [
            ([[20, 20], [200, 20], [200, 35], [20, 35]], "Line 1 paragraph 1", 0.95),
            ([[20, 55], [200, 55], [200, 70], [20, 70]], "Line 2 paragraph 1", 0.92),
            ([[20, 140], [250, 140], [250, 155], [20, 155]], "Line 1 paragraph 2", 0.90),
        ]
        full_text, paras, avg_conf = reconstruct_paragraphs_from_boxes(boxes)
        self.assertIn("Line 1 paragraph 1", full_text)
        self.assertIn("Line 2 paragraph 1", full_text)
        self.assertIn("Line 1 paragraph 2", full_text)
        # Should detect two distinct paragraphs due to the vertical spacing gap (140 - 55 > 1.6 * spacing)
        self.assertGreaterEqual(len(paras), 2)
        self.assertGreaterEqual(avg_conf, 0.90)

    def test_extract_text_from_image(self):
        """extract_text_from_image should extract text and return structured metrics."""
        res = extract_text_from_image(self.img_bytes, language_hint="en")
        self.assertIsInstance(res, OCRResult)
        self.assertGreater(len(res.text), 0)
        self.assertGreaterEqual(res.line_count, 1)
        self.assertGreaterEqual(res.word_count, 1)
        self.assertEqual(res.detected_script, "Latin")
        self.assertIn(res.detected_language, ("en", "auto"))

    def test_ocr_to_translation_flow(self):
        """Text extracted from OCR should seamlessly feed into the NLP translation engine."""
        ocr_res = extract_text_from_image(self.img_bytes, language_hint="en")
        engine = get_translation_engine()
        trans_res = engine.translate(
            text=ocr_res.text,
            source_lang=ocr_res.detected_language,
            target_lang="hi",
        )
        self.assertGreater(len(trans_res.translated_text), 0)
        self.assertIn(trans_res.target_language, ("hi", "hin_Deva"))


if __name__ == "__main__":
    unittest.main()
