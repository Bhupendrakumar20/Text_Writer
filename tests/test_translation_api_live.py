"""Integration test verifying live FastAPI translation endpoints on running server."""
import json
import urllib.parse
import urllib.request
import unittest


class TestLiveTranslationAPI(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def test_get_languages(self):
        url = f"{self.BASE_URL}/api/translate/languages"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("categories", data)
            self.assertIn("grouped_languages", data)
            self.assertIn("all_languages", data)
            langs = [l["code"] for l in data["all_languages"]]
            self.assertIn("ta", langs)
            self.assertIn("te", langs)
            self.assertIn("kn", langs)
            self.assertIn("ml", langs)
            self.assertIn("hi", langs)
            self.assertIn("cy", langs)
            self.assertIn("sa", langs)
            print("[LIVE API] GET /api/translate/languages: Verified 200 OK.")

    def test_get_models(self):
        url = f"{self.BASE_URL}/api/translate/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            models = json.loads(resp.read().decode("utf-8"))
            ids = [m["id"] for m in models]
            self.assertIn("indictrans2", ids)
            self.assertIn("nllb200", ids)
            self.assertIn("madlad400", ids)
            print("[LIVE API] GET /api/translate/models: Verified 200 OK.")

    def test_translate_english_to_tamil(self):
        url = f"{self.BASE_URL}/api/translate"
        payload = urllib.parse.urlencode({
            "text": "Education is the most powerful weapon which you can use to change the world.",
            "sourceLanguage": "en",
            "targetLanguage": "ta",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["targetLanguage"], "ta")
            self.assertTrue(len(data["translatedText"]) > 5)
            self.assertEqual(data["translationStatus"], "Success")
            print("[LIVE API] POST /api/translate (EN -> TA):", data["translatedText"])

    def test_translate_english_to_welsh(self):
        url = f"{self.BASE_URL}/api/translate"
        payload = urllib.parse.urlencode({
            "text": "Knowledge is freedom.",
            "sourceLanguage": "en",
            "targetLanguage": "cy",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["targetLanguage"], "cy")
            self.assertTrue(data["isLowResource"])
            print("[LIVE API] POST /api/translate (EN -> CY):", data["translatedText"])

    def test_handwriting_generation_from_translated_text(self):
        """Verify handwriting engine can render translated text into notebook page."""
        url = f"{self.BASE_URL}/api/multilingual/generate"
        payload = urllib.parse.urlencode({
            "text": "கல்வி ஒரு சக்திவாய்ந்த ஆயுதம்",
            "language": "ta",
            "paper_type": "ruled",
            "font_size": 36,
            "reference_filename": "sample_cursive_blue.webp",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("image_url", data)
            print("[LIVE API] POST /api/multilingual/generate with Tamil translation:", data["image_url"])


if __name__ == "__main__":
    unittest.main()
