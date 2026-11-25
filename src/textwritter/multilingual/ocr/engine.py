"""Multi-Tier Optical Character Recognition (OCR) Engine.

Supports printed and handwritten text extraction across English, Hindi,
Gujarati, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Punjabi,
Welsh, and other languages.

Architecture:
- Tier 1: Local Neural EasyOCR with Reader caching, OpenCV pre-processing,
          and median-spacing layout/paragraph reconstruction.
- Tier 2: Neural OCR Gateway for languages without local weights (e.g. Gujarati,
          Punjabi, Malayalam) or graceful fallback.
- Auto Script & Language Detection: Uses Unicode script classification to detect
          source language and confidence score.
"""
from __future__ import annotations

import io
import os
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from textwritter.multilingual.detector import detect_language_and_script


@dataclass
class OCRResult:
    """Structured result returned by OCR extraction."""
    text: str
    detected_language: str
    detected_script: str
    confidence: float
    line_count: int
    word_count: int
    execution_time: float
    engine_used: str
    paragraphs: List[str] = field(default_factory=list)
    diagnostic_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "detected_language": self.detected_language,
            "detected_script": self.detected_script,
            "confidence": round(self.confidence, 3),
            "line_count": self.line_count,
            "word_count": self.word_count,
            "execution_time": round(self.execution_time, 2),
            "engine_used": self.engine_used,
            "paragraphs": self.paragraphs,
            "diagnostic_notes": self.diagnostic_notes,
        }


# EasyOCR supported languages mapping (local)
EASYOCR_LANG_MAP: Dict[str, List[str]] = {
    "en": ["en"],
    "hi": ["hi", "en"],
    "mr": ["mr", "hi", "en"],
    "ta": ["ta", "en"],
    "te": ["te", "en"],
    "kn": ["kn", "en"],
    "bn": ["bn", "en"],
    "sa": ["sa", "en"],
    "cy": ["cy", "en"],
    "auto": ["en", "hi"],  # default fast dual reader for auto
}

# OCR.space 3-letter codes for gateway fallback
OCR_SPACE_LANG_MAP: Dict[str, str] = {
    "en": "eng",
    "hi": "hin",
    "gu": "guj",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
    "bn": "ben",
    "pa": "pan",
    "mr": "mar",
    "sa": "san",
    "cy": "wel",
    "ar": "ara",
    "ur": "urd",
    "zh": "chs",
    "es": "spa",
    "fr": "fre",
    "de": "ger",
    "auto": "eng",
}

# Global Reader Cache for EasyOCR instances
_EASYOCR_CACHE: Dict[Tuple[str, ...], Any] = {}


def _get_easyocr_reader(langs: List[str]) -> Any:
    """Retrieve or initialize a cached EasyOCR reader."""
    import easyocr
    key = tuple(sorted(langs))
    if key not in _EASYOCR_CACHE:
        # gpu=False for safe CPU execution (and GPU if torch cuda is active)
        import torch
        has_cuda = torch.cuda.is_available()
        _EASYOCR_CACHE[key] = easyocr.Reader(list(key), gpu=has_cuda)
    return _EASYOCR_CACHE[key]


def preprocess_image(image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Normalize and enhance image contrast for higher OCR accuracy on handwriting."""
    # Convert input to OpenCV numpy array (BGR or Grayscale)
    if isinstance(image_input, (str, Path)):
        img_np = cv2.imread(str(image_input))
        if img_np is None:
            pil_img = Image.open(str(image_input))
            img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_np is None:
            pil_img = Image.open(io.BytesIO(image_input))
            img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, Image.Image):
        img_np = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        img_np = image_input
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    # Ensure 3-channel BGR
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Apply gentle contrast normalization
    try:
        lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return img_np


def reconstruct_paragraphs_from_boxes(results: List[Any]) -> Tuple[str, List[str], float]:
    """Reconstruct lines and paragraphs from OCR bounding boxes using median vertical spacing.
    
    Returns:
        (full_text, list_of_paragraphs, average_confidence)
    """
    if not results:
        return "", [], 0.0

    items = []
    confs = []
    for bbox, text, conf in results:
        clean = text.strip()
        if not clean:
            continue
        confs.append(float(conf))
        y_center = (bbox[0][1] + bbox[2][1]) / 2.0
        x_left = min(bbox[0][0], bbox[3][0])
        height = max(1.0, abs(bbox[2][1] - bbox[0][1]))
        items.append({
            "text": clean,
            "x": x_left,
            "y": y_center,
            "h": height,
            "conf": conf,
        })

    if not items:
        return "", [], 0.0

    # Sort vertically
    items.sort(key=lambda it: it["y"])

    # Group into lines
    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []
    current_y: Optional[float] = None

    for it in items:
        if current_y is None:
            current_line.append(it)
            current_y = it["y"]
        else:
            if abs(it["y"] - current_y) <= max(14.0, it["h"] * 0.7):
                current_line.append(it)
                current_y = sum(x["y"] for x in current_line) / len(current_line)
            else:
                lines.append(current_line)
                current_line = [it]
                current_y = it["y"]
    if current_line:
        lines.append(current_line)

    # Sort each line horizontally
    for line in lines:
        line.sort(key=lambda it: it["x"])

    # Compute typical vertical spacing between lines for paragraph break detection
    line_ys = [sum(it["y"] for it in line) / len(line) for line in lines]
    spacings = [line_ys[i+1] - line_ys[i] for i in range(len(line_ys) - 1)]
    if not spacings:
        typical_spacing = 25.0
    elif len(spacings) <= 2:
        typical_spacing = min(spacings)
    else:
        # Use lower half median to prevent large paragraph gaps from inflating line spacing
        sorted_spacings = sorted(spacings)
        typical_spacing = statistics.median(sorted_spacings[: max(1, len(sorted_spacings) // 2 + 1)])

    paragraphs: List[str] = []
    current_para_lines: List[str] = []

    for i, line in enumerate(lines):
        line_text = " ".join(it["text"] for it in line if it["text"])
        if i > 0:
            gap = line_ys[i] - line_ys[i-1]
            if gap >= typical_spacing * 1.5:
                # Paragraph break
                if current_para_lines:
                    paragraphs.append("\n".join(current_para_lines))
                    current_para_lines = []
        current_para_lines.append(line_text)

    if current_para_lines:
        paragraphs.append("\n".join(current_para_lines))

    full_text = "\n\n".join(paragraphs)
    avg_conf = sum(confs) / len(confs) if confs else 0.8
    return full_text, paragraphs, avg_conf


def _run_gateway_ocr(image_bytes: bytes, lang_code: str = "auto") -> Tuple[str, List[str], float]:
    """Tier 2: Call neural OCR gateway (OCR.space free engine) for languages like Gujarati, Punjabi, Malayalam."""
    import requests

    ocr_lang = OCR_SPACE_LANG_MAP.get(lang_code, "eng")
    url = "https://api.ocr.space/parse/image"
    payload = {
        "apikey": "helloworld",
        "language": ocr_lang,
        "isOverlayRequired": False,
        "detectOrientation": True,
        "scale": True,
        "OCREngine": "2",  # Engine 2 specialized for non-Latin and handwriting
    }

    resp = requests.post(
        url,
        files={"file": ("input.png", image_bytes, "image/png")},
        data=payload,
        timeout=15,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"OCR Gateway returned HTTP {resp.status_code}")

    data = resp.json()
    if data.get("IsErroredOnProcessing"):
        err = data.get("ErrorMessage", ["OCR error"])[0]
        raise RuntimeError(f"OCR Gateway processing error: {err}")

    parsed_results = data.get("ParsedResults", [])
    if not parsed_results:
        return "", [], 0.0

    raw_text = parsed_results[0].get("ParsedText", "").strip()
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Group into paragraphs
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    full_text = "\n\n".join(raw_paras) if raw_paras else raw_text
    return full_text, raw_paras, 0.90


def extract_text_from_image(
    image_input: Union[str, Path, bytes, BinaryIO, Image.Image],
    language_hint: str = "auto",
    enhance_contrast: bool = True,
) -> OCRResult:
    """Extract printed or handwritten text from an image with line and paragraph preservation.

    Args:
        image_input: Image path, bytes, file object, or PIL Image.
        language_hint: Language hint ('auto', 'en', 'hi', 'ta', 'te', 'gu', 'kn', 'ml', 'bn', 'mr', 'pa', etc.).
        enhance_contrast: Whether to apply CLAHE contrast enhancement for handwriting.

    Returns:
        OCRResult with extracted text, detected language/script, paragraph structure, and metrics.
    """
    t0 = time.time()
    clean_hint = language_hint.strip().lower() if language_hint else "auto"

    # Read bytes if file object
    if hasattr(image_input, "read"):
        image_bytes = image_input.read()
    elif isinstance(image_input, (str, Path)):
        with open(str(image_input), "rb") as f:
            image_bytes = f.read()
    elif isinstance(image_input, bytes):
        image_bytes = image_input
    elif isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="PNG")
        image_bytes = buf.getvalue()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    full_text = ""
    paragraphs: List[str] = []
    confidence = 0.85
    engine_used = "EasyOCR (Local Neural)"
    diagnostic_notes = ""

    # Check if requested language requires Gateway (e.g. Gujarati, Punjabi, Malayalam not in local EasyOCR)
    needs_gateway = clean_hint in ("gu", "pa", "ml")

    if not needs_gateway:
        try:
            # Preprocess image
            processed_np = preprocess_image(image_bytes) if enhance_contrast else None
            feed_img = processed_np if processed_np is not None else image_bytes

            # Determine EasyOCR languages
            easy_langs = EASYOCR_LANG_MAP.get(clean_hint, ["en", "hi"])
            reader = _get_easyocr_reader(easy_langs)
            raw_boxes = reader.readtext(feed_img)

            full_text, paragraphs, confidence = reconstruct_paragraphs_from_boxes(raw_boxes)
            engine_used = f"EasyOCR ({'+'.join(easy_langs)})"

        except Exception as e:
            diagnostic_notes = f"Local OCR fallback triggered: {e}"
            needs_gateway = True

    # Fallback or primary Gateway for specialized Indic languages
    if needs_gateway or not full_text.strip():
        try:
            g_text, g_paras, g_conf = _run_gateway_ocr(image_bytes, clean_hint)
            if g_text.strip():
                full_text = g_text
                paragraphs = g_paras
                confidence = g_conf
                engine_used = f"Neural OCR Gateway ({clean_hint})"
        except Exception as e2:
            if not full_text.strip():
                full_text = ""
                diagnostic_notes += f"; Gateway error: {e2}"

    # Unicode normalization
    full_text = unicodedata.normalize("NFC", full_text).strip()

    # Script & Language Detection on Extracted Text
    detect_res = detect_language_and_script(full_text)
    detected_lang = detect_res.primary_language
    detected_script = detect_res.primary_script

    # If user provided a specific language hint other than auto and text was found, honor it
    if clean_hint != "auto" and clean_hint in OCR_SPACE_LANG_MAP:
        detected_lang = clean_hint

    # Compute line and word count
    lines = [l for l in full_text.splitlines() if l.strip()]
    words = full_text.split()

    elapsed = time.time() - t0

    return OCRResult(
        text=full_text,
        detected_language=detected_lang,
        detected_script=detected_script,
        confidence=confidence,
        line_count=len(lines),
        word_count=len(words),
        execution_time=elapsed,
        engine_used=engine_used,
        paragraphs=paragraphs,
        diagnostic_notes=diagnostic_notes,
    )
