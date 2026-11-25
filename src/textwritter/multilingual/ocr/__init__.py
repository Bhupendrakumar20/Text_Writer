"""OCR package for Text_Writter Studio."""
from .engine import (
    OCRResult,
    extract_text_from_image,
    preprocess_image,
    reconstruct_paragraphs_from_boxes,
)

__all__ = [
    "OCRResult",
    "extract_text_from_image",
    "preprocess_image",
    "reconstruct_paragraphs_from_boxes",
]
