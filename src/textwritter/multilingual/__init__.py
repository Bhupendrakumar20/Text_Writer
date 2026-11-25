"""Multilingual AI Handwriting Generation Package for Text_Writter."""

from .detector import DetectionResult, detect_language_and_script
from .model_registry import REGISTRY, ModelInfo, ModelRegistry
from .ocr import OCRResult, extract_text_from_image
from .registry import SUPPORTED_LANGUAGES, LanguageConfig, get_font_for_language, get_language_config
from .shaper import ShapedGlyph, ShapedRun, render_shaped_line_to_mask, shape_multilingual_text
from .style_analyzer import HandwritingStyleProfile, analyze_handwriting_references, analyze_single_sample
from .style_transfer import StyleTransferReport, plan_style_transfer
from .synthesis_engine import synthesize_handwriting_page
from .validator import QualityValidationResult, validate_text_accuracy

__all__ = [
    "SUPPORTED_LANGUAGES",
    "LanguageConfig",
    "get_language_config",
    "get_font_for_language",
    "DetectionResult",
    "detect_language_and_script",
    "HandwritingStyleProfile",
    "analyze_single_sample",
    "analyze_handwriting_references",
    "StyleTransferReport",
    "plan_style_transfer",
    "QualityValidationResult",
    "validate_text_accuracy",
    "ShapedGlyph",
    "ShapedRun",
    "shape_multilingual_text",
    "render_shaped_line_to_mask",
    "synthesize_handwriting_page",
    "ModelInfo",
    "ModelRegistry",
    "REGISTRY",
    "OCRResult",
    "extract_text_from_image",
]
