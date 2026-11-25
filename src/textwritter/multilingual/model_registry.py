"""Modular Model Registry for Multilingual Handwriting Generation.

Provides a pluggable model provider architecture separating:
1. Multilingual Style Synthesis Engine (all 10 languages, CPU-safe, reference-based)
2. Neural HWT Model (English IAM transformer, CPU/GPU)
3. Neural One-DM Diffusion Model (English single-shot mimicker, CUDA GPU)

Allows easily adding future language-specific GAN or diffusion checkpoints
without touching core application code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

from .registry import SUPPORTED_LANGUAGES, LanguageConfig, get_language_config
from .style_analyzer import HandwritingStyleProfile, analyze_handwriting_references
from .style_transfer import StyleTransferReport
from .synthesis_engine import synthesize_handwriting_page
from .validator import QualityValidationResult


@dataclass
class ModelInfo:
    model_id: str
    name: str
    description: str
    supported_languages: List[str]  # e.g. ["en", "hi", "ta", ...] or ["en"]
    supports_reference_style: bool
    supports_cross_script: bool
    requires_cuda: bool
    is_available: bool
    license: str
    citation_notes: str


class ModelRegistry:
    """Registry managing handwriting generation models and dispatching."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelInfo] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        all_langs = list(SUPPORTED_LANGUAGES.keys())

        # 1. Primary Multilingual Style Synthesis Engine
        self._models["multilingual_style"] = ModelInfo(
            model_id="multilingual_style",
            name="Multilingual Style Replication Engine",
            description="Reference-conditioned handwriting synthesis with HarfBuzz shaping, distance-transform stroke modeling, and affine shear.",
            supported_languages=all_langs,
            supports_reference_style=True,
            supports_cross_script=True,
            requires_cuda=False,
            is_available=True,
            license="Apache 2.0 / OFL",
            citation_notes="Native Text_Writter Multilingual Engine (HarfBuzz + FreeType + Biometric Style Transfer).",
        )

        # 2. HWT (Handwriting Transformers)
        self._models["hwt"] = ModelInfo(
            model_id="hwt",
            name="HWT (Handwriting Transformers)",
            description="Pretrained transformer generating English lines word-by-word conditioned on IAM style crops.",
            supported_languages=["en"],
            supports_reference_style=True,
            supports_cross_script=False,
            requires_cuda=False,
            is_available=True,
            license="MIT / Academic (IAM Dataset)",
            citation_notes="Bhunia et al., 'Handwriting Transformers', ICCV 2021.",
        )

        # 3. One-DM (One-Shot Diffusion Mimicker)
        import torch
        cuda_avail = torch.cuda.is_available()
        self._models["onedm"] = ModelInfo(
            model_id="onedm",
            name="One-DM (Diffusion Mimicker)",
            description="Latent diffusion model for high-fidelity English handwriting replication from a single style photo.",
            supported_languages=["en"],
            supports_reference_style=True,
            supports_cross_script=False,
            requires_cuda=True,
            is_available=cuda_avail,
            license="Academic / Non-commercial",
            citation_notes="One-DM: One-Shot Diffusion Mimicker for Character Synthesis.",
        )

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self._models.get(model_id)

    def get_compatible_models(self, language_code: str) -> List[ModelInfo]:
        """Return all models that can generate text in the specified language."""
        return [
            m for m in self._models.values()
            if m.is_available and (language_code in m.supported_languages or "all" in m.supported_languages)
        ]

    def select_best_model(self, language_code: str, preferred_engine: Optional[str] = None) -> ModelInfo:
        """Auto-select the most capable model for a given language."""
        if preferred_engine and preferred_engine in self._models:
            m = self._models[preferred_engine]
            if language_code in m.supported_languages and m.is_available:
                return m

        # For non-English languages, multilingual_style is the primary robust engine
        if language_code != "en":
            return self._models["multilingual_style"]

        # For English, if neural engine requested and available
        if preferred_engine in ("hwt", "onedm"):
            m = self._models[preferred_engine]
            if m.is_available:
                return m

        return self._models["multilingual_style"]


# Global singleton
REGISTRY = ModelRegistry()
