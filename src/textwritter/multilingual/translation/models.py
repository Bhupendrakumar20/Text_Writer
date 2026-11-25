"""Pretrained NLP Translation Model Definitions for Text_Writter Studio.

Provides metadata, architecture specs, and capabilities for:
1. Google MADLAD-400 (google/madlad400-3b-mt)
2. AI4Bharat IndicTrans2 (Indic regional translation)
3. Meta NLLB-200 (facebook/nllb-200-distilled-600M)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TranslationModelSpec:
    model_id: str
    display_name: str
    organization: str
    huggingface_repo: str
    parameter_count: str
    description: str
    supported_tasks: List[str]
    specialization: str
    recommended_for: List[str]
    low_resource_capable: bool
    requires_token: bool
    hf_pipeline_type: str = "translation"


TRANSLATION_MODELS: Dict[str, TranslationModelSpec] = {
    # 1. Primary Model: Google MADLAD-400
    "madlad400": TranslationModelSpec(
        model_id="madlad400",
        display_name="Google MADLAD-400 (3B-MT)",
        organization="Google Research",
        huggingface_repo="google/madlad400-3b-mt",
        parameter_count="3 Billion",
        description="Massive multilingual machine translation model covering 400+ languages including extensive low-resource and regional languages.",
        supported_tasks=["Multilingual Translation", "Low-Resource Language Synthesis"],
        specialization="General multilingual translation and cross-lingual transfer across 400+ languages.",
        recommended_for=["global", "european", "african", "asian", "indigenous"],
        low_resource_capable=True,
        requires_token=True,
    ),

    # 2. Indian Regional Model: AI4Bharat IndicTrans2
    "indictrans2": TranslationModelSpec(
        model_id="indictrans2",
        display_name="AI4Bharat IndicTrans2 (Distilled)",
        organization="AI4Bharat / IIT Madras",
        huggingface_repo="ai4bharat/indictrans2-indic-indic-dist-320M",
        parameter_count="320 Million / 200 Million",
        description="State-of-the-art specialized multilingual model trained on 22 scheduled Indian languages + English, optimized for Indic script nuances and grammar.",
        supported_tasks=["Indic-to-Indic Translation", "English-to-Indic Translation", "Indic-to-English Translation"],
        specialization="High-fidelity translation across 22 official Indian languages and English.",
        recommended_for=["indian", "south_indian"],
        low_resource_capable=True,
        requires_token=True,
    ),

    # 3. Research & Fallback Model: Meta NLLB-200
    "nllb200": TranslationModelSpec(
        model_id="nllb200",
        display_name="Meta NLLB-200 (distilled-600M)",
        organization="Meta AI",
        huggingface_repo="facebook/nllb-200-distilled-600M",
        parameter_count="600 Million",
        description="No Language Left Behind model covering 200+ languages with direct sentence-level cross-attention and low-resource support.",
        supported_tasks=["Direct 200-Language Translation", "Low-Resource Benchmarking"],
        specialization="Deep coverage of indigenous, minority, and low-resource languages worldwide.",
        recommended_for=["indigenous", "african", "european", "south_indian"],
        low_resource_capable=True,
        requires_token=True,
    ),
}


def get_model_spec(model_key: str) -> Optional[TranslationModelSpec]:
    """Retrieve model specification by key (e.g. 'indictrans2', 'nllb200', 'madlad400')."""
    return TRANSLATION_MODELS.get(model_key.lower())


def list_available_models() -> List[Dict]:
    """Return serializable list of available NLP models."""
    return [
        {
            "id": m.model_id,
            "name": m.display_name,
            "organization": m.organization,
            "repo": m.huggingface_repo,
            "params": m.parameter_count,
            "description": m.description,
            "specialization": m.specialization,
            "low_resource_capable": m.low_resource_capable,
        }
        for m in TRANSLATION_MODELS.values()
    ]
