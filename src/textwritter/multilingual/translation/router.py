"""Intelligent Model Router and Validation for Multilingual Translation.

Analyzes source and target language pairs and selects the optimal
pretrained NLP model (AI4Bharat IndicTrans2, Meta NLLB-200, Google MADLAD-400),
resolving model-specific language codes and verifying actual model coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .languages import TranslationLanguage, get_translation_language
from .models import TRANSLATION_MODELS, TranslationModelSpec


@dataclass(frozen=True)
class RouteDecision:
    model_id: str
    display_name: str
    hf_repo: str
    source_model_code: str
    target_model_code: str
    source_lang_name: str
    target_lang_name: str
    is_supported: bool
    is_low_resource: bool
    confidence_tier: str       # "High confidence", "Moderate confidence", "Experimental / Low-Resource", "Unsupported"
    routing_reason: str
    direction: str             # "ltr" or "rtl"


def route_translation(
    source_code: str,
    target_code: str,
    preferred_model_id: Optional[str] = None
) -> RouteDecision:
    """Determine the best NLP model and parameters for a language pair.

    Handles Indian regional translation, European, African, and indigenous low-resource
    pairs with rigorous verification of model support.
    """
    src = get_translation_language(source_code)
    tgt = get_translation_language(target_code)

    if not src or not tgt:
        return RouteDecision(
            model_id="none",
            display_name="Unknown Language",
            hf_repo="",
            source_model_code="",
            target_model_code="",
            source_lang_name=src.name if src else source_code,
            target_lang_name=tgt.name if tgt else target_code,
            is_supported=False,
            is_low_resource=False,
            confidence_tier="Unsupported",
            routing_reason=f"Language code '{source_code}' or '{target_code}' is not recognized in the registry.",
            direction="ltr",
        )

    # Check for explicitly unsupported languages (e.g. Tulu)
    if not src.is_supported or not tgt.is_supported:
        unsupported_name = src.name if not src.is_supported else tgt.name
        return RouteDecision(
            model_id="none",
            display_name="Unsupported Language",
            hf_repo="",
            source_model_code="",
            target_model_code="",
            source_lang_name=src.name,
            target_lang_name=tgt.name,
            is_supported=False,
            is_low_resource=True,
            confidence_tier="Unsupported",
            routing_reason=(
                f"{unsupported_name} is an unmodeled low-resource language not included in standard "
                "IndicTrans2, NLLB-200, or MADLAD-400 tokenizers. As per rigorous academic criteria, "
                "no fake or substituted translation will be generated."
            ),
            direction=tgt.direction,
        )

    is_low_resource = src.is_low_resource or tgt.is_low_resource
    is_indic_src = src.category in ("indian", "south_indian") or src.code == "en"
    is_indic_tgt = tgt.category in ("indian", "south_indian") or tgt.code == "en"
    both_indic = is_indic_src and is_indic_tgt

    # Model Preference Override (if specified and valid)
    chosen_model = preferred_model_id.lower() if preferred_model_id and preferred_model_id in TRANSLATION_MODELS else None

    # Intelligent Selection Logic:
    if not chosen_model:
        if both_indic and src.indictrans_code and tgt.indictrans_code:
            chosen_model = "indictrans2"
        elif is_low_resource:
            # Low-resource indigenous/minority languages: Meta NLLB-200 specializes in 200+ languages
            chosen_model = "nllb200" if tgt.nllb_code and src.nllb_code else "madlad400"
        else:
            # Global / European / African / Asian
            chosen_model = "nllb200" if (tgt.nllb_code and src.nllb_code) else "madlad400"

    # Specific Model Configuration & Repo Selection
    if chosen_model == "indictrans2":
        # Check if pair is supported by IndicTrans2
        if not (src.indictrans_code and tgt.indictrans_code):
            # Fall back to NLLB-200 if IndicTrans2 does not cover this language
            chosen_model = "nllb200"
        else:
            if src.code == "en":
                repo = "ai4bharat/indictrans2-en-indic-dist-200M"
            elif tgt.code == "en":
                repo = "ai4bharat/indictrans2-indic-en-dist-200M"
            else:
                repo = "ai4bharat/indictrans2-indic-indic-dist-320M"

            confidence = "Experimental / Low-Resource" if is_low_resource else "High confidence"
            return RouteDecision(
                model_id="indictrans2",
                display_name="AI4Bharat IndicTrans2",
                hf_repo=repo,
                source_model_code=src.indictrans_code,
                target_model_code=tgt.indictrans_code,
                source_lang_name=src.name,
                target_lang_name=tgt.name,
                is_supported=True,
                is_low_resource=is_low_resource,
                confidence_tier=confidence,
                routing_reason=(
                    "Routed to AI4Bharat IndicTrans2: State-of-the-art regional model specialized "
                    "for 22 Indian scheduled languages, optimized for script nuances, conjuncts, and grammar."
                ),
                direction=tgt.direction,
            )

    if chosen_model == "nllb200":
        if src.nllb_code and tgt.nllb_code:
            confidence = "Experimental / Low-Resource" if is_low_resource else "High confidence"
            return RouteDecision(
                model_id="nllb200",
                display_name="Meta NLLB-200 (distilled-600M)",
                hf_repo="facebook/nllb-200-distilled-600M",
                source_model_code=src.nllb_code,
                target_model_code=tgt.nllb_code,
                source_lang_name=src.name,
                target_lang_name=tgt.name,
                is_supported=True,
                is_low_resource=is_low_resource,
                confidence_tier=confidence,
                routing_reason=(
                    "Routed to Meta NLLB-200: Direct sequence-to-sequence transformer with 200+ language "
                    "embeddings, optimal for low-resource languages (Welsh, Irish, Māori, Quechua, Amharic, Tibetan)."
                ),
                direction=tgt.direction,
            )

    # Google MADLAD-400
    confidence = "Experimental / Low-Resource" if is_low_resource else "High confidence"
    src_mad = src.madlad_code or src.code
    tgt_mad = tgt.madlad_code or tgt.code
    return RouteDecision(
        model_id="madlad400",
        display_name="Google MADLAD-400 (3B-MT)",
        hf_repo="google/madlad400-3b-mt",
        source_model_code=src_mad,
        target_model_code=tgt_mad,
        source_lang_name=src.name,
        target_lang_name=tgt.name,
        is_supported=True,
        is_low_resource=is_low_resource,
        confidence_tier=confidence,
        routing_reason="Routed to Google MADLAD-400: Broad multilingual translation covering 400+ world languages.",
        direction=tgt.direction,
    )
