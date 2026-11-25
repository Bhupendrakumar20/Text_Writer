"""Multilingual Language & Script Registry for Text_Writter.

Defines supported languages, Unicode script ranges, font selection hierarchy,
sample sentences, and script characteristics for 10 initial languages:
1. English (Latin)
2. Hindi (Devanagari)
3. Tamil (Tamil)
4. Telugu (Telugu)
5. Bengali (Bengali)
6. Gujarati (Gujarati)
7. Marathi (Devanagari)
8. Punjabi (Gurmukhi)
9. Malayalam (Malayalam)
10. Kannada (Kannada)

Extensible architecture allows easily appending new languages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
SYSTEM_FONTS = [
    Path("C:/Windows/Fonts/Nirmala.ttf"),
    Path("C:/Windows/Fonts/NirmalaS.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
]


@dataclass(frozen=True)
class LanguageConfig:
    code: str
    name: str
    native_name: str
    script: str
    unicode_ranges: Tuple[Tuple[int, int], ...]
    primary_font: str
    fallback_fonts: List[str]
    sample_text_short: str
    sample_text_long: str
    has_headline: bool = False  # Devanagari / Bengali shirorekha
    direction: str = "ltr"


SUPPORTED_LANGUAGES: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        code="en",
        name="English",
        native_name="English",
        script="Latin",
        unicode_ranges=((0x0000, 0x007F), (0x0080, 0x00FF), (0x0100, 0x017F)),
        primary_font="Caveat.ttf",
        fallback_fonts=["Kalam.ttf", "PatrickHand.ttf", "IndieFlower.ttf", "Nirmala.ttf"],
        sample_text_short="The quick brown fox jumps over the lazy dog.",
        sample_text_long="Handwriting is a reflection of thought and personality. Authentic handwritten notes retain personal warmth across every curve and stroke.",
        has_headline=False,
    ),
    "hi": LanguageConfig(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        script="Devanagari",
        unicode_ranges=((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
        primary_font="Kalam.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="मेरा नाम भारत है और मैं एक विद्यार्थी हूँ।",
        sample_text_long="सत्य और अहिंसा ही मानव जीवन के सर्वोच्च आदर्श हैं। कठिन परिश्रम और अनुशासन से ही प्रत्येक लक्ष्य को प्राप्त किया जा सकता है।",
        has_headline=True,
    ),
    "ta": LanguageConfig(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        script="Tamil",
        unicode_ranges=((0x0B80, 0x0BFF),),
        primary_font="Kavivanar.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="என் பெயர் பாரத் மற்றும் நான் ஒரு மாணவன்.",
        sample_text_long="கற்க கசடறக் கற்பவை கற்றபின் நிற்க அதற்குத் தக. உண்மை, அன்பு மற்றும் நற்பண்புகளே வாழ்வின் உண்மையான செல்வம் ஆகும்.",
        has_headline=False,
    ),
    "te": LanguageConfig(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        script="Telugu",
        unicode_ranges=((0x0C00, 0x0C7F),),
        primary_font="Dhurjati.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="నా పేరు భారత్ మరియు నేను ఒక విద్యార్థిని.",
        sample_text_long="జ్ఞానము మరియు సత్యము మానవ ప్రగతికి మూలాధారాలు. నిరంతర శ్రమ ద్వారా మాత్రమే విజయము సాధించవచ్చు.",
        has_headline=False,
    ),
    "bn": LanguageConfig(
        code="bn",
        name="Bengali",
        native_name="বাংলা",
        script="Bengali",
        unicode_ranges=((0x0980, 0x09FF),),
        primary_font="Atma.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="আমার নাম ভারত এবং আমি একজন ছাত্র।",
        sample_text_long="চিত্ত যেথা ভয়শূন্য, উচ্চ যেথা শির, জ্ঞান যেথা মুক্ত, যেথা গৃহের প্রাচীর আপন প্রাঙ্গণতলে দিবসশর্বরী বসুধারে রাখে নাই খণ্ড ক্ষুদ্র করি।",
        has_headline=True,
    ),
    "gu": LanguageConfig(
        code="gu",
        name="Gujarati",
        native_name="ગુજરાતી",
        script="Gujarati",
        unicode_ranges=((0x0A80, 0x0AFF),),
        primary_font="Mogra.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="મારું નામ ભારત છે અને હું એક વિદ્યાર્થી છું.",
        sample_text_long="સત્ય અને અહિંસા જીવનના સર્વોચ્ચ મૂલ્યો છે. પરિશ્રમ જ સફળતાની ખરી ચાવી છે અને સમર્પણથી દરેક મુશ્કેલી પાર કરી શકાય છે.",
        has_headline=False,
    ),
    "mr": LanguageConfig(
        code="mr",
        name="Marathi",
        native_name="मराठी",
        script="Devanagari",
        unicode_ranges=((0x0900, 0x097F),),
        primary_font="Kalam.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="माझे नाव भारत आहे आणि मी एक विद्यार्थी आहे.",
        sample_text_long="ध्येय साध्य करण्यासाठी कठोर परिश्रम आणि निरंतर प्रयत्नांची आवश्यकता असते. ज्ञान आणि सुसंस्कार हेच जीवनाचे खरे अलंकार आहेत.",
        has_headline=True,
    ),
    "pa": LanguageConfig(
        code="pa",
        name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        script="Gurmukhi",
        unicode_ranges=((0x0A00, 0x0A7F),),
        primary_font="Nirmala.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="ਮੇਰਾ ਨਾਮ ਭਾਰਤ ਹੈ ਅਤੇ ਮੈਂ ਇੱਕ ਵਿਦਿਆਰਥੀ ਹਾਂ।",
        sample_text_long="ਸੱਚ ਅਤੇ ਮਿਹਨਤ ਹੀ ਸਫਲਤਾ ਦਾ ਮੂਲ ਮੰਤਰ ਹਨ। ਸਦਾ ਨੇਕ ਕੰਮ ਕਰੋ ਅਤੇ ਸਭ ਦਾ ਭਲਾ ਮੰਗੋ।",
        has_headline=True,
    ),
    "ml": LanguageConfig(
        code="ml",
        name="Malayalam",
        native_name="മലയാളം",
        script="Malayalam",
        unicode_ranges=((0x0D00, 0x0D7F),),
        primary_font="Chilanka.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="എന്റെ പേര് ഭാരത് എന്നാണ്, ഞാൻ ഒരു വിദ്യാർത്ഥിയാണ്.",
        sample_text_long="അറിവും സത്യവുമാണ് മനുഷ്യ പുരോഗതിയുടെ അടിസ്ഥാനം. ആത്മാർത്ഥമായ പരിശ്രമത്തിലൂടെ ഏത് ഉയരവും കീഴടക്കാൻ സാധിക്കും.",
        has_headline=False,
    ),
    "kn": LanguageConfig(
        code="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        script="Kannada",
        unicode_ranges=((0x0C80, 0x0CFF),),
        primary_font="Chathura.ttf",
        fallback_fonts=["Nirmala.ttf"],
        sample_text_short="ನನ್ನ ಹೆಸರು ಭಾರತ್ ಮತ್ತು ನಾನು ಒಬ್ಬ ವಿದ್ಯಾರ್ಥಿ.",
        sample_text_long="ಜ್ಞಾನ ಮತ್ತು ಸತ್ಯವೇ ಮನುಕುಲದ ನಿಜವಾದ ಸಂಪತ್ತು. ಕಠಿಣ ಪರಿಶ್ರಮ ಮತ್ತು ಸತತ ಪ್ರಯತ್ನಗಳಿಂದ ಮಾತ್ರವೇ ಯಶಸ್ಸು ಸಾಧ್ಯ.",
        has_headline=False,
    ),
}

# Maps scripts to primary languages
SCRIPT_TO_LANGUAGE = {
    "Latin": "en",
    "Devanagari": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Gurmukhi": "pa",
    "Malayalam": "ml",
    "Kannada": "kn",
}


def get_language_config(code_or_script: str) -> Optional[LanguageConfig]:
    """Retrieve language configuration by ISO code or script name."""
    code = code_or_script.lower()
    if code in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[code]
    for lang in SUPPORTED_LANGUAGES.values():
        if lang.script.lower() == code or lang.name.lower() == code:
            return lang

    # Map extended translation languages by their script family
    try:
        from .translation.languages import get_translation_language
        t_lang = get_translation_language(code)
        if t_lang:
            script_map = {
                "Latin": "en",
                "Devanagari": "hi",
                "Tamil": "ta",
                "Telugu": "te",
                "Malayalam": "ml",
                "Kannada": "kn",
                "Bengali": "bn",
                "Gujarati": "gu",
                "Gurmukhi": "pa",
            }
            mapped_code = script_map.get(t_lang.script)
            if mapped_code and mapped_code in SUPPORTED_LANGUAGES:
                return SUPPORTED_LANGUAGES[mapped_code]
    except Exception:
        pass

    return None


def resolve_font_path(font_filename: str) -> Optional[Path]:
    """Locate font file in assets/fonts/ or OS fonts."""
    # 1. Check local assets/fonts
    p = FONTS_DIR / font_filename
    if p.exists():
        return p
    # 2. Check system font locations
    for sys_font in SYSTEM_FONTS:
        if sys_font.name.lower() == font_filename.lower() and sys_font.exists():
            return sys_font
        if sys_font.exists():
            return sys_font
    return None


FONT_FAMILY_MAP = {
    "caveat": "Caveat.ttf",
    "patrick": "PatrickHand.ttf",
    "indie": "IndieFlower.ttf",
    "kalam": "Kalam.ttf",
}


def get_font_for_language(lang_code: str, font_family: Optional[str] = None) -> Path:
    """Return the best available font path for the language, conditioned on font_family if supplied."""
    cfg = get_language_config(lang_code) or SUPPORTED_LANGUAGES["en"]

    # If font_family override/match is requested and supported
    if font_family:
        norm_fam = font_family.lower()
        if cfg.script == "Latin" and norm_fam in FONT_FAMILY_MAP:
            target_font = FONT_FAMILY_MAP[norm_fam]
            font_path = resolve_font_path(target_font)
            if font_path:
                return font_path
        elif cfg.script == "Devanagari":
            # Kalam is the primary Devanagari handwriting font
            kalam_path = resolve_font_path("Kalam.ttf")
            if kalam_path:
                return kalam_path

    # Try primary font
    font_path = resolve_font_path(cfg.primary_font)
    if font_path:
        return font_path
    # Try fallbacks
    for fb in cfg.fallback_fonts:
        font_path = resolve_font_path(fb)
        if font_path:
            return font_path
    # Fallback to Nirmala or first available font in assets
    for candidate in [FONTS_DIR / "Kalam.ttf", FONTS_DIR / "Caveat.ttf"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No usable font found for language {lang_code}")
