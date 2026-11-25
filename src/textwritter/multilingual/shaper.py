"""Complex Text Shaping & Rasterization for Multilingual Scripts.

Leverages uharfbuzz (HarfBuzz) and freetype-py to achieve 100% accurate OpenType
conjunct formation, vowel mark reordering, ligature substitution, and glyph positioning
for Indic scripts (Hindi, Tamil, Telugu, Bengali, Gujarati, Punjabi, Malayalam, Kannada)
and Latin (English).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import freetype
import numpy as np
from PIL import Image
import uharfbuzz as hb

from .detector import detect_language_and_script
from .registry import get_font_for_language, get_language_config, SCRIPT_TO_LANGUAGE

SCRIPT_TO_HB_TAG = {
    "Latin": "Latn",
    "Devanagari": "Deva",
    "Tamil": "Taml",
    "Telugu": "Telu",
    "Bengali": "Beng",
    "Gujarati": "Gujr",
    "Gurmukhi": "Guru",
    "Malayalam": "Mlym",
    "Kannada": "Knda",
}

# Cache HarfBuzz face & FreeType face objects to avoid repeated disk reads
_HB_FACE_CACHE: Dict[str, hb.Face] = {}
_FT_FACE_CACHE: Dict[str, freetype.Face] = {}


def _get_hb_face(font_path: Path) -> hb.Face:
    key = str(font_path.resolve())
    if key not in _HB_FACE_CACHE:
        with open(font_path, "rb") as f:
            _HB_FACE_CACHE[key] = hb.Face(f.read())
    return _HB_FACE_CACHE[key]


def _get_ft_face(font_path: Path) -> freetype.Face:
    key = str(font_path.resolve())
    if key not in _FT_FACE_CACHE:
        _FT_FACE_CACHE[key] = freetype.Face(str(font_path.resolve()))
    return _FT_FACE_CACHE[key]


@dataclass
class ShapedGlyph:
    glyph_id: int
    cluster: int
    x_advance: float
    y_advance: float
    x_offset: float
    y_offset: float
    bitmap: Optional[np.ndarray] = None
    bitmap_left: int = 0
    bitmap_top: int = 0


@dataclass
class ShapedRun:
    text: str
    script: str
    font_path: Path
    font_size: int
    glyphs: List[ShapedGlyph]
    total_advance_x: float
    ascent: float
    descent: float


def shape_text_run(
    text: str,
    font_path: Path,
    font_size: int = 36,
    script_name: Optional[str] = None,
) -> ShapedRun:
    """Shape a single-script text string into positioned glyphs via HarfBuzz."""
    hb_face = _get_hb_face(font_path)
    hb_font = hb.Font(hb_face)
    scale = font_size * 64
    hb_font.scale = (scale, scale)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    if script_name and script_name in SCRIPT_TO_HB_TAG:
        buf.script = SCRIPT_TO_HB_TAG[script_name]

    hb.shape(hb_font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    ft_face = _get_ft_face(font_path)
    ft_face.set_char_size(font_size * 64)
    # FT metrics are in 1/64th pt
    metrics = ft_face.size
    ascent = metrics.ascender / 64.0
    descent = abs(metrics.descender / 64.0)

    shaped_glyphs: List[ShapedGlyph] = []
    total_advance = 0.0

    for info, pos in zip(infos, positions):
        x_adv = pos.x_advance / 64.0
        y_adv = pos.y_advance / 64.0
        x_off = pos.x_offset / 64.0
        y_off = pos.y_offset / 64.0
        total_advance += x_adv

        shaped_glyphs.append(
            ShapedGlyph(
                glyph_id=info.codepoint,
                cluster=info.cluster,
                x_advance=x_adv,
                y_advance=y_adv,
                x_offset=x_off,
                y_offset=y_off,
            )
        )

    return ShapedRun(
        text=text,
        script=script_name or "Unknown",
        font_path=font_path,
        font_size=font_size,
        glyphs=shaped_glyphs,
        total_advance_x=total_advance,
        ascent=ascent,
        descent=descent,
    )


def rasterize_glyph(font_path: Path, font_size: int, glyph_id: int) -> Tuple[np.ndarray, int, int]:
    """Rasterize a single glyph ID using FreeType with anti-aliasing."""
    ft_face = _get_ft_face(font_path)
    ft_face.set_char_size(font_size * 64)

    # Load glyph with render mode normal (8-bit grayscale antialiased)
    try:
        ft_face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
        slot = ft_face.glyph
        bitmap = slot.bitmap
        width = bitmap.width
        rows = bitmap.rows

        if width > 0 and rows > 0:
            arr = np.array(bitmap.buffer, dtype=np.uint8).reshape((rows, width))
        else:
            arr = np.zeros((0, 0), dtype=np.uint8)

        return arr, slot.bitmap_left, slot.bitmap_top
    except Exception:
        return np.zeros((0, 0), dtype=np.uint8), 0, 0


def render_shaped_line_to_mask(
    runs: List[ShapedRun],
    extra_spacing: float = 0.0,
    vertical_jitter_func=None,
) -> Tuple[np.ndarray, float, float]:
    """Render a sequence of shaped runs onto a 2D float32/uint8 alpha mask.

    Returns:
        (mask_array, total_width, baseline_y)
    """
    if not runs:
        return np.zeros((10, 10), dtype=np.uint8), 10.0, 5.0

    max_ascent = max(r.ascent for r in runs)
    max_descent = max(r.descent for r in runs)

    pad_top = int(max_ascent + 20)
    pad_bottom = int(max_descent + 20)
    line_h = pad_top + pad_bottom
    baseline_y = pad_top

    # Calculate total width
    total_w = sum(r.total_advance_x for r in runs) + (len(runs) - 1) * extra_spacing + 50
    canvas_w = max(int(total_w), 50)

    mask = np.zeros((line_h, canvas_w), dtype=np.float32)

    cursor_x = 10.0

    for run_idx, run in enumerate(runs):
        for glyph_idx, g in enumerate(run.glyphs):
            if g.glyph_id == 0 and g.x_advance > 0:
                # Space character
                cursor_x += g.x_advance + extra_spacing
                continue

            bm, bm_left, bm_top = rasterize_glyph(run.font_path, run.font_size, g.glyph_id)
            if bm.size > 0:
                h, w = bm.shape
                jitter_y = vertical_jitter_func(cursor_x) if vertical_jitter_func else 0.0

                gx = int(cursor_x + g.x_offset + bm_left)
                gy = int(baseline_y - g.y_offset - bm_top + jitter_y)

                # Safe blit into mask
                if 0 <= gx < canvas_w and 0 <= gy < line_h:
                    x0 = max(0, gx)
                    y0 = max(0, gy)
                    x1 = min(canvas_w, gx + w)
                    y1 = min(line_h, gy + h)

                    bm_x0 = x0 - gx
                    bm_y0 = y0 - gy
                    bm_x1 = bm_x0 + (x1 - x0)
                    bm_y1 = bm_y0 + (y1 - y0)

                    target_region = mask[y0:y1, x0:x1]
                    source_region = bm[bm_y0:bm_y1, bm_x0:bm_x1].astype(np.float32)

                    # Maximum composition for smooth anti-aliased overlaps
                    mask[y0:y1, x0:x1] = np.maximum(target_region, source_region)

            cursor_x += g.x_advance + extra_spacing

    return np.clip(mask, 0, 255).astype(np.uint8), cursor_x, float(baseline_y)


def shape_multilingual_text(
    text: str,
    target_lang: str = "auto",
    font_size: int = 36,
    max_line_width: Optional[float] = None,
    font_family: Optional[str] = None,
) -> List[List[ShapedRun]]:
    """Split text into lines and shape each line preserving mixed scripts, with natural word wrapping."""
    if target_lang == "auto":
        detection = detect_language_and_script(text)
        primary_lang = detection.primary_language
    else:
        primary_lang = target_lang

    paragraphs = text.splitlines()
    if not paragraphs:
        paragraphs = [""]

    shaped_lines: List[List[ShapedRun]] = []

    for para in paragraphs:
        if not para.strip():
            # Empty paragraph = blank ruled line
            cfg = get_language_config(primary_lang)
            fp = get_font_for_language(primary_lang, font_family=font_family)
            empty_run = shape_text_run(" ", fp, font_size, cfg.script if cfg else "Latin")
            shaped_lines.append([empty_run])
            continue

        if max_line_width is None:
            # Direct single line mode (used by unit tests for phrase shaping)
            detection = detect_language_and_script(para, default_lang=primary_lang)
            line_runs: List[ShapedRun] = []
            for seg_text, seg_script in detection.script_segments:
                seg_lang = SCRIPT_TO_LANGUAGE.get(seg_script, primary_lang)
                font_path = get_font_for_language(seg_lang, font_family=font_family)
                run = shape_text_run(seg_text, font_path, font_size, seg_script)
                line_runs.append(run)
            shaped_lines.append(line_runs)
            continue

        # Word-wrapping mode: break words and wrap before max_line_width
        words = para.split(" ")
        current_line_runs: List[ShapedRun] = []
        current_line_w = 0.0

        # Jitter wrap target by ±15px per line for authentic human ragged right margin
        import random
        line_rng = random.Random(hash(para) & 0xFFFFFFFF)
        target_wrap_w = max_line_width + line_rng.uniform(-15.0, 15.0)

        # Space run helper
        space_fp = get_font_for_language(primary_lang, font_family=font_family)
        space_cfg = get_language_config(primary_lang)
        space_script = space_cfg.script if space_cfg else "Latin"
        space_run = shape_text_run(" ", space_fp, font_size, space_script)
        space_w = space_run.total_advance_x

        for word in words:
            if not word:
                continue

            word_detection = detect_language_and_script(word, default_lang=primary_lang)
            word_runs: List[ShapedRun] = []
            word_w = 0.0

            for seg_text, seg_script in word_detection.script_segments:
                seg_lang = SCRIPT_TO_LANGUAGE.get(seg_script, primary_lang)
                font_path = get_font_for_language(seg_lang, font_family=font_family)
                run = shape_text_run(seg_text, font_path, font_size, seg_script)
                word_runs.append(run)
                word_w += run.total_advance_x

            # Check if this word fits on the current line
            needed_w = (space_w if current_line_runs else 0.0) + word_w
            if current_line_runs and (current_line_w + needed_w > target_wrap_w):
                # Wrap to next line
                shaped_lines.append(current_line_runs)
                current_line_runs = list(word_runs)
                current_line_w = word_w
                target_wrap_w = max_line_width + line_rng.uniform(-15.0, 15.0)
            else:
                if current_line_runs:
                    current_line_runs.append(space_run)
                    current_line_w += space_w
                current_line_runs.extend(word_runs)
                current_line_w += word_w

        if current_line_runs:
            shaped_lines.append(current_line_runs)

    return shaped_lines
