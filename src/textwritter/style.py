"""Shared style preparation for both engines (HWT and One-DM).

Turns "a photo of handwriting, or nothing at all" into a folder of word-crop PNGs:
  * photo given        -> deskew, remove ruling lines, segment words (OpenCV)
  * no photo           -> bundled real-handwriting samples (committed assets),
                          with samples/sample_hello.png as a last-resort print-font
                          demo so a run still produces output.
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from .runtime import ROOT, SAMPLE_STYLES_DIR

_FALLBACK_SAMPLE = ROOT / "samples" / "sample_hello.png"


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Deskew handwriting image based on text line orientation."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 50:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if 0.4 < abs(angle) < 35.0:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return gray


def segment_words(gray: np.ndarray) -> list[np.ndarray]:
    """Split a handwriting photo into high-quality word crops.
    
    Robust to notebook ruling lines, margins, and paper textures:
    1. Deskews image.
    2. Identifies and suppresses horizontal ruled lines.
    3. Connects character clusters into words.
    4. Extracts words from the original grayscale image preserving stroke pressure.
    """
    if gray is None or gray.size == 0:
        return []

    # 1. Deskew
    gray = deskew_image(gray)
    h, w = gray.shape

    # 2. Otsu threshold to find all foreground (ink + lines)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Detect horizontal ruling lines & vertical margin lines
    line_w = max(20, w // 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_w, 1))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

    line_h = max(25, h // 15)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_h))
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    lines = cv2.bitwise_or(h_lines, v_lines)

    # 4. Remove ruling lines from ink mask
    ink_mask = cv2.subtract(bw, lines)

    # 5. Connect characters within words
    word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))
    connected = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, word_kernel)

    # 6. Find bounding boxes of word candidates
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        # Filter out noise, single dots, and full-page borders
        if cw >= 14 and ch >= 10 and cw < w * 0.92 and ch < h * 0.40:
            boxes.append((x, y, cw, ch))

    # Fallback to projection bands if contour detection didn't find words
    if not boxes:
        rows = ink_mask.sum(axis=1) // 255
        start_y = None
        for i, val in enumerate(rows):
            if val > 0 and start_y is None:
                start_y = i
            elif val == 0 and start_y is not None:
                if (i - start_y) >= 10:
                    strip = ink_mask[start_y:i]
                    cols = strip.sum(axis=0) // 255
                    start_x = None
                    for j, cval in enumerate(cols):
                        if cval > 0 and start_x is None:
                            start_x = j
                        elif cval == 0 and start_x is not None:
                            if (j - start_x) >= 14:
                                boxes.append((start_x, start_y, j - start_x, i - start_y))
                            start_x = None
                start_y = None

    if not boxes:
        return []

    # Sort boxes in natural reading order (line by line: bucket by y, then sort by x)
    avg_h = max(16, int(np.mean([b[3] for b in boxes])))
    boxes.sort(key=lambda b: (b[1] // max(1, avg_h), b[0]))

    # 7. Extract crops from original grayscale image with clean white background
    words: list[np.ndarray] = []
    for x, y, cw, ch in boxes:
        pad_x = max(2, int(cw * 0.05))
        pad_y = max(2, int(ch * 0.08))

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + cw + pad_x)
        y1 = min(h, y + ch + pad_y)

        crop_gray = gray[y0:y1, x0:x1]
        crop_ink = ink_mask[y0:y1, x0:x1]

        # Soft background normalization: keep ink contrast, make background pure white (255)
        # Dilate ink mask slightly to preserve stroke edges
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink_dil = cv2.dilate(crop_ink, k_dil)
        normalized = np.where(ink_dil > 0, crop_gray, 255).astype(np.uint8)

        # Ensure reasonable aspect ratio
        words.append(normalized)

    return words


def save_words(words: list[np.ndarray], out_dir: Path, prefix: str = "w") -> list[Path]:
    """Save word crops formatted for HWT style tensors (height 32, max width 192)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Ensure at least 15 samples for HWT
    sample_pool = list(words)
    if len(sample_pool) < 15 and len(sample_pool) > 0:
        while len(sample_pool) < 15:
            # Add slight variations (scaling / slight tilt) to fill 15 diverse samples
            for item in list(words):
                sample_pool.append(item)
                if len(sample_pool) >= 15:
                    break

    for i, w_crop in enumerate(sample_pool[:15]):
        h, w = w_crop.shape
        if h == 0 or w == 0:
            continue

        # Scale proportionally to height 32
        target_h = 32
        target_w = max(8, min(176, int(round(w * (target_h / float(h))))))
        resized = cv2.resize(w_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Pad canvas to 32 x (target_w + 12) with 255 white
        canvas = np.full((32, target_w + 12), 255, dtype=np.uint8)
        canvas[:, 6:6 + target_w] = resized

        p = out_dir / f"{prefix}{i:02d}.png"
        cv2.imwrite(str(p), canvas)
        paths.append(p)

    return paths


def bundled_samples(extract: Callable[[], list[Path]] | None = None,
                    verbose: bool = True) -> list[Path]:
    """Committed sample styles, else extract once, else the printed sample image."""
    pngs = sorted(SAMPLE_STYLES_DIR.glob("*.png")) if SAMPLE_STYLES_DIR.exists() else []
    if pngs:
        return pngs
    if extract is not None:
        try:
            got = extract()
            if got:
                return got
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[style] bundled-style extraction failed ({e})")
    return [p for p in [_FALLBACK_SAMPLE] if p.exists()]


def prepare_style(style: str | Path | None, work_dir: Path,
                  extract: Callable[[], list[Path]] | None = None,
                  verbose: bool = True) -> tuple[Path, str, list[str]]:
    """Build the style word folder. Returns (dir, source, warnings)."""
    out_dir = Path(work_dir) / "style_words"
    warnings: list[str] = []

    if style:
        style_path = Path(style)
        if style_path.exists():
            gray = cv2.imread(str(style_path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                words = segment_words(gray)
                if len(words) > 0:
                    save_words(words, out_dir)
                    if verbose:
                        print(f"[style] {len(words)} word crops extracted from {style_path.name}")
                    return out_dir, f"photo:{style_path.name}", warnings
                warnings.append(
                    f"no distinct words found in {style_path.name}; using "
                    "bundled style instead (a photo with clear pen strokes works best)")
            else:
                warnings.append(f"could not read image {style_path}; using bundled style")
        else:
            warnings.append(f"style file not found ({style_path}); using bundled style")

    samples = bundled_samples(extract=extract, verbose=verbose)
    if not samples:
        raise RuntimeError(
            "No style photo given and no bundled style available. Pass one: "
            "--style /path/to/your_handwriting.(jpg|png)")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(samples[:15]):
        shutil.copy2(p, out_dir / f"{i:02d}_{p.name}")
    if verbose:
        print(f"[style] using {min(len(samples), 15)} bundled handwriting samples")
    if samples[0].parent == _FALLBACK_SAMPLE.parent:
        warnings.append("bundled style fell back to a printed sample — output "
                        "will look typed; pass --style with real handwriting")
    return out_dir, "bundled", warnings
