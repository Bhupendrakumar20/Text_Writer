#!/usr/bin/env python3
"""HWT engine — real handwriting creation that runs on CPU *and* GPU.

This is the "works anywhere" generation path: one small pretrained bundle
(~685 MB, auto-downloaded), a few style word images (a photo of your handwriting,
or the bundled IAM samples when you don't give one), and the text you want.

It drives the official HWT repo's own model (no retraining) through a small
driver script executed inside the vendored clone, because that repo uses
repo-relative imports (`from params import *`). Everything around it — style
preparation, word segmentation, output cropping, judging hand-off — lives here.

Engine contract (shared with 01_onedm/engine.py):
    generate(text, style_dir, out_png, device, ...) -> {"image": Path, "text": str}
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ENGINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ENGINE_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from textwritter.runtime import SAMPLE_STYLES_DIR, detect_environment  # noqa: E402
from textwritter.style import prepare_style as _prepare_style  # noqa: E402

REPO_URL = "https://github.com/ankanbhunia/Handwriting-Transformers"
VENDOR_DIR = _ENGINE_DIR / "_vendor" / "HWT"
GDRIVE_ID = "16g9zgysQnWk7-353_tMig92KsZsrcM6k"  # official bundle (repo README)
DRIVER_NAME = "_tw_hwt_driver.py"


# ---------------------------------------------------------------------------
# Vendor repo + weights (get-or-guide, never a raw crash)
# ---------------------------------------------------------------------------

def _sh(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_gdown() -> None:
    """gdown is in requirements.txt; install it here if the env lacks it."""
    try:
        import gdown  # noqa: F401
        return
    except ImportError:
        pass
    attempts = [
        [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        ["uv", "pip", "install", "--python", sys.executable, "gdown"],
    ]
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True)
            import gdown  # noqa: F401
            return
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("gdown is required to fetch HWT weights — install it: "
                       "pip install gdown")


def ensure_repo(verbose: bool = True) -> Path:
    """Clone HWT and fetch/unzip the official bundle; returns the vendor dir."""
    if not (VENDOR_DIR / "models" / "model.py").exists():
        VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        _sh(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])
    weights = VENDOR_DIR / "files" / "iam_model.pth"
    if not weights.exists():
        import zipfile

        _ensure_gdown()
        import gdown

        zip_path = VENDOR_DIR / "files.zip"
        if not zip_path.exists():
            gdown.download(id=GDRIVE_ID, output=str(zip_path), quiet=False)
        print(f"[hwt] unzipping {zip_path.name}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(VENDOR_DIR)
        zip_path.unlink(missing_ok=True)
        if not weights.exists():
            raise RuntimeError(
                f"HWT weights missing after download: {weights}. Place the "
                "official files.zip contents into "
                f"{VENDOR_DIR / 'files'} (see repo INSTALL.md).")
    # The repo's TRGAN reads these at import; create if the bundle lacked them.
    for name, default in (("mytext.txt", "hello world\n"),
                          ("files/english_words.txt", "hello\nworld\n")):
        p = VENDOR_DIR / name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(default)
    if verbose:
        print(f"[hwt] vendor ready: {VENDOR_DIR}")
    return VENDOR_DIR


def alphabet() -> str:
    """The trained character set (from the vendor params.py, parsed statically)."""
    from textwritter.runtime import read_string_constant

    vendor = ensure_repo(verbose=False)
    return read_string_constant(vendor / "params.py", "ALPHABET")


def filter_text(text: str) -> tuple[str, list[str]]:
    """Drop characters the model was never trained on (warn instead of crash)."""
    allowed = set(alphabet())
    if not allowed:
        return text, []
    kept, dropped = [], []
    for ch in text:
        (kept if ch in allowed or ch in "\n\r\t " else dropped).append(ch)
    warnings = []
    if dropped:
        warnings.append(f"dropped {len(dropped)} unsupported character(s): "
                        f"{''.join(sorted(set(dropped)))!r}")
    return "".join(kept).strip(), warnings


def extract_iam_styles(verbose: bool = True) -> list[Path]:
    """Save a writer's IAM word images (from the official bundle) as bundled styles."""
    import pickle

    ensure_repo(verbose=False)
    data_path = VENDOR_DIR / "files" / "IAM-32.pickle"
    if not data_path.exists():
        return []
    with open(data_path, "rb") as f:
        writers = pickle.load(f)["train"]
    for wid in sorted(writers, key=str):
        items = writers[wid]
        if len(items) < 5:
            continue
        SAMPLE_STYLES_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for k, item in enumerate(items[:15]):
            img = item["img"].convert("L")
            label = re.sub(r"[^A-Za-z0-9]+", "_", str(item.get("label", k)))[:24]
            p = SAMPLE_STYLES_DIR / f"{k:02d}_{label or 'w'}.png"
            img.save(p)
            saved.append(p)
        if verbose:
            print(f"[hwt] extracted bundled style words from IAM writer {wid}")
        return saved
    return []


def prepare_style(style, work_dir: Path, verbose: bool = True):
    """Style word folder — shared logic in textwritter.style."""
    return _prepare_style(style, work_dir, extract=extract_iam_styles, verbose=verbose)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_DRIVER_SRC = r'''
import argparse, json, math, os, random, sys, types
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw

# model.py does `import pandas` but never uses it for inference; stub if absent.
try:
    import pandas  # noqa: F401
except ImportError:
    import importlib.machinery
    _pandas = types.ModuleType("pandas")
    _pandas.__spec__ = importlib.machinery.ModuleSpec("pandas", None)
    sys.modules.setdefault("pandas", _pandas)

from params import *
from models.model import TRGAN
from data.dataset import crop_, get_transform

ap = argparse.ArgumentParser()
ap.add_argument("--style-dir", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--model", default="files/iam_model.pth")
ap.add_argument("--num-samples", type=int, default=15)
ap.add_argument("--paper", default="ruled")
ap.add_argument("--ink", default="blue")
ap.add_argument("--font-size", type=int, default=36)
ap.add_argument("--line-spacing", type=float, default=1.0)
a = ap.parse_args()
device = torch.device(a.device)


def parse_ink_color(ink_str):
    if ink_str.startswith("#") and len(ink_str) == 7:
        try:
            return tuple(int(ink_str.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            pass
    presets = {
        "blue": (24, 62, 132),
        "black": (26, 28, 32),
        "darkblue": (15, 38, 85),
        "red": (185, 30, 40),
    }
    return presets.get(str(ink_str).lower(), (24, 62, 132))


def load_style(folder, n):
    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
    if not paths:
        raise ValueError(f"No valid image files found in style directory {folder}")
    
    idx = np.random.choice(len(paths), n, replace=len(paths) < n)
    imgs = []
    widths = []
    tf = get_transform(grayscale=True)
    for i in idx:
        im = np.array(Image.open(paths[i]).convert("L"))
        im = crop_(im)
        if im.shape[0] == 0 or im.shape[1] == 0:
            im = np.full((32, 64), 255, dtype="uint8")
        im = cv2.resize(im, (int(32 * (im.shape[1] / max(1, im.shape[0]))), 32))
        img = 255 - im
        h, w = img.shape
        out = np.zeros((h, 192), dtype="float32")
        out[:, :min(w, 192)] = img[:, :min(w, 192)]
        img = 255 - out
        imgs.append(tf(Image.fromarray(img)))
        widths.append(min(w, 192))
    stacked = torch.cat(imgs, 0).unsqueeze(0).to(device)
    return stacked, torch.Tensor(widths).unsqueeze(0).to(device)


# 1. Initialize HWT Generator
model = TRGAN(batch_size=1)
model.netG.load_state_dict(torch.load(a.model, map_location="cpu"))
model.netG.to(device)
model.eval()

ST, SLEN = load_style(a.style_dir, a.num_samples)

# 2. Tokenize text into words, chunking words > 8 chars to preserve HWT attention stability
lines = a.text.split("\n")
token_runs = []  # list of lines, each line is list of (token_str, is_word_end)

for line_idx, line in enumerate(lines):
    words = line.strip().split()
    line_tokens = []
    for w in words:
        if len(w) <= 8:
            line_tokens.append((w, True))
        else:
            # Chunk long word into syllables of max 7 chars
            for k in range(0, len(w), 7):
                chunk = w[k:k+7]
                is_end = (k + 7 >= len(w))
                line_tokens.append((chunk, is_end))
    token_runs.append(line_tokens)

# Flatten tokens for model batch evaluation
flat_tokens = [tok for l in token_runs for tok, _ in l]
if not flat_tokens:
    flat_tokens = ["empty"]

enc, lens = model.netconverter.encode([tok.encode() for tok in flat_tokens])
enc = enc.to(device).repeat(1, 1, 1)

with torch.no_grad():
    fakes = model.netG.Eval(ST, enc)

# 3. Process generated patches: extract true ink boundary & alpha transparency
token_alphas = []
for idx, fake_ in enumerate(fakes):
    raw_patch = (fake_[0, 0, :, :].cpu().numpy() + 1.0) / 2.0
    expected_w = int(lens[idx]) * resolution
    
    # Locate actual stroke boundary via column minimum projection
    col_min = raw_patch.min(axis=0)
    ink_cols = np.where(col_min < 0.88)[0]
    if len(ink_cols) > 0:
        actual_end = ink_cols[-1] + 4  # keep ligature trailing exit
        w = max(int(lens[idx] * 10), min(raw_patch.shape[1], actual_end))
    else:
        w = min(raw_patch.shape[1], expected_w)

    cropped = raw_patch[:, :w]
    # Convert GAN grayscale into clean antialiased stroke alpha (1.0 = solid ink, 0.0 = clean paper)
    # Eliminates square gray boxes and patch seams completely
    alpha = np.clip((0.92 - cropped) / 0.65, 0.0, 1.0)
    token_alphas.append((alpha, w))

# 4. Canvas setup: Ruled notebook paper with blue ruling lines and red margin
PAGE_WIDTH = 1200
MARGIN_LEFT = 190
MARGIN_RIGHT = 1100
MARGIN_TOP = 170
LINE_HEIGHT_BASE = 56
line_h = max(38, int(LINE_HEIGHT_BASE * a.line_spacing))
target_token_h = max(24, int(36 * (a.font_size / 36.0)))
baseline_offset = int(target_token_h * 0.76)

# Compute required canvas height based on line wrapping
total_wrapped_lines = 1
for line_toks in token_runs:
    curr_x = MARGIN_LEFT + 25
    for (tok_str, is_end) in line_toks:
        # Approximate width
        tw = int(len(tok_str) * 16 * (target_token_h / 32.0))
        if curr_x + tw > MARGIN_RIGHT:
            total_wrapped_lines += 1
            curr_x = MARGIN_LEFT + 25
        curr_x += tw + (14 if is_end else 1)
    total_wrapped_lines += 1

PAGE_HEIGHT = max(1600, int(MARGIN_TOP + total_wrapped_lines * line_h + 200))

# Create paper canvas with subtle texture
canvas = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), (253, 251, 247))
grain = np.random.normal(0, 1.4, (PAGE_HEIGHT, PAGE_WIDTH, 3)).astype(np.int16)
arr = np.array(canvas).astype(np.int16)
canvas = Image.fromarray(np.clip(arr + grain, 0, 255).astype(np.uint8))
draw = ImageDraw.Draw(canvas)

if a.paper == "ruled":
    # Red vertical margin lines
    draw.line([(MARGIN_LEFT, 0), (MARGIN_LEFT, PAGE_HEIGHT)], fill=(225, 110, 110), width=2)
    draw.line([(MARGIN_LEFT - 6, 0), (MARGIN_LEFT - 6, PAGE_HEIGHT)], fill=(238, 160, 160), width=1)

    # Blue ruling lines
    y = MARGIN_TOP
    while y < PAGE_HEIGHT - 60:
        draw.line([(0, y), (PAGE_WIDTH, y)], fill=(185, 210, 235), width=1)
        y += line_h

    # Date & Page No header
    draw.text((PAGE_WIDTH - 320, 55), "Date: ___/___/20___", fill=(170, 170, 190))
    draw.text((PAGE_WIDTH - 320, 90), "Page No: ________", fill=(170, 170, 190))

elif a.paper == "grid":
    grid_sz = 40
    for gx in range(0, PAGE_WIDTH, grid_sz):
        draw.line([(gx, 0), (gx, PAGE_HEIGHT)], fill=(215, 230, 245), width=1)
    for gy in range(0, PAGE_HEIGHT, grid_sz):
        draw.line([(0, gy), (PAGE_WIDTH, gy)], fill=(215, 230, 245), width=1)

# 5. Composite tokens with authentic ink color along ruled lines
ink_rgb = parse_ink_color(a.ink)
r_ink, g_ink, b_ink = ink_rgb

curr_x = MARGIN_LEFT + 25
curr_y = MARGIN_TOP
token_ptr = 0

for line_idx, line_toks in enumerate(token_runs):
    if not line_toks:
        curr_y += line_h
        curr_x = MARGIN_LEFT + 25
        continue

    for (tok_str, is_word_end) in line_toks:
        if token_ptr >= len(token_alphas):
            break
        alpha_mask, orig_w = token_alphas[token_ptr]
        token_ptr += 1

        # Scale token to match target line height
        scale = target_token_h / 32.0
        tw = max(4, int(round(orig_w * scale)))
        th = target_token_h

        # Resample alpha
        alpha_resized = cv2.resize(alpha_mask, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Word wrap if exceeding right margin
        if curr_x + tw > MARGIN_RIGHT and curr_x > MARGIN_LEFT + 50:
            curr_x = MARGIN_LEFT + 25
            curr_y += line_h

        # Build colored RGBA token layer
        rgba = np.zeros((th, tw, 4), dtype=np.uint8)
        rgba[:, :, 0] = r_ink
        rgba[:, :, 1] = g_ink
        rgba[:, :, 2] = b_ink
        rgba[:, :, 3] = (alpha_resized * 255.0).astype(np.uint8)

        token_img = Image.fromarray(rgba, mode="RGBA")

        # Paste onto ruled line (baseline sits on curr_y)
        paste_x = curr_x
        paste_y = curr_y - baseline_offset
        canvas.paste(token_img, (paste_x, paste_y), token_img)

        # Advance x (inter-word gap if word end, else join seamlessly for sub-word chunks)
        curr_x += tw + (14 if is_word_end else 1)

    # Advance to next ruled line
    curr_x = MARGIN_LEFT + 25
    curr_y += line_h

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
canvas.save(a.out, format="PNG")
print("TW_RESULT " + json.dumps({"image": a.out, "text": a.text, "device": a.device, "paper": a.paper, "ink": a.ink}))
'''


def generate(text: str, style_dir: Path, out_png: Path,
             device: str | None = None, num_samples: int = 15,
             paper_type: str = "ruled", ink_color: str = "blue",
             line_spacing: float = 1.0, font_size: int = 36) -> dict:
    """Render `text` in the style of `style_dir`'s word images on ruled paper. Returns dict."""
    ensure_repo()
    device = device or detect_environment().device
    clean, warnings = filter_text(text)
    for w in warnings:
        print(f"[hwt] {w}")
    if not clean.strip() or len(clean) < len(text) * 0.4:
        # Non-English / regional script fallback: use style-conditioned handwriting synthesis
        from textwritter.multilingual.detector import detect_language_and_script
        from textwritter.multilingual.synthesis_engine import synthesize_handwriting_page
        from textwritter.multilingual.style_analyzer import analyze_handwriting_references

        detection = detect_language_and_script(text)
        if detection.primary_language != "en":
            style_files = []
            if style_dir and Path(style_dir).exists():
                style_files = [p for p in Path(style_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
            ref_profile = analyze_handwriting_references(style_files[:5]) if style_files else None

            ink_rgb = None
            if ink_color and ink_color.startswith("#") and len(ink_color) == 7:
                ink_rgb = (int(ink_color[1:3], 16), int(ink_color[3:5], 16), int(ink_color[5:7], 16))
            elif ink_color in ("blue", "black", "darkblue", "red"):
                from textwritter.notebook_engine import INKS
                ink_rgb = INKS.get(ink_color, (24, 62, 132))

            canvas, _, val = synthesize_handwriting_page(
                text=text,
                target_language=detection.primary_language,
                reference_profile=ref_profile,
                paper_type=paper_type,
                font_size=font_size,
                ink_color_override=ink_rgb,
            )
            canvas.save(out_png, format="PNG")
            print(f"[hwt] synthesized authentic {detection.primary_script} page matching reference -> {out_png}")
            return {"image": Path(out_png), "text": text, "warnings": warnings, "paper": paper_type, "ink": ink_color, "style": ref_profile.font_family if ref_profile else "kalam"}

        raise ValueError("nothing left to render after filtering unsupported chars")

    driver = VENDOR_DIR / DRIVER_NAME
    driver.write_text(_DRIVER_SRC, encoding="utf-8")
    cmd = [
        sys.executable, DRIVER_NAME,
        "--style-dir", str(Path(style_dir).resolve()),
        "--text", clean,
        "--out", str(Path(out_png).resolve()),
        "--device", device,
        "--num-samples", str(num_samples),
        "--paper", paper_type,
        "--ink", ink_color,
        "--font-size", str(font_size),
        "--line-spacing", str(line_spacing),
    ]
    proc = subprocess.run(cmd, cwd=VENDOR_DIR, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"HWT generation failed (exit {proc.returncode}):\n{tail}")
    if not Path(out_png).exists():
        raise RuntimeError("HWT generation reported success but produced no image")
    print(f"[hwt] wrote {out_png}")
    return {"image": Path(out_png), "text": clean, "warnings": warnings, "paper": paper_type, "ink": ink_color}
