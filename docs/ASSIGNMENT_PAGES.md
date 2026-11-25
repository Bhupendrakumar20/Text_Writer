# Assignment pages (CPU notebook)

This is the path for “write my CS743 solutions on ruled paper in blue pen.”
It is **not** handwriting cloning. It is Caveat (or another bundled face) with
math markup, ink, and a light scan look.

Longer agent notes: [AGENTS.md](../AGENTS.md).

## One command

```bash
cd /home/jeevant/Desktop/Text_Writter
source .venv/bin/activate   # if you use the venv
PYTHONPATH=src python3 src/textwritter/assignment_engine.py \
  samples/assignment2_solutions.txt \
  output/a2/Assignment2_solutions.pdf \
  --style caveat --ink blue
```

Windows: `write_pages.bat notes.txt` (see repo root). Older `write_assignment.bat`
is the **HWT** path and needs `samples/my_handwriting.png`.

## Content file

Plain UTF-8 text. Example header + working:

```
# Q1.   Artificial neuron, z = w^{T}x + b

Given x = (2, -1, 3), w = (1, 2, -1), b = 1/2

a) \to z = w^{T}x + b
        = (1)(2) + (2)(-1) + (-1)(3) + 1/2
   \to z = -2.5
```

| Markup | Result |
|---|---|
| `# title` | underlined heading |
| `\frac{a}{b}` or `5/6` | stacked fraction (numerator over bar over denominator) |
| `\to \in \times \partial \Sigma \sigma \nabla` | symbols |
| indent with spaces | hanging indent |
| `---` | horizontal rule |
| `__word__` / `~~word~~` | underline / strike |

Full CS743 Assignment 2 answers: `samples/assignment2_solutions.txt`.
Latest PDF: `output/a2/Assignment2_solutions.pdf`.

## Styles

| `--style` | Use when |
|---|---|
| `caveat` | Default for A2. Joined cursive. Digit `1` is readable. Capital `L` is swapped to a clearer face. |
| `kalam` | Neater print. Digit `1` looks like `l` — bad for `2×1`. |
| `indie` / `patrick` | Casual / neat print alternatives |

## Glyph sheet (you write the math — this is the next real upgrade)

English words come from Caveat. You do **not** rewrite a–z. You rewrite the symbols
the font cannot draw. Print `output/a2/glyph_sheet.pdf`. Starred boxes first.

```bash
PYTHONPATH=src python3 src/textwritter/glyphs.py sheet --out output/a2/glyph_sheet.pdf
# fill with the same blue pen, photograph square-on, all 4 black corners in frame
PYTHONPATH=src python3 src/textwritter/glyphs.py slice --image samples/glyph_sheet_filled.jpg
PYTHONPATH=src python3 src/textwritter/glyphs.py check
```

Must-write (3 copies each): `→ ∂ Σ ✓ ∇ δ σ ℓ θ ε η ∏`  
Optional rest of the sheet: `∈ √ ∞ α …`

Then re-run `assignment_engine.py`. Empty boxes are skipped, so a half-filled sheet still works.

## Neural English (optional, different tool)

If you have a photo of **English** handwriting:

```bash
python src/textwritter/extract_style.py --pdf scan.pdf --page 1 --out samples/my_handwriting.png
python src/textwritter/write_assignment.py --engine hwt --style samples/my_handwriting.png --text-file notes.txt
```

Colab notebook: `notebooks/03_assignment_pages.ipynb` (HWT path). For the font
notebook used for A2, stay on `assignment_engine.py` — it does not need a GPU.

## What not to do

- Do not feed the web UI screenshot as `--style`.
- Do not split Caveat into per-letter glyphs (joins die).
- Do not sit text on a different grid than the blue rules.
- Do not expect One-DM to write a 27-page assignment on a free Colab session.
