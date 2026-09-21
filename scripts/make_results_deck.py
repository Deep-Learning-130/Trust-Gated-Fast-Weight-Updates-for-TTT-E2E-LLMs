#!/usr/bin/env python
"""Generate the results deck: what has been done, stated plainly.

Written for a reader outside the project: no experiment numbers, file names or
commit pins on the slides, and every figure is followed by what it means.
Austere on purpose. Times New Roman, black on white, no accent colour, no cards,
no banners, greyscale figures. One column, tables and short statements.

`make_review_deck.py` is the other deck and keeps its own visual system; this
one deliberately does not share it.

The pilot figures are read from its result files at build time, so the deck moves
when those results move; that is only worth anything because the corpus is
pinned (`experiments/002-pilot-tiny-ttt/run_pilot.py`). Figures from GPU runs and
from the cost model are held as constants, because their sources are git-ignored
or live in prose documents.

    .venv/Scripts/python scripts/make_results_deck.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import MSO_ANCHOR  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R002 = ROOT / "experiments" / "002-pilot-tiny-ttt" / "results"
OUT_DIR = ROOT / "docs" / "review"
FIG_DIR = OUT_DIR / "figures"
DATE = "2026-09-16"
OUT = OUT_DIR / f"TrustGate_Results_{DATE}.pptx"

BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Times New Roman"

W, H = Inches(13.333), Inches(7.5)
L0 = Inches(1.0)
CW = Inches(11.333)
TITLE_TOP = Inches(0.62)
BODY_TOP = Inches(1.78)

#: Bottom margin every body box is clamped to. There is no footer, so this is
#: whitespace rather than a reservation.
FOOT_MARGIN = Inches(0.30)


# ----------------------------------------------------------- text metrics --
# The previous layout estimated a bullet at a flat 0.40in regardless of how many
# lines it actually wrapped to, so any table under a two-line bullet had to be
# nudged down by hand. Measuring against the real font removes the guess.
_FONT_FILES = {
    (False, False): "times.ttf",
    (True, False): "timesbd.ttf",
    (False, True): "timesi.ttf",
    (True, True): "timesbi.ttf",
}
_FONT_DIR = Path("C:/Windows/Fonts")
_SCALE = 4  # measure at 4x nominal size, for sub-point resolution
_font_cache: dict = {}


def _font(size_pt: float, bold: bool, italic: bool):
    key = (round(size_pt * _SCALE), bold, italic)
    if key not in _font_cache:
        from PIL import ImageFont

        _font_cache[key] = ImageFont.truetype(
            str(_FONT_DIR / _FONT_FILES[(bold, italic)]),
            int(round(size_pt * _SCALE)),
        )
    return _font_cache[key]


def _advance(text: str, size_pt: float, bold: bool, italic: bool) -> float:
    """Width of `text` in points. Falls back to a ratio off Windows."""
    try:
        return _font(size_pt, bold, italic).getlength(text) / _SCALE
    except (OSError, ImportError, KeyError):
        return len(text) * size_pt * (0.52 if bold else 0.48)


def wrapped_lines(runs, width_in: float, size_pt: float) -> int:
    """Rendered line count for a sequence of (text, bold, italic) runs.

    Greedy word wrap, the same rule PowerPoint applies. Runs are measured with
    their own face, so a bold lead-in is not costed as regular text.
    """
    limit = width_in * 72.0
    count, used = 1, 0.0
    for text, bold, italic in runs:
        for i, word in enumerate(text.split(" ")):
            if not word and i:
                continue
            piece = word if used == 0.0 else " " + word
            w = _advance(piece, size_pt, bold, italic)
            if used and used + w > limit:
                count += 1
                used = _advance(word, size_pt, bold, italic)
            else:
                used += w
    return count


def text_height(n_lines: int, size_pt: float, spacing: float = 1.0,
                space_after: float = 0.0) -> float:
    """Inches occupied. 1.2x nominal size is PowerPoint's single-line height."""
    return (n_lines * size_pt * 1.2 * spacing + space_after) / 72.0


# ------------------------------------------------------------- primitives --
def _box(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def _para(tf, first, text, size, *, bold=False, space_after=8, align=None,
          line=None, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = BLACK
    r.font.name = FONT
    p.space_after = Pt(space_after)
    if align:
        p.alignment = align
    if line:
        p.line_spacing = line
    return p


def _rule(slide, top, width=CW, left=L0, weight=0.75):
    ln = slide.shapes.add_connector(1, left, top, left + width, top)
    ln.line.color.rgb = BLACK
    ln.line.width = Pt(weight)
    return ln


def slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE

    tf = _box(s, L0, TITLE_TOP, CW, Inches(0.9))
    _para(tf, True, title, 28, bold=True, space_after=0, line=1.04)
    _rule(s, BODY_TOP - Inches(0.26))
    return s


def lines(s, top, items, size=17, gap=11, width=Inches(11.0)):
    """Short statements, one line each where possible.

    The box is clamped to the slide rather than given a fixed height. A textbox
    taller than the slide does not complain and does not show -- until someone
    adds one more line, at which point it clips silently.

    Returns the measured bottom of the text, not an estimate of it, so whatever
    is placed underneath does not have to be positioned by eye.
    """
    tf = _box(s, L0, top, width, min(Inches(4.0), H - top - FOOT_MARGIN))
    width_in = width / 914400
    used = 0.0
    for i, item in enumerate(items):
        head, tail = item if isinstance(item, tuple) else (item, "")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        # A plain bullet, not an em dash. The dash read as a stylistic tic
        # once there were thirty of them down the deck.
        r.text = "•  " + head
        r.font.size = Pt(size)
        r.font.bold = bool(tail)
        r.font.color.rgb = BLACK
        r.font.name = FONT
        runs = [(r.text, bool(tail), False)]
        if tail:
            r2 = p.add_run()
            r2.text = "  " + tail
            r2.font.size = Pt(size)
            r2.font.color.rgb = BLACK
            r2.font.name = FONT
            runs.append((r2.text, False, False))
        p.space_after = Pt(gap)
        p.line_spacing = 1.16
        used += text_height(wrapped_lines(runs, width_in, size), size, 1.16, gap)
    return top + Inches(used)


def table(s, top, headers, rows, widths, *, row_h=0.4, size=14):
    n_rows, n_cols = len(rows) + 1, len(headers)
    grid = [list(headers)] + [list(r) for r in rows]

    # A cell whose text wraps makes PowerPoint grow the row, so a fixed row
    # height understates the table and anything placed under it by arithmetic
    # lands on top of it. Measure instead.
    MARGIN_R, PAD_V = 0.16, 0.08
    heights = []
    for r in range(n_rows):
        need = row_h
        for c in range(n_cols):
            n = wrapped_lines([(str(grid[r][c]), r == 0, r == 0)],
                              widths[c] - MARGIN_R, size)
            need = max(need, text_height(n, size) + PAD_V)
        heights.append(need)

    shape = s.shapes.add_table(n_rows, n_cols, L0, top,
                               Emu(int(sum(widths) * 914400)),
                               Emu(int(sum(heights) * 914400)))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)

    for r in range(n_rows):
        tbl.rows[r].height = Emu(int(heights[r] * 914400))
        for c in range(n_cols):
            cell = tbl.cell(r, c)
            cell.fill.background()
            cell.margin_left = Inches(0.0)
            cell.margin_right = Inches(0.16)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(headers[c] if r == 0 else rows[r - 1][c])
            run.font.size = Pt(size)
            run.font.bold = r == 0
            run.font.italic = r == 0
            run.font.color.rgb = BLACK
            run.font.name = FONT
    return top + Inches(sum(heights)) + Inches(0.18)


def picture(s, path, top, height_in):
    pic = s.shapes.add_picture(str(path), L0, top, height=Inches(height_in))
    pic.left = int(L0 + (CW - pic.width) / 2)
    return top + Inches(height_in) + Inches(0.16)


def note(s, top, text, size=12.5, width=Inches(11.0)):
    """The reading of the numbers above it, in the deck's one italic voice.

    Sized to its own measured height. The previous fixed 0.9in cap silently let
    a fourth line hang below the box, which is invisible on screen and wrong in
    print.
    """
    width_in = width / 914400
    n = wrapped_lines([(text, False, True)], width_in, size)
    height = Inches(text_height(n, size, 1.3))
    tf = _box(s, L0, top, width, min(height, H - top - FOOT_MARGIN))
    _para(tf, True, text, size, space_after=0, line=1.3, italic=True)
    return top + height


# ---------------------------------------------------------------- figures --
# Greyscale, to match the deck. Identity is carried by mark shape and by direct
# labels rather than by hue, so nothing is lost in print or to a colourblind
# reader.
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=11, length=4, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname("Times New Roman")


def fig_null(null, deep_d) -> Path:
    vals = sorted(null["null_effect_sizes"])
    ys, placed = [], []
    for v in vals:
        row = 0
        while any(abs(v - p) < 0.09 for p, r in placed if r == row):
            row += 1
        placed.append((v, row))
        ys.append(row)

    fig, ax = plt.subplots(figsize=(10.4, 2.95), dpi=200)
    ax.axvspan(-2.2, -0.8, color="black", alpha=0.07, lw=0)
    ax.axvspan(0.8, 2.2, color="black", alpha=0.07, lw=0)
    for x in (-0.8, 0.8):
        ax.axvline(x, color="black", ls=(0, (4, 3)), lw=1.0)

    ax.scatter(vals, [y * 0.14 for y in ys], s=76, color="black", zorder=3,
               linewidths=0)
    # vlines, not axvline: the marker must stop below its own label, or the rule
    # is drawn straight through the text.
    ax.vlines(deep_d, -0.10, 0.50, color="black", lw=2.4, zorder=4)
    ax.annotate(f"attack result  d = {deep_d:+.3f}", xy=(deep_d, 0.60), ha="center",
                fontsize=12, fontweight="bold", fontname="Times New Roman")
    ax.text(-2.12, 0.50, "past the threshold", fontsize=11,
            fontname="Times New Roman")
    ax.text(2.12, 0.50, "past the threshold", fontsize=11, ha="right",
            fontname="Times New Roman")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.12, 0.74)
    ax.set_yticks([])
    ax.set_xlabel("Effect size measured with no attacker present (20 repeats, 5 runs each)",
                  fontsize=12, fontname="Times New Roman", labelpad=7)
    ax.spines["left"].set_visible(False)
    _style(ax)
    fig.tight_layout()
    p = FIG_DIR / "null_distribution.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


#: Per-chunk cross-entropy, experiments/003-smoke-125m, runner a7447ce.
#: Held here rather than read from results/smoke.json because that file is
#: git-ignored and produced on a GPU box, not on the machine building the deck.
CE_CURVE = [11.886, 10.708, 10.163, 10.128, 9.949, 9.923, 9.851, 9.868]
CE_PREDICTED_INIT = 11.9154


def fig_curve() -> Path:
    fig, ax = plt.subplots(figsize=(10.4, 3.0), dpi=200)
    xs = list(range(1, len(CE_CURVE) + 1))
    ax.axhline(CE_PREDICTED_INIT, color="black", ls=(0, (5, 4)), lw=1.0)
    # Above the rule, not on it: va="center" puts the dashes through the glyphs.
    ax.text(9.3, CE_PREDICTED_INIT + 0.06,
            f"predicted in advance  {CE_PREDICTED_INIT}", va="bottom", ha="right",
            fontsize=11, fontname="Times New Roman")
    ax.plot(xs, CE_CURVE, color="black", lw=1.8, zorder=3)
    ax.scatter(xs, CE_CURVE, s=58, color="black", zorder=4,
               edgecolors="white", linewidths=1.4)
    ax.annotate(f"{CE_CURVE[0]:.3f}", xy=(1.12, CE_CURVE[0] + 0.07),
                fontsize=11.5, fontname="Times New Roman")
    ax.annotate(f"{CE_CURVE[-1]:.3f}", xy=(7.5, CE_CURVE[-1] - 0.30),
                fontsize=11.5, fontname="Times New Roman")
    ax.set_xlim(0.6, 9.4)
    ax.set_ylim(9.4, 12.3)
    ax.set_xticks(xs)
    ax.grid(axis="y", color="black", alpha=0.12, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("Section of text read (1,024 tokens each)", fontsize=12, fontname="Times New Roman",
                  labelpad=7)
    ax.set_ylabel("Prediction error (nats)", fontsize=12, fontname="Times New Roman",
                  labelpad=7)
    _style(ax)
    fig.tight_layout()
    p = FIG_DIR / "inner_loop_curve.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ build --
def build() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    deep = json.loads((R002 / "deep.json").read_text())
    null = json.loads((R002 / "null.json").read_text())

    deep_d = deep["effect_size"]
    floor_d = deep["noise_floor_effect_size"]
    rel = deep["relative_degradation"]
    fpr = null["false_positive_rate_at_bar"]
    n_draws = null["n_draws"]
    n_clear = round(fpr * n_draws)

    f_null = fig_null(null, deep_d)
    f_curve = fig_curve()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # Written for a reader with no contact with the project: no experiment
    # numbers, file names, commit pins or hardware model numbers on any slide.
    # docs/ORIENTATION.md is the long form for anyone who wants those.

    # -- 1. title -----------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    tf = _box(s, L0, Inches(2.7), Inches(11.0), Inches(1.24))
    _para(tf, True, "Trust-Gated Fast-Weight Updates for TTT-E2E LLMs", 36,
          bold=True, space_after=0, line=1.06)
    _rule(s, Inches(4.86), width=Inches(3.0), weight=1.25)

    # -- 2. the problem -----------------------------------------------------
    s = slide(prs, "The problem")
    y = lines(s, BODY_TOP, [
        ("Models that keep learning.",
         "A new kind of language model, TTT-E2E, keeps adjusting part of its "
         "own internal settings while it reads. This is what lets it handle "
         "very long documents cheaply. The part that changes is called the "
         "fast weights."),
        ("Why that is a risk.",
         "If a model learns from whatever it reads, then whoever controls what "
         "it reads can influence what it learns. An attacker who slips "
         "ordinary-looking text into a document, email or web page might steer "
         "the model into doing worse on later, unrelated requests."),
        ("Why current defences would miss it.",
         "No single sentence in such an attack looks suspicious, so content "
         "filters have nothing to catch. The model's original, stored settings "
         "are never touched, so checks on those see nothing either."),
    ], size=16, gap=10)
    note(s, y + Inches(0.14),
         "If this attack exists, it would be quiet, gradual, and invisible to "
         "the tools used to protect language models today. This project first "
         "tests whether it exists, and builds a defence only if it does.")

    # -- 3. the proposed defence --------------------------------------------
    s = slide(prs, "The proposed defence")
    y = lines(s, BODY_TOP, [
        ("A check on every change.",
         "Each time the model is about to update its fast weights, a trust gate "
         "inspects the change first and can refuse it."),
        ("What it looks for.",
         "Whether the change would pull the model's answers away from those of "
         "a frozen, trusted copy of itself, tested on a small rotating set of "
         "passages the attacker cannot see."),
        ("A cap on total change.",
         "The gate keeps a running total of how far the model has moved. Once "
         "that total reaches a set budget, further changes are refused, however "
         "harmless each one looks on its own."),
        ("A way back.",
         "Recent trusted versions of the fast weights are kept, so the model "
         "can be restored instantly."),
    ], size=16, gap=10)
    note(s, y + Inches(0.14),
         "The cap on total change is the part that can be proven. Within each "
         "budget period, the model cannot move further from its trusted "
         "starting point than the budget allows, whatever the attacker feeds "
         "it. That is a mathematical guarantee rather than a detection rate, "
         "and it holds even for harmful changes the gate fails to recognise.")

    # -- 4. how the decision is made ----------------------------------------
    s = slide(prs, "Deciding whether to build it")
    y = lines(s, BODY_TOP, [
        ("Attack first, defence second.",
         "A defence against an attack nobody has shown to work would be wasted "
         "effort. So the first job is to test, on the real model, whether the "
         "attack works at all."),
        ("Rules fixed in advance.",
         "What counts as a working attack was written down and committed on "
         "28 July 2026, before any attack code existed. It cannot be adjusted "
         "after the results are in."),
    ], size=16, gap=10)
    y = table(s, y + Inches(0.18),
              ["Condition", "Threshold", "In plain terms"],
              [["Effect size (Cohen's d)", "≥ 0.80",
                "attacked and normal runs must differ by far more than chance"],
               ["Loss of performance", "≥ 10%",
                "the model must get at least a tenth worse at ordinary tasks"],
               ["Naturalness of the attack text", "≤ 1.5",
                "the attack text must read almost as naturally as normal text, "
                "judged by a separate model"]],
              [3.2, 1.3, 6.5])
    note(s, y + Inches(0.04),
         "All three must hold. If they do, the defence gets built. If any one "
         "fails, the project stops and publishes that finding, which is a "
         "useful result in its own right.")

    # -- 5. where things stand ----------------------------------------------
    s = slide(prs, "Where things stand")
    y = lines(s, BODY_TOP, [
        ("Done.", "The measuring tools are built and tested. A small practice "
                  "version of the experiment runs on a fixed, verified input. "
                  "The real model has been run for the first time. Costs have "
                  "been measured rather than guessed."),
        ("Not done.", "The decisive test on the real model. It needs a rented "
                      "data-centre GPU, which is being arranged."),
        ("Verdict.", "None yet. Nothing in this deck says whether the attack "
                     "works."),
        ("Spent.", "$0.13 in total. All computing so far has used free "
                   "hardware."),
    ], size=17)
    note(s, y + Inches(0.14),
         "Most of what follows is groundwork for an answer that can be trusted: "
         "testing the measuring tools, finding and fixing errors in them, and "
         "costing the real test before paying for it.")

    # -- 6. the small stand-in ----------------------------------------------
    s = slide(prs, "A first test on a small stand-in model")
    y = lines(s, BODY_TOP, [
        ("Why a stand-in.",
         "The real model needs expensive hardware, so the whole pipeline was "
         "first run on a tiny model built the same way, with about 100,000 "
         "adjustable settings instead of a billion."),
        ("What the attacker could do.",
         "Only choose which real, harmless sentences to feed the model, and in "
         "what order. No invented or unusual text."),
    ], size=16, gap=8)
    y = table(s, y + Inches(0.18),
              ["Measurement", "Result", "How to read it"],
              [["Effect size (Cohen's d)", f"{deep_d:+.3f}",
                "far below the 0.80 required"],
               ["Same measurement, no attacker", f"{floor_d:+.3f}",
                "bigger than the attack's effect, so that effect is just noise"],
               ["Loss of performance", f"{rel * 100:.4f}%",
                f"about {round(0.10 / rel)} times smaller than the 10% required"],
               ["Learning while reading",
                f"{deep['inner_loop_gain_nats']:.4f} nats",
                f"clearly active (a model with almost no capacity to adapt "
                f"scores {deep['shallow_inner_loop_gain_nats']})"]],
              [3.2, 1.8, 6.0])
    note(s, y + Inches(0.04),
         "The attack had no measurable effect on this small model. Because the "
         "model was genuinely learning from what it read, this is a real "
         "negative result rather than a broken test. It says nothing yet about "
         "the full-size model, which is around ten thousand times larger and "
         "built differently in important ways.")

    # -- 7. checking the tool -----------------------------------------------
    s = slide(prs, "Checking the measuring tool itself")
    y = picture(s, f_null, BODY_TOP, 2.45)
    y = lines(s, y + Inches(0.08), [
        ("How to read the chart.",
         f"Each dot is one of {n_draws} repeats of the measurement with no "
         f"attacker present, placed left or right by the effect size it "
         f"reported. With nothing to find, every dot should sit near the "
         f"middle. The dashed lines mark the 0.80 threshold, and dots in or at "
         f"the edge of the grey zones are false alarms. The bold line is the "
         f"attack's result from the previous slide."),
        ("What it shows.",
         f"{n_clear} of the {n_draws} crossed the threshold with nothing to "
         f"find, a false-alarm rate of about {fpr:.0%}. The attack's result "
         f"sits among the no-attacker dots, so it cannot be told apart from "
         f"chance."),
    ], size=15, gap=7)
    note(s, y + Inches(0.06),
         "This is a finding about the rule itself. With five runs per condition, "
         "clearing the threshold once is not proof of an attack. The threshold "
         "was deliberately left unchanged, since moving it after seeing data "
         "would defeat the point of fixing it in advance.")

    # -- 8. our own error ---------------------------------------------------
    s = slide(prs, "An error in our own experiment, found and fixed")
    y = lines(s, BODY_TOP, [
        ("The error.",
         "The practice experiment read its text from the project's own "
         "documentation at the moment it ran. Every edit to the documentation "
         "quietly changed the experiment's input."),
        ("Why it mattered.",
         "Editing two unrelated files was enough to swing the result from "
         "slightly positive to strongly negative, past the threshold in the "
         "wrong direction. Identical code gave different answers on different "
         "days."),
    ], size=16, gap=8)
    y = table(s, y + Inches(0.18),
              ["", "Before the fix", "After the fix"],
              [["Input text", "read live, changed with every edit",
                "frozen copy, checked before each run"],
               ["If the input changes", "result silently moves (+0.05 to −1.28)",
                "the run refuses to start"],
               ["False-alarm rate", "55%", f"{fpr:.0%}"]],
              [3.0, 4.0, 4.0])
    note(s, y + Inches(0.04),
         "The error was in this project's own code, and it was caught before it "
         "could lead to a wrong conclusion. Every result in this deck was "
         "produced after the fix. The 55% false-alarm rate belongs to the "
         "faulty setup and should not be quoted.")

    # -- 9. the real model --------------------------------------------------
    s = slide(prs, "Running the real model for the first time")
    y = lines(s, BODY_TOP, [
        "A small, untrained version of the real model, run only to confirm "
        "our code drives it correctly.",
    ], size=16, gap=4)
    y = table(s, y + Inches(0.10),
              ["Measurement", "Result", "What it means"],
              [["Adjustable settings in total", "184,363,776",
                "matches the published design"],
               ["Settings rewritten while reading", "11,501,568",
                "the part an attacker could influence"],
               ["Share open to influence", "6.24%", "about 1 setting in 16"]],
              [3.6, 2.3, 5.1])
    y = picture(s, f_curve, y + Inches(0.02), 1.72)
    note(s, y - Inches(0.04),
         f"How to read the chart: each point is one section of 1,024 tokens, "
         f"read in order, and its height is the model's prediction error on "
         f"that section, so lower is better. The dashed line is the starting "
         f"error worked out in advance from the model's design. The first point "
         f"lands within {abs(CE_CURVE[0] - CE_PREDICTED_INIT):.3f} of it, which "
         f"confirms the setup is right, and the line then drops by "
         f"{max(CE_CURVE) - min(CE_CURVE):.3f} as the model reads. That drop is "
         f"the fast weights learning from what they read in real time, which is "
         f"exactly the behaviour an attacker would try to exploit.")

    # -- 10. costing it -----------------------------------------------------
    s = slide(prs, "Costing the real test before paying for it")
    y = table(s, BODY_TOP,
              ["Item", "Assumed", "Found on checking", "Consequence"],
              [["Evaluation text", "up to 1 billion tokens", "2.0 billion tokens",
                "a full pass takes about 7.5 hours, not under 1"],
               ["Model file", "5.9 GB", "5.35 GB", "within the expected range"],
               ["GPU memory needed", "about 25 GB", "36 to 49 GB, estimated",
                "an 80 GB card is required"],
               ["Whole session", "about 4 hours", "about 10.5 hours",
                "too close to the 12-hour spending limit"]],
              [2.4, 2.3, 2.5, 3.8])
    note(s, y + Inches(0.04),
         "All of this was found by inspecting data and code for free, before "
         "renting anything. Because the full evaluation text is too large to "
         "use in one session, the real test will use a fixed sample of it. The "
         "sample is still far larger than needed to detect the differences "
         "that matter, and it was chosen and written down before any result was "
         "seen.")

    # -- 11. the tools ------------------------------------------------------
    s = slide(prs, "The measuring tools are ready")
    y = lines(s, BODY_TOP, [
        ("An independent judge.",
         "Whether attack text reads naturally is scored by a separate, "
         "well-known model, GPT-2, never by the model under attack. A "
         "compromised model cannot be trusted to judge."),
        ("The full test runs end to end.",
         "Every step of the decisive test now runs. With an untrained model "
         "plugged in, it gives no verdict and says so at the top of its report."),
    ], size=16, gap=8)
    y = table(s, y + Inches(0.18),
              ["Check", "Result", "Required", "What it shows"],
              [["Tells normal text from scrambled text", "66×", "> 2×",
                "the judge spots unnatural text easily"],
               ["Difference from the official GPT-2", "0.0006", "< 0.001",
                "our copy is the same model"],
               ["Agrees on the next word", "100%", "",
                "identical choices over 1,024 tokens"],
               ["Automated tests on the tools", "319 passing", "",
                "checked by code, not by eye"]],
              [4.0, 1.5, 1.4, 4.1])
    note(s, y + Inches(0.04),
         "The naturalness condition could not be scored until a trustworthy "
         "judge existed. It can be now, at the same threshold set in July.")

    # -- 12. not yet known --------------------------------------------------
    s = slide(prs, "What is not yet known")
    y = lines(s, BODY_TOP, [
        ("Whether the attack works on the real model.",
         "The decisive test has not been run, so there is no verdict."),
        ("Whether the real model's learning builds up correctly over a long "
         "text.",
         "This check could not finish on a home graphics card. A 24 GB card "
         "has been arranged."),
        ("How the full-size model behaves.",
         "It needs a rented 80 GB data-centre GPU. Free services such as Colab "
         "and Kaggle cannot run it."),
        ("Anything about the real model from the small test.",
         "The practice model is far smaller and built differently, so its "
         "result does not carry over."),
    ], size=16)
    note(s, y + Inches(0.20),
         "This slide is here so the earlier results are not read as more than "
         "they are. Along the way, seven setup failures were found and fixed on "
         "free hardware, each of which would otherwise have cost paid GPU time.")

    # -- 13. next -----------------------------------------------------------
    s = slide(prs, "Next steps")
    y = table(s, BODY_TOP,
              ["Step", "Needs", "Estimated cost"],
              [["1. Finish the real-model check", "rented 24 GB GPU",
                "about 1 hour, ≈ ₹50"],
               ["2. Dry run of the full test", "same session",
                "about 30 minutes, ≈ ₹25"],
               ["3. Rehearsal on a small trained model",
                "model file already downloaded", "about 30 minutes"],
               ["4. Confirm the full-size model is set up correctly",
                "rented 80 GB GPU", "about 4 hours, ≈ ₹740"],
               ["5. The decisive test", "all of the above", "separate session"]],
              [5.0, 3.6, 2.4])
    note(s, y + Inches(0.06),
         f"Steps 1 to 4 make sure the final answer can be trusted, and step 5 "
         f"gives it. Either outcome, attack confirmed or not, is worth "
         f"publishing. Step 5 will be read against the {fpr:.0%} false-alarm "
         f"rate found earlier.")

    # -- 14. glossary -------------------------------------------------------
    s = slide(prs, "Glossary")
    y = table(s, BODY_TOP,
              ["Term", "Meaning"],
              [["TTT-E2E",
                "A recent language model design that keeps learning from its "
                "input while running, to handle very long texts."],
               ["Fast weights",
                "The part of such a model that changes while it reads. The "
                "target of the attack."],
               ["Trust gate",
                "The proposed defence: a check on every change to the fast "
                "weights, with a cap on total change."],
               ["Kill gate",
                "The one-time decision on whether the attack is real enough to "
                "justify building the defence."],
               ["Pre-registration",
                "Writing down the success criteria before an experiment, so "
                "they cannot be bent to fit the results."],
               ["Cohen's d",
                "A standard measure of how far apart two sets of results are, "
                "relative to their natural spread. 0.8 counts as large."],
               ["Null test",
                "Running a measurement with nothing to find, to learn how often "
                "it raises a false alarm."],
               ["Token",
                "A word or part of a word, the unit a language model reads."],
               ["Nats",
                "A unit of prediction error. Lower means the model is better at "
                "guessing the next token."]],
              [2.2, 8.8], size=13)
    note(s, y + Inches(0.06),
         f"Deck prepared {DATE}. Total spend to date: $0.13.")

    try:
        prs.save(OUT)
    except PermissionError:
        # PowerPoint holds an exclusive lock on an open file, and python-pptx
        # surfaces that as a zipfile traceback naming neither the application
        # nor the fix. Rebuilding the deck right after looking at it is the
        # normal workflow, so this is the common failure, not an exotic one.
        raise SystemExit(
            f"Cannot write {OUT.name}: the file is open in PowerPoint. "
            "Close it and re-run. Nothing else is wrong; the figures in "
            f"{FIG_DIR.name}/ were rebuilt and are current."
        ) from None
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
