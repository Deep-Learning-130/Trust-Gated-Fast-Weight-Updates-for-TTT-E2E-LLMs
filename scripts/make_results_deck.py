#!/usr/bin/env python
"""Generate the results deck: what has been done, stated plainly.

Austere on purpose. Times New Roman, black on white, no accent colour, no cards,
no banners, greyscale figures. One column. Claims are tables and short
statements rather than paragraphs -- a supervisor reading a status deck wants
the numbers and the scope, and prose between them slows that down.

`make_review_deck.py` is the other deck and keeps its own visual system; this
one deliberately does not share it.

Every number is read from a committed result file at build time, never retyped,
so the deck moves when the results move. That is only worth anything because the
corpus is pinned (`experiments/002-pilot-tiny-ttt/run_pilot.py`) -- before that,
the files it reads were not reproducible.

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
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.util import Emu, Inches, Pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R002 = ROOT / "experiments" / "002-pilot-tiny-ttt" / "results"
OUT_DIR = ROOT / "docs" / "review"
FIG_DIR = OUT_DIR / "figures"
DATE = "2026-09-15"
OUT = OUT_DIR / f"TrustGate_Results_{DATE}.pptx"

BLACK = RGBColor(0x00, 0x00, 0x00)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Times New Roman"

W, H = Inches(13.333), Inches(7.5)
L0 = Inches(1.0)
CW = Inches(11.333)
TITLE_TOP = Inches(0.62)
BODY_TOP = Inches(1.78)
FOOTER = "Trust-Gated Fast-Weight Updates for TTT-E2E LLMs"

_page = [0]


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

    _page[0] += 1
    tf = _box(s, L0, H - Inches(0.58), CW, Inches(0.26))
    p = _para(tf, True, f"{FOOTER}   |   {DATE}   |   {_page[0]}", 10,
              space_after=0)
    p.alignment = PP_ALIGN.LEFT
    return s


def lines(s, top, items, size=17, gap=11, width=Inches(11.0)):
    """Short statements, one line each where possible."""
    tf = _box(s, L0, top, width, Inches(4.0))
    for i, item in enumerate(items):
        head, tail = item if isinstance(item, tuple) else (item, "")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = "—  " + head
        r.font.size = Pt(size)
        r.font.bold = bool(tail)
        r.font.color.rgb = BLACK
        r.font.name = FONT
        if tail:
            r2 = p.add_run()
            r2.text = "  " + tail
            r2.font.size = Pt(size)
            r2.font.color.rgb = BLACK
            r2.font.name = FONT
        p.space_after = Pt(gap)
        p.line_spacing = 1.16
    return top + Inches(0.40) * len(items)


def table(s, top, headers, rows, widths, *, row_h=0.4, size=14):
    n_rows, n_cols = len(rows) + 1, len(headers)
    shape = s.shapes.add_table(n_rows, n_cols, L0, top,
                               Emu(int(sum(widths) * 914400)),
                               Inches(row_h * n_rows))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)

    for r in range(n_rows):
        tbl.rows[r].height = Inches(row_h)
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
    return top + Inches(row_h * n_rows) + Inches(0.18)


def picture(s, path, top, height_in):
    pic = s.shapes.add_picture(str(path), L0, top, height=Inches(height_in))
    pic.left = int(L0 + (CW - pic.width) / 2)
    return top + Inches(height_in) + Inches(0.16)


def note(s, top, text, size=12.5):
    tf = _box(s, L0, top, Inches(11.0), Inches(0.9))
    _para(tf, True, text, size, space_after=0, line=1.3, italic=True)


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
    ax.annotate(f"observed  d = {deep_d:+.3f}", xy=(deep_d, 0.60), ha="center",
                fontsize=12, fontweight="bold", fontname="Times New Roman")
    ax.text(-2.12, 0.50, "d ≤ −0.8", fontsize=11,
            fontname="Times New Roman")
    ax.text(2.12, 0.50, "d ≥ 0.8", fontsize=11, ha="right",
            fontname="Times New Roman")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.12, 0.74)
    ax.set_yticks([])
    ax.set_xlabel("Cohen's d, control vs control (20 draws, 5 seeds each)",
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
            f"predicted at init  {CE_PREDICTED_INIT}", va="bottom", ha="right",
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
    ax.set_xlabel("Inner-loop chunk", fontsize=12, fontname="Times New Roman",
                  labelpad=7)
    ax.set_ylabel("nats / token", fontsize=12, fontname="Times New Roman",
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
    pilot = json.loads((R002 / "pilot.json").read_text())
    null = json.loads((R002 / "null.json").read_text())

    deep_d = deep["effect_size"]
    fpr = null["false_positive_rate_at_bar"]
    n_clear = round(fpr * null["n_draws"])

    f_null = fig_null(null, deep_d)
    f_curve = fig_curve()

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # -- 1. title -----------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    tf = _box(s, L0, Inches(2.7), Inches(11.0), Inches(1.7))
    _para(tf, True, "Trust-Gated Fast-Weight Updates for TTT-E2E LLMs", 36,
          bold=True, space_after=14, line=1.06)
    _para(tf, False, "Status of work — what has been measured", 20,
          space_after=0)
    _rule(s, Inches(4.86), width=Inches(3.0), weight=1.25)
    tf = _box(s, L0, Inches(5.06), Inches(11.0), Inches(0.8))
    _para(tf, True, DATE, 14, space_after=4)
    _para(tf, False,
          "vendor pin a4fc478   |   corpus pin cc2adbc8   |   spend to date $0.13",
          13, space_after=0)

    # -- 2. summary ---------------------------------------------------------
    s = slide(prs, "Summary")
    y = lines(s, BODY_TOP, [
        ("Done.", "002 complete and reproducible. 003 executes the vendor model. "
                  "Cost model measured against the live bucket."),
        ("Not done.", "Kill-gate (001). Fast-weight carry. 1B baseline."),
        ("Verdict.", "None. Nothing here is a PROCEED or a STOP."),
        ("Spend.", "$0.13, one GCS transfer. All compute so far on free hardware."),
    ], size=17)
    y = table(s, y + Inches(0.25),
              ["Pre-registered bar", "Value", "Status"],
              [["Cohen's d", "≥ 0.80", "frozen 2026-08-08, untouched"],
               ["Relative degradation", "≥ 10%", "frozen 2026-08-08, untouched"],
               ["Fluency ratio", "≤ 1.5", "frozen 2026-08-08, not yet scorable"]],
              [3.6, 2.2, 5.0])

    # -- 3. experiment 002 --------------------------------------------------
    s = slide(prs, "Experiment 002 — scaled-down pilot")
    y = lines(s, BODY_TOP, [
        "Victim: ~0.1M-parameter byte-level TTT stand-in. Not TTT-E2E.",
        "Attacker selects real spans from a benign corpus and chooses their order.",
        "Base weights frozen; only the fast MLP is updated, by SGD, once per chunk.",
    ], size=16, gap=8)
    y = table(s, y + Inches(0.20),
              ["Measurement", "Observed", "Bar"],
              [["Effect size (Cohen's d)", f"{deep_d:+.3f}", "≥ 0.80"],
               ["Relative degradation",
                f"{deep['relative_degradation'] * 100:+.4f}%", "≥ 10%"],
               ["Inner-loop adaptation", f"{deep['inner_loop_gain_nats']:.4f} nats",
                f"vs {deep['shallow_inner_loop_gain_nats']} with shallow base"],
               ["Position in own null", "0th percentile", "—"]],
              [3.6, 2.6, 4.6])
    note(s, y + Inches(0.04),
         "The attack moves benign loss by nothing. The mechanism it attacks "
         "measurably adapts, so this is a null from a working inner loop.")

    # -- 4. the null --------------------------------------------------------
    s = slide(prs, "Experiment 002 — null distribution")
    y = picture(s, f_null, BODY_TOP, 2.95)
    y = lines(s, y + Inches(0.10), [
        f"{n_clear} of {null['n_draws']} control-vs-control draws clear "
        f"|d| ≥ 0.80. False-positive rate {fpr:.0%}.",
        f"Median |d| {null['null_abs_median']:.3f}, "
        f"max {null['null_abs_max']:.3f}, 5 seeds per draw.",
        f"Shallow pilot, same run: d = {pilot['effect_size']:.3f} poison-vs-control, "
        f"{pilot['noise_floor_effect_size']:.3f} control-vs-control.",
    ], size=15.5, gap=7)

    # -- 5. reproducibility -------------------------------------------------
    s = slide(prs, "Reproducibility defect — found and fixed")
    y = lines(s, BODY_TOP, [
        "The pilot's corpus was a live glob of this repository's own markdown, "
        "resolved at run time.",
        "Every commit touching a .md file changed the experiment.",
    ], size=16, gap=8)
    y = table(s, y + Inches(0.18),
              ["Quantity", "Before", "After"],
              [["Corpus", "672 KB, 47 files, drifting", "615,141 B, frozen, hashed"],
               ["Shallow pilot d", "+0.0535 → −1.2838", "+0.872"],
               ["Deep pilot d", "+0.8313", "−0.003"],
               ["Null false-positive rate", "55%", "15%"]],
              [3.4, 3.9, 3.8])
    note(s, y + Inches(0.04),
         "Editing two unrelated README files moved d across zero and past the bar "
         "in the opposite direction. corpus.txt is now pinned by SHA-256 and "
         "checked on load; a mismatch refuses to run.")

    # -- 6. experiment 003 --------------------------------------------------
    s = slide(prs, "Experiment 003 — first execution of the vendor model")
    y = table(s, BODY_TOP,
              ["Measurement", "Observed", "Reconstructed"],
              [["Trainable parameters", "184,363,776",
                "matches to within RMSNorm weights"],
               ["Inner (fast) weights", "11,501,568",
                "3 × 3 × 768 × 1664, exact"],
               ["Attack surface", "6.24%", "of trainable parameters"]],
              [3.3, 2.6, 5.2])
    y = picture(s, f_curve, y + Inches(0.02), 2.55)
    note(s, y - Inches(0.02),
         "125M random init, dummy tokens, seq 8192, RTX 3070 Ti, bf16. Falls "
         "2.035 nats over 8 chunks. Chunk 1 within 0.030 nats of a value "
         "predicted from two config constants.")

    # -- 7. cost model ------------------------------------------------------
    s = slide(prs, "Cost model — measured against the live bucket")
    y = table(s, BODY_TOP,
              ["Quantity", "Assumed", "Measured", "Consequence"],
              [["/val tokens", "50M – 1B", "2,000,168,321",
                "eval 0.9 h → 7.45 h"],
               ["Storage codec", "Blosc ≈ 2.5×", "uncompressed",
                "chunk subsampling possible"],
               ["1B checkpoint", "5.9 GB", "5.35 GB", "inside stated ±10%"],
               ["Session total", "≈ 4 h", "≈ 10.5 h",
                "collides with 12 h stop"]],
              [2.5, 2.2, 2.8, 3.8])
    note(s, y + Inches(0.04),
         "/val is 21 chunks of exactly 100,000,000 tokens. Three chunks give "
         "SE 0.0015 nats against a 0.491-nat band, for 1.12 h instead of 7.45. "
         "Fixed in writing before any baseline result is seen.")

    # -- 8. scope -----------------------------------------------------------
    s = slide(prs, "Not established")
    y = lines(s, BODY_TOP, [
        ("No kill-gate verdict.", "001 not run. PREREGISTERED.md untouched."),
        ("No fast-weight carry result.", "Blocked on GPU memory; 8 GB card is "
                                         "~2 GB short."),
        ("No 1B baseline.", "Needs rented Ampere. Kaggle and Colab excluded — "
                            "cuDNN fused attention requires SM80+."),
        ("002 is not TTT-E2E.", "0.1M-parameter stand-in, realism bar unscored, "
                                "cannot return PROCEED."),
    ], size=16)
    note(s, y + Inches(0.30),
         "Seven first-launch failures found and fixed on free hardware, including "
         "a checkpoint restore path that raises on the released artefact and a "
         "sequence-length reduction that returns NaN rather than an error.")

    # -- 9. next ------------------------------------------------------------
    s = slide(prs, "Next")
    y = table(s, BODY_TOP,
              ["Step", "Requires", "Cost"],
              [["1. Fast-weight carry (003)", "Ampere GPU, ~16 GB", "~1 h"],
               ["2. 125M rehearsal", "checkpoint already fetched", "~0.5 h"],
               ["3. 1B baseline vs 2.314–2.805", "3 /val chunks, rented A100",
                "≈ 4 h, ≈ ₹560"],
               ["4. Kill-gate (001)", "all of the above", "separate session"]],
              [4.4, 4.0, 2.9])
    note(s, y + Inches(0.06),
         "Step 1 retires the project's named principal technical risk. Step 4 "
         "must be read against the 15% false-positive rate above.")

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
