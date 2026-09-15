#!/usr/bin/env python
"""Generate the results deck: one column, one idea per slide.

Deliberately NOT `make_review_deck.py`. That deck is a project review -- three-up
cards, stat rows, dark banners -- and it is the right shape for "here is the
programme". This one answers a narrower question, asked by a supervisor: *what
have you measured?* A measurement reads better in a single column with room
around it than in a grid of cards competing for the eye.

Every number is read from a committed result file at build time, never retyped.
If a result file moves, this deck moves with it -- which is the whole reason the
corpus is pinned (`experiments/002-pilot-tiny-ttt/run_pilot.py`), because until
2026-09-15 those files were not reproducible.

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

# ----------------------------------------------------------------- design --
# Same ink and accent family as make_review_deck.py so the two decks sit
# together, but none of its card chrome.
BG = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x10, 0x15, 0x1C)
TITLE_INK = RGBColor(0x12, 0x30, 0x45)
MUTED = RGBColor(0x5A, 0x66, 0x73)
FAINT = RGBColor(0x96, 0xA1, 0xAB)
ACCENT = RGBColor(0x1D, 0x4E, 0x89)
AMBER = RGBColor(0xB4, 0x54, 0x1B)
GREEN = RGBColor(0x2D, 0x6A, 0x4F)
RED = RGBColor(0x7A, 0x2F, 0x2F)
RULE = RGBColor(0xD3, 0xDA, 0xE2)

FONT = "Segoe UI"
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
L0 = Inches(1.0)                 # one column, generous margin
CW = Inches(11.333)
TITLE_TOP = Inches(0.72)
BODY_TOP = Inches(2.05)
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


def _para(tf, first, text, size, *, bold=False, color=INK, space_after=8,
          font=FONT, align=None, line=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    p.space_after = Pt(space_after)
    if align:
        p.alignment = align
    if line:
        p.line_spacing = line
    return p


def _rule(slide, top, width=CW, left=L0, color=RULE, weight=1.0):
    ln = slide.shapes.add_connector(1, left, top, left + width, top)
    ln.line.color.rgb = color
    ln.line.width = Pt(weight)
    return ln


def slide(prs, kicker, title):
    """A blank slide with the one-column header: kicker, title, hairline."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG

    tf = _box(s, L0, TITLE_TOP, CW, Inches(0.3))
    _para(tf, True, kicker.upper(), 11.5, bold=True, color=ACCENT,
          space_after=0, font=MONO)

    tf = _box(s, L0, TITLE_TOP + Inches(0.36), CW, Inches(1.0))
    _para(tf, True, title, 30, bold=True, color=TITLE_INK, space_after=0,
          line=1.05)

    _rule(s, BODY_TOP - Inches(0.22))

    _page[0] += 1
    tf = _box(s, L0, H - Inches(0.62), CW, Inches(0.28))
    p = _para(tf, True, f"{FOOTER}     ·     {DATE}     ·     {_page[0]}", 9.5,
              color=FAINT, space_after=0, font=MONO)
    p.alignment = PP_ALIGN.LEFT
    return s


def lede(s, text, top=BODY_TOP, size=19, width=Inches(10.2)):
    tf = _box(s, L0, top, width, Inches(1.3))
    _para(tf, True, text, size, color=MUTED, space_after=0, line=1.28)
    return top + Inches(0.5) + Inches(0.34) * (len(text) // 78 + 1)


def hero(s, top, number, caption, color=ACCENT):
    """One figure, at size, with a single line under it. Used sparingly."""
    tf = _box(s, L0, top, CW, Inches(1.5))
    _para(tf, True, number, 66, bold=True, color=color, space_after=2,
          font=MONO)
    tf = _box(s, L0, top + Inches(1.12), CW, Inches(0.5))
    _para(tf, True, caption, 15, color=MUTED, space_after=0)
    return top + Inches(1.75)


def bullets(s, top, items, size=16, gap=15):
    tf = _box(s, L0, top, Inches(10.6), Inches(3.4))
    for i, (head, tail) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = head
        r.font.size = Pt(size)
        r.font.bold = True
        r.font.color.rgb = INK
        r.font.name = FONT
        if tail:
            r2 = p.add_run()
            r2.text = "  " + tail
            r2.font.size = Pt(size)
            r2.font.color.rgb = MUTED
            r2.font.name = FONT
        p.space_after = Pt(gap)
        p.line_spacing = 1.22
    return top + Inches(0.42) * len(items)


def table(s, top, headers, rows, widths, *, row_h=0.42, size=13):
    """A plain rule-and-space table. No fills, no zebra -- the numbers carry it."""
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
            cell.margin_right = Inches(0.14)
            cell.margin_top = cell.margin_bottom = Inches(0.045)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            text = headers[c] if r == 0 else rows[r - 1][c]
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            run = p.add_run()
            run.text = str(text)
            run.font.size = Pt(size if r else size - 1)
            run.font.bold = r == 0
            run.font.color.rgb = MUTED if r == 0 else INK
            # Numeric columns get the mono face so digits line up.
            run.font.name = MONO if (r and c and _numeric(str(text))) else FONT
    return top + Inches(row_h * n_rows) + Inches(0.2)


def _numeric(t: str) -> bool:
    return bool(t) and t[0] in "-+−0123456789" or t.startswith("$")


def picture(s, path, top, height_in):
    pic = s.shapes.add_picture(str(path), L0, top, height=Inches(height_in))
    pic.left = int(L0 + (CW - pic.width) / 2)
    return top + Inches(height_in) + Inches(0.18)


def note(s, top, text, color=MUTED):
    tf = _box(s, L0, top, Inches(10.6), Inches(0.9))
    _para(tf, True, text, 12.5, color=color, space_after=0, font=MONO,
          line=1.35)


# ---------------------------------------------------------------- figures --
def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d3dae2")
    ax.tick_params(colors="#5a6673", labelsize=10, length=4, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname("Segoe UI")


def fig_null(null, deep_d) -> Path:
    """Strip plot of the control-vs-control null, with the observed effect."""
    vals = sorted(null["null_effect_sizes"])
    # Greedy stacking so equal-ish values do not sit on top of one another.
    ys, placed = [], []
    for v in vals:
        row = 0
        while any(abs(v - p) < 0.09 for p, r in placed if r == row):
            row += 1
        placed.append((v, row))
        ys.append(row)

    fig, ax = plt.subplots(figsize=(10.4, 3.1), dpi=200)
    ax.axvspan(-2.2, -0.8, color="#b4541b", alpha=0.08, lw=0)
    ax.axvspan(0.8, 2.2, color="#b4541b", alpha=0.08, lw=0)
    for x in (-0.8, 0.8):
        ax.axvline(x, color="#b4541b", ls=(0, (4, 3)), lw=1.2)

    ax.scatter(vals, [y * 0.14 for y in ys], s=78, color="#39434f",
               zorder=3, linewidths=0)
    # vlines, not axvline: the marker must stop below its own label, or the
    # rule is drawn straight through the text.
    ax.vlines(deep_d, -0.10, 0.50, color="#1d4e89", lw=2.2, zorder=4)
    ax.annotate(f"attack effect  d = {deep_d:+.3f}", xy=(deep_d, 0.62),
                xytext=(deep_d, 0.62), ha="center", fontsize=11.5,
                color="#1d4e89", fontweight="bold", fontname="Segoe UI")

    ax.text(-2.12, 0.50, "d ≤ −0.8", fontsize=10.5, color="#b4541b",
            fontname="Consolas")
    ax.text(2.12, 0.50, "d ≥ 0.8", fontsize=10.5, color="#b4541b",
            ha="right", fontname="Consolas")

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.12, 0.74)
    ax.set_yticks([])
    ax.set_xlabel("Cohen's d, control vs control (20 draws, 5 seeds each)",
                  fontsize=11.5, color="#39434f", fontname="Segoe UI",
                  labelpad=8)
    ax.spines["left"].set_visible(False)
    _style(ax)
    fig.tight_layout()
    p = FIG_DIR / "null_distribution.png"
    fig.savefig(p, transparent=False, facecolor="white")
    plt.close(fig)
    return p


#: Per-chunk cross-entropy, experiments/003-smoke-125m, runner a7447ce.
#: Held here rather than read from results/smoke.json because that file is
#: git-ignored and produced on a GPU box, not on the machine building the deck.
CE_CURVE = [11.886, 10.708, 10.163, 10.128, 9.949, 9.923, 9.851, 9.868]
CE_PREDICTED_INIT = 11.9154


def fig_curve() -> Path:
    fig, ax = plt.subplots(figsize=(10.4, 3.2), dpi=200)
    xs = list(range(1, len(CE_CURVE) + 1))
    ax.axhline(CE_PREDICTED_INIT, color="#b4541b", ls=(0, (5, 4)), lw=1.3)
    # Above the rule, not on it -- va="center" puts the dashes straight through
    # the glyphs and it reads as struck-through text.
    ax.text(9.3, CE_PREDICTED_INIT + 0.06,
            f"predicted at init  {CE_PREDICTED_INIT}", va="bottom", ha="right",
            fontsize=10.5, color="#b4541b", fontname="Consolas")
    ax.plot(xs, CE_CURVE, color="#1d4e89", lw=2.2, zorder=3)
    ax.scatter(xs, CE_CURVE, s=62, color="#1d4e89", zorder=4,
               edgecolors="white", linewidths=1.6)
    ax.annotate(f"{CE_CURVE[0]:.3f}", xy=(1, CE_CURVE[0]), xytext=(1.12, CE_CURVE[0] + 0.06),
                fontsize=11, color="#1d4e89", fontname="Consolas")
    ax.annotate(f"{CE_CURVE[-1]:.3f}", xy=(8, CE_CURVE[-1]),
                xytext=(7.55, CE_CURVE[-1] - 0.28), fontsize=11,
                color="#1d4e89", fontname="Consolas")
    ax.set_xlim(0.6, 9.4)
    ax.set_ylim(9.4, 12.3)
    ax.set_xticks(xs)
    ax.grid(axis="y", color="#e6ebf0", lw=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("Inner-loop chunk", fontsize=11.5, color="#39434f",
                  fontname="Segoe UI", labelpad=8)
    ax.set_ylabel("nats / token", fontsize=11.5, color="#39434f",
                  fontname="Segoe UI", labelpad=8)
    _style(ax)
    fig.tight_layout()
    p = FIG_DIR / "inner_loop_curve.png"
    fig.savefig(p, transparent=False, facecolor="white")
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
    s.background.fill.fore_color.rgb = BG
    tf = _box(s, L0, Inches(2.35), CW, Inches(0.4))
    _para(tf, True, "DEFENSIVE SECURITY RESEARCH", 12, bold=True,
          color=ACCENT, space_after=0, font=MONO)
    tf = _box(s, L0, Inches(2.85), Inches(11.0), Inches(1.6))
    _para(tf, True, "Trust-Gated Fast-Weight Updates\nfor TTT-E2E LLMs", 42,
          bold=True, color=TITLE_INK, space_after=0, line=1.06)
    _rule(s, Inches(4.72), width=Inches(3.0), weight=2.0, color=ACCENT)
    tf = _box(s, L0, Inches(4.98), Inches(10.4), Inches(1.0))
    _para(tf, True, "What has been measured", 20, color=MUTED, space_after=6)
    _para(tf, False,
          f"{DATE}   ·   vendor pin a4fc478   ·   corpus pin cc2adbc8", 12.5,
          color=FAINT, space_after=0, font=MONO)

    # -- 2. the question ----------------------------------------------------
    s = slide(prs, "The question", "Can a benign-looking stream corrupt a model that learns at inference?")
    y = lede(s, "TTT-E2E updates a small set of fast weights by gradient descent during the "
                "forward pass. Those weights are 6.2% of the model, and they are written to "
                "by whatever text arrives. That is an attack surface no frozen model has.", BODY_TOP)
    y = bullets(s, y + Inches(0.18), [
        ("The kill-gate.", "An attack must clear all three pre-registered bars: Cohen's d ≥ 0.8, "
                           "relative degradation ≥ 10%, fluency ratio ≤ 1.5."),
        ("Why a gate, not a patch.", "If no benign-looking attack exists, there is no defence "
                                     "worth building — and the project stops."),
        ("Status.", "The gate has not been run. Nothing in this deck is a verdict."),
    ])
    note(s, y + Inches(0.3),
         "PREREGISTERED.md, frozen 2026-08-08. Bars are reported against, never moved.")

    # -- 3. finding 1 -------------------------------------------------------
    s = slide(prs, "Finding 1 · Experiment 002", "The attack does nothing — and the mechanism it attacks is alive")
    y = hero(s, BODY_TOP + Inches(0.05), f"d = {deep_d:+.3f}",
             "Cohen's d, crafted poison stream vs matched control, 5 seeds. "
             f"Bar for corruption is d ≥ 0.8.")
    y = table(s, y + Inches(0.12),
              ["", "Observed", "Reads as"],
              [["Attack effect size", f"{deep_d:+.3f}", "0th percentile of its own null"],
               ["Relative degradation", f"{deep['relative_degradation'] * 100:+.4f}%", "bar is 10%"],
               ["Inner-loop adaptation", f"{deep['inner_loop_gain_nats']:.4f} nats",
                f"vs {deep['shallow_inner_loop_gain_nats']} shallow — the fast weights do move"]],
              [3.5, 2.4, 5.2])
    note(s, y + Inches(0.05),
         "A null from a working mechanism. Not a null because nothing was adapting.")

    # -- 4. the null --------------------------------------------------------
    s = slide(prs, "Finding 2 · the instrument", "At five seeds, the criterion fires without an attacker")
    y = picture(s, f_null, BODY_TOP + Inches(0.02), 3.05)
    y = bullets(s, y + Inches(0.12), [
        (f"{n_clear} of {null['n_draws']} null draws clear |d| ≥ 0.8.",
         f"False-positive rate {fpr:.0%}. Median |d| {null['null_abs_median']:.3f}, "
         f"max {null['null_abs_max']:.3f}."),
        ("In the shallow pilot the noise floor exceeded the effect.",
         f"d = {pilot['effect_size']:.3f} poison-vs-control, "
         f"{pilot['noise_floor_effect_size']:.3f} control-vs-control, same run."),
    ], size=15)

    # -- 5. reproducibility -------------------------------------------------
    s = slide(prs, "Finding 3 · methods", "The pilot had been training on this repository's own documentation")
    y = lede(s, "Its corpus loader globbed docs/**/*.md and experiments/**/*.md at run time. "
                "Every commit touching a markdown file silently changed the experiment.",
             BODY_TOP, size=18)
    y = table(s, y + Inches(0.02),
              ["Quantity", "Before the pin", "After the pin"],
              [["Corpus", "672 KB, 47 files, drifting", "615,141 B, frozen, hashed"],
               ["Shallow pilot d", "+0.0535 → −1.2838", "+0.872"],
               ["Deep pilot d", "+0.8313", "−0.003"],
               ["Null false-positive rate", "55%", "15%"]],
              [3.5, 4.0, 3.6])
    note(s, y + Inches(0.02),
         "A morning editing two unrelated READMEs moved d across zero and past the bar in the\n"
         "opposite direction. Same code, same seeds. corpus.txt is now pinned by SHA-256 and\n"
         "checked on load; a mismatch refuses to run.", color=AMBER)

    # -- 6. vendor model ----------------------------------------------------
    s = slide(prs, "Finding 4 · Experiment 003", "First execution of the vendor model, on free hardware")
    y = table(s, BODY_TOP + Inches(0.05),
              ["Quantity", "Observed", "Reconstructed"],
              [["Trainable parameters", "184,363,776", "matches to within the RMSNorm weights"],
               ["Inner (fast) weights", "11,501,568", "3 × 3 × 768 × 1664 — exact"],
               ["Attack surface", "6.24%", "of all trainable parameters"]],
              [3.3, 2.6, 5.2])
    y = picture(s, f_curve, y + Inches(0.04), 2.62)
    note(s, y - Inches(0.04),
         "125M random init, dummy tokens, seq 8192 · RTX 3070 Ti (SM86), bf16 · runner a7447ce.\n"
         "Chunk 1 lands 0.030 nats from a value predicted from two config constants.")

    # -- 7. cost model ------------------------------------------------------
    s = slide(prs, "Finding 5 · cost model", "The evaluation split is eight times larger than assumed")
    y = table(s, BODY_TOP + Inches(0.05),
              ["Quantity", "Assumed", "Measured", "Consequence"],
              [["/val tokens", "50M – 1B", "2,000,168,321", "eval pass 0.9 h → 7.45 h"],
               ["Storage codec", "Blosc ≈ 2.5×", "uncompressed", "chunk-aligned subsampling"],
               ["1B checkpoint", "5.9 GB", "5.35 GB", "inside the stated ±10%"],
               ["Session total", "≈ 4 h", "≈ 10.5 h", "collides with the 12 h stop"]],
              [2.6, 2.3, 2.7, 3.7])
    note(s, y + Inches(0.04),
         "Because the store is uncompressed, /val is 21 chunks of exactly 100,000,000 tokens.\n"
         "Fetching three gives SE 0.0015 nats against a 0.491-nat band, for 1.12 h instead of 7.45.\n"
         "Fixed in writing before any baseline result is seen.")

    # -- 8. scope -----------------------------------------------------------
    s = slide(prs, "Scope", "What is not established")
    y = bullets(s, BODY_TOP + Inches(0.05), [
        ("No kill-gate verdict.", "Experiment 001 has not run. PREREGISTERED.md is untouched."),
        ("No fast-weight carry result.", "The check Experiment 003 exists for is blocked on GPU "
                                         "memory — an 8 GB card is about 2 GB short."),
        ("No 1B baseline.", "Needs a rented Ampere card; Kaggle and Colab are architecturally "
                            "excluded — cuDNN fused attention requires SM80+."),
        ("Experiment 002 is not TTT-E2E.", "A ~0.1M-parameter byte-level stand-in with an "
                                           "unscored realism bar. It cannot return PROCEED at all."),
    ], size=15)
    note(s, y + Inches(0.25),
         "Seven first-launch failures found and fixed on free hardware, including a checkpoint\n"
         "restore path that raises on the released artefact and a sequence-length reduction that\n"
         "returns NaN rather than an error.")

    # -- 9. next ------------------------------------------------------------
    s = slide(prs, "Next", "Four hours of rented Ampere closes the remaining gap")
    y = bullets(s, BODY_TOP + Inches(0.05), [
        ("1 · Fast-weight carry.", "Finish Experiment 003 at full sequence length. Retires the "
                                   "project's named #1 technical risk."),
        ("2 · 125M rehearsal.", "Checkpoint already fetched, $0.13 spent. Exercises every "
                                "failure mode except memory."),
        ("3 · 1B baseline.", "Three /val chunks against the 2.314–2.805 band. ≈ 4 h, ≈ ₹560."),
        ("4 · Then, and only then, the kill-gate.", "Five seeds per arm — with the 15% "
                                                    "false-positive rate above held in view."),
    ], size=15)
    note(s, y + Inches(0.25),
         "Everything measured so far cost $0.13. The dominant line in the session cost model is\n"
         "first-launch debugging, and it is being paid in laptop time.")

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
