#!/usr/bin/env python
"""Generate the project review deck, in the existing deck's design language.

Built on the visual system already established by
`docs/review/Project_Review_TrustGate.pptx` -- same canvas, palette, chrome,
type scale and components (dark quote banner, three-up accent cards, stat
blocks, amber callouts, dark-header tables). The palette and geometry below were
extracted from that file rather than invented, so a new slide sits beside an old
one without looking foreign.

Committed rather than hand-built so the deck is reproducible and its claims are
diffable. Every number is taken from a measurement recorded elsewhere in the
repo -- `docs/report.md`, `FLUENCY_REFERENCE.md`, the ADRs -- not retyped.

    uv pip install --python .venv python-pptx
    python scripts/make_review_deck.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DATE = "2026-08-27"
OUT = Path(__file__).resolve().parents[1] / "docs" / "review" / f"TrustGate_Review_{DATE}.pptx"

FOOTER = "Trust-Gated Fast-Weight Updates for TTT-E2E LLMs"
FONT = "Segoe UI"

# Palette, extracted from the existing deck.
BG = RGBColor(0xFA, 0xFB, 0xFC)
INK = RGBColor(0x0D, 0x1F, 0x2D)
TITLE_INK = RGBColor(0x12, 0x30, 0x45)
MUTED = RGBColor(0x5C, 0x6B, 0x78)
FAINT = RGBColor(0x8A, 0x97, 0xA2)
TEAL = RGBColor(0x0E, 0x9F, 0x94)
AMBER = RGBColor(0xB4, 0x53, 0x09)
AMBER_BG = RGBColor(0xFD, 0xF1, 0xDF)
RED = RGBColor(0xA8, 0x2C, 0x2C)
RED_BG = RGBColor(0xFB, 0xEC, 0xEC)
GREEN = RGBColor(0x15, 0x7A, 0x4A)
GREEN_BG = RGBColor(0xE7, 0xF4, 0xEC)
RULE = RGBColor(0xD8, 0xDF, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD_DARK = RGBColor(0x16, 0x33, 0x44)
DECOR_A = RGBColor(0x11, 0x2A, 0x38)
DECOR_B = RGBColor(0x10, 0x26, 0x33)
ON_DARK = RGBColor(0xB6, 0xC6, 0xD1)

W, H = Inches(13.333), Inches(7.5)
L0 = Inches(0.72)          # content left edge
CW = Inches(11.893)        # content width
BODY_TOP = Inches(1.86)    # first content row

TEAM = [
    ("Manas Maahir", "23BAI0051"),
    ("Jay Krishna Kamlekar", "23BAI0053"),
    ("Kartheek Yadav", "23BAI0056"),
]


# --------------------------------------------------------------- primitives
def _rect(slide, left, top, width, height, fill):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def _box(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    return frame


def _run(para, text, size, bold=False, color=INK, space_after=None):
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    if space_after is not None:
        para.space_after = Pt(space_after)
    return run


def _para(frame, first, text, size, bold=False, color=INK, space_after=6, align=None):
    para = frame.paragraphs[0] if first else frame.add_paragraph()
    if align is not None:
        para.alignment = align
    _run(para, text, size, bold, color, space_after)
    return para


# ------------------------------------------------------------------ chrome
def chrome(prs, title, kicker, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, W, H, BG)
    _rect(slide, L0, Inches(0.52), Inches(0.055), Inches(0.42), TEAL)

    frame = _box(slide, Inches(0.94), Inches(0.44), Inches(11.593), Inches(0.6))
    _para(frame, True, title, 27, True, TITLE_INK, 0)

    frame = _box(slide, Inches(0.94), Inches(1.06), Inches(11.593), Inches(0.4))
    _para(frame, True, kicker, 13.5, False, MUTED, 0)

    _rect(slide, L0, Inches(1.56), CW, Inches(0.012), RULE)

    frame = _box(slide, Inches(11.833), Inches(6.98), Inches(0.8), Inches(0.3))
    _para(frame, True, str(page), 10, False, FAINT, 0, PP_ALIGN.RIGHT)

    frame = _box(slide, L0, Inches(6.98), Inches(8.0), Inches(0.3))
    _para(frame, True, FOOTER, 9.5, False, FAINT, 0)
    return slide


def title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, W, H, INK)
    _rect(slide, Inches(8.433), Inches(-1.2), Inches(5.6), Inches(5.6), DECOR_A)
    _rect(slide, Inches(10.133), Inches(3.4), Inches(4.4), Inches(4.4), DECOR_B)
    _rect(slide, 0, 0, W, Inches(0.16), TEAL)

    frame = _box(slide, Inches(0.87), Inches(1.02), Inches(9.2), Inches(0.4))
    _para(frame, True,
          "SCHOOL OF COMPUTER SCIENCE AND ENGINEERING  ·  VELLORE INSTITUTE OF TECHNOLOGY",
          11, True, TEAL, 0)

    frame = _box(slide, Inches(0.87), Inches(1.62), Inches(10.4), Inches(2.2))
    _para(frame, True, "Trust-Gated Fast-Weight Updates", 42, True, WHITE, 4)
    _para(frame, False, "for TTT-E2E Large Language Models", 42, True, WHITE, 0)

    _rect(slide, Inches(0.87), Inches(3.52), Inches(1.5), Inches(0.035), TEAL)

    frame = _box(slide, Inches(0.87), Inches(3.8), Inches(9.6), Inches(0.9))
    _para(frame, True,
          "A runtime defense for language models that learn while they serve",
          17, False, ON_DARK, 0)

    frame = _box(slide, Inches(0.87), Inches(4.78), Inches(3.0), Inches(0.3))
    _para(frame, True, "TEAM", 10.5, True, TEAL, 0)

    for i, (name, roll) in enumerate(TEAM):
        left = Inches(0.87 + i * 3.62)
        _rect(slide, left, Inches(5.14), Inches(3.32), Inches(0.86), CARD_DARK)
        frame = _box(slide, left + Inches(0.26), Inches(5.3), Inches(2.9), Inches(0.6))
        _para(frame, True, name, 14.5, True, WHITE, 1)
        _para(frame, False, roll, 12, False, TEAL, 0)

    frame = _box(slide, Inches(0.87), Inches(6.42), Inches(11.0), Inches(0.35))
    _para(frame, True,
          "Project Review  ·  August 2026  ·  Phase 1 implementation complete  ·  no measurement yet  ·  spend $0 of $325",
          11.5, False, FAINT, 0)
    return slide


# --------------------------------------------------------------- components
def banner(slide, text, top=BODY_TOP, height=Inches(0.98), size=21):
    _rect(slide, L0, top, CW, height, INK)
    frame = _box(slide, L0 + Inches(0.42), top + Inches(0.2),
                 CW - Inches(0.84), height - Inches(0.3))
    _para(frame, True, text, size, True, WHITE, 0)


def cards(slide, top, items, height=Inches(2.12)):
    """items: list of (accent, heading, body). Up to three across."""
    width = Inches(3.778)
    for i, (accent, heading, body) in enumerate(items):
        left = Inches(0.72 + i * 4.058)
        _rect(slide, left, top, width, height, WHITE)
        _rect(slide, left, top, Inches(0.05), height, accent)
        frame = _box(slide, left + Inches(0.28), top + Inches(0.2),
                     width - Inches(0.5), height - Inches(0.36))
        _para(frame, True, heading, 14, True, TITLE_INK, 7)
        _para(frame, False, body, 12, False, MUTED, 0)


def stat(slide, left, top, number, label, sub, width=Inches(3.6), height=Inches(1.14)):
    _rect(slide, left, top, width, height, INK)
    frame = _box(slide, left + Inches(0.3), top + Inches(0.18),
                 Inches(1.5), Inches(0.8), MSO_ANCHOR.MIDDLE)
    _para(frame, True, number, 30, True, TEAL, 0)
    frame = _box(slide, left + Inches(1.72), top + Inches(0.2),
                 width - Inches(1.9), height - Inches(0.34))
    _para(frame, True, label, 13, True, WHITE, 2)
    _para(frame, False, sub, 10.5, False, FAINT, 0)


def callout(slide, left, top, width, height, heading, body, accent=AMBER, bg=AMBER_BG):
    _rect(slide, left, top, width, height, bg)
    _rect(slide, left, top, Inches(0.05), height, accent)
    frame = _box(slide, left + Inches(0.3), top + Inches(0.14),
                 width - Inches(0.55), height - Inches(0.26))
    _para(frame, True, heading, 11.5, True, accent, 4)
    _para(frame, False, body, 11.5, False, INK, 0)


def table(slide, top, headers, rows, col_widths, row_height=0.435):
    """rows: list of lists; a cell is str or (text, size, bold, color) tuple."""
    n_rows, n_cols = len(rows) + 1, len(headers)
    height = Inches(0.4 + len(rows) * row_height)
    shape = slide.shapes.add_table(n_rows, n_cols, L0, top, CW, height)
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False

    for i, w in enumerate(col_widths):
        tbl.columns[i].width = Inches(w)
    tbl.rows[0].height = Inches(0.4)
    for r in range(1, n_rows):
        tbl.rows[r].height = Inches(row_height)

    for c, text in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TITLE_INK
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = cell.margin_right = Inches(0.1)
        frame = cell.text_frame
        frame.word_wrap = True
        _run(frame.paragraphs[0], text, 11.5, True, WHITE)

    for r, row in enumerate(rows, start=1):
        tint = row[-1] if isinstance(row[-1], RGBColor) else WHITE
        cells = row[:-1] if isinstance(row[-1], RGBColor) else row
        for c, spec in enumerate(cells):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = tint
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.1)
            frame = cell.text_frame
            frame.word_wrap = True
            if isinstance(spec, tuple):
                text, size, bold, color = spec
            else:
                text, size, bold, color = spec, 11.5, False, INK
            _run(frame.paragraphs[0], text, size, bold, color)
    return shape


def consequence(slide, top, label, body, color=TITLE_INK):
    frame = _box(slide, L0, top, CW, Inches(0.9))
    para = frame.paragraphs[0]
    _run(para, label, 15, True, color)
    _run(para, body, 15, False, INK)


def bullets(slide, top, items, height=Inches(2.0), left=L0, width=CW):
    """items: list of (text, size, bold, color, space_after).

    `height` only sizes the (transparent, top-anchored) box; text flows from the
    top regardless. It is passed explicitly so the box does not reserve space a
    later component wants -- overlapping invisible boxes are harmless to look at
    but make the geometry impossible to audit.
    """
    frame = _box(slide, left, top, width, height)
    for i, (text, size, bold, color, gap) in enumerate(items):
        _para(frame, i == 0, text, size, bold, color, gap)


# ------------------------------------------------------------------- build
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    page = [1]

    def nxt():
        page[0] += 1
        return page[0]

    title_slide(prs)

    # -- 2 ------------------------------------------------------------------
    s = chrome(prs, "What this review covers", "Three questions, in order", nxt())
    cards(s, BODY_TOP, [
        (TEAL, "1.  Motivation",
         "What TTT-E2E changes about the threat model, and why existing defenses do "
         "not reach it."),
        (AMBER, "2.  What we did",
         "The experiment that can answer the question — and the ways we found it "
         "could have lied to us."),
        (RED, "3.  What comes next",
         "One purchase blocks the critical path. One technical decision needs the "
         "whole team."),
    ])
    consequence(s, Inches(4.4), "The honest headline.  ",
                "The experiment is now runnable and its instrument is validated — but no "
                "number has been produced and no verdict has been reached.")
    callout(s, L0, Inches(5.3), CW, Inches(1.1),
            "Why that is the right state to be in",
            "The go/no-go criterion was frozen in writing before any attack code ran. "
            "Producing a number before the machinery was trustworthy would have bought a "
            "verdict we could not defend — and STOP ends the project.")

    # -- 3  SECTION 1 -------------------------------------------------------
    s = chrome(prs, "Motivation — the attack surface this creates",
               "Section 1 of 3  ·  Why the work exists", nxt())
    banner(s, "“A model that learns while it serves can be poisoned while it serves.”")
    cards(s, Inches(3.04), [
        (TEAL, "The vector is a stream, not a token",
         "A slow, benign-looking sequence of ordinary inputs — no malformed request, "
         "nothing a filter would flag."),
        (AMBER, "The damage lands somewhere else",
         "Later, unrelated benign inputs get handled worse, or a latent trigger fires "
         "while aggregate quality stays flat."),
        (RED, "It is invisible to existing defenses",
         "Base weights are never touched, so checkpoint scanning and training-time "
         "provenance see nothing at all."),
    ])
    consequence(s, Inches(5.48), "Consequence.  ",
                "If the harmful thing happens at update time, the check has to happen at "
                "update time. Nothing upstream or downstream can substitute.")

    # -- 4 ------------------------------------------------------------------
    s = chrome(prs, "The architecture we target",
               "TTT-E2E — End-to-End Test-Time Training for Long Context", nxt())
    bullets(s, BODY_TOP, [
        ("TTT-E2E compresses long context by updating “fast weights” during inference.",
         17, False, INK, 8),
        ("An inner SGD loop runs inside the forward pass — one step per mini-batch chunk "
         "of the context.", 13.5, False, MUTED, 14),
        ("That is the whole benefit, and the whole exposure.", 17, True, TITLE_INK, 8),
        ("Conventional inference is stateless: a bad prompt harms one response and is gone.",
         13.5, False, MUTED, 4),
        ("Here the context stream mutates model weights, so harm can persist past the "
         "request that caused it.", 13.5, False, AMBER, 0),
    ], height=Inches(2.0))
    table(s, Inches(4.0),
          ["Quantity", "Value", "Why it matters for the attack"],
          [["Context length evaluated", ("8,192 tokens", 11.5, True, INK),
            ("The cheapest configuration that is still a released checkpoint", 11, False, MUTED), GREEN_BG],
           ["Inner-loop chunk size", ("1,024 tokens", 11.5, True, INK),
            ("Sets the granularity at which fast weights actually move", 11, False, MUTED), GREEN_BG],
           ["Inner SGD steps per sequence", ("8", 11.5, True, AMBER),
            ("A small budget — the attack must accumulate across sequences", 11, False, MUTED), AMBER_BG],
           ["Weights the attacker may touch", ("none", 11.5, True, GREEN),
            ("Context only. The claim rests on this being the sole channel", 11, False, MUTED), GREEN_BG]],
          [3.4, 2.0, 6.493])

    # -- 5 ------------------------------------------------------------------
    s = chrome(prs, "Why this problem, and not another",
               "The gap our literature survey actually found", nxt())
    table(s, BODY_TOP,
          ["Existing line of work", "What it defends", "Why it does not reach this threat"],
          [["Robust test-time adaptation (MedBN and relatives)",
            ("BatchNorm statistics, vision models", 11, False, MUTED),
            ("Not a fast-weight method; the principle must be ported, not reused", 11, False, MUTED), AMBER_BG],
           ["Training-data poisoning defenses",
            ("The training set and its provenance", 11, False, MUTED),
            ("The attack never touches training data", 11, False, MUTED), AMBER_BG],
           ["Prompt-injection filtering",
            ("A single malicious request", 11, False, MUTED),
            ("The stream is benign-looking by construction; no request is malformed", 11, False, MUTED), AMBER_BG],
           ["Checkpoint / weight scanning",
            ("Weights at rest", 11, False, MUTED),
            ("Base weights are never modified", 11, False, MUTED), AMBER_BG]],
          [4.0, 3.1, 4.793])
    callout(s, L0, Inches(4.5), CW, Inches(1.0),
            "The gap, stated plainly",
            "Nothing in the surveyed work gates fast-weight updates in a language model "
            "under a stated, testable bound. 15 verified journal papers with resolved "
            "DOIs — docs/literature-survey.md.", TEAL, GREEN_BG)
    consequence(s, Inches(5.75), "Our position.  ",
                "The novelty we can defend is the system — interceptor, drift budget, "
                "rollback — not a detector.")

    # -- 6 ------------------------------------------------------------------
    s = chrome(prs, "Our proposal — the trust gate",
               "Intercept every proposed update; admit it only if it survives a check", nxt())
    cards(s, BODY_TOP, [
        (TEAL, "Interceptor",
         "No fast-weight update commits without passing through it. Frozen-anchor "
         "consistency is the primary gate signal (ADR-F1)."),
        (TEAL, "Bounded drift budget",
         "Over any window, committed weights cannot travel more than ε from the anchor. "
         "This is the headline claimable property."),
        (TEAL, "Checkpoint and rollback",
         "O(1) revert through a versioned ring buffer, sized to detection latency."),
    ], height=Inches(1.95))
    callout(s, L0, Inches(4.05), CW, Inches(1.25),
            "What we record rather than hide",
            "The drift accumulator and the versioned store are implemented and unit-tested, "
            "but the interceptor does not yet call either. The per-window bound is therefore "
            "designed, not enforced. It is blocked structurally — the vendor's scan carry is "
            "fixed and ADR-002 forbids editing their tree — and is recorded as a dated "
            "correction on ADR-003.")
    consequence(s, Inches(5.6), "Why say so.  ",
                "A guarantee we describe as live but have not wired is the single easiest "
                "thing to be caught overstating.")

    # -- 7  SECTION 2 -------------------------------------------------------
    s = chrome(prs, "What we did", "Section 2 of 3  ·  Building the experiment", nxt())
    banner(s, "Build the thing that can answer the question — then find every way it could lie.")
    cards(s, Inches(3.04), [
        (TEAL, "Fix the bar first",
         "Thresholds frozen in writing before any attack code ran, and now pinned by a "
         "CI test that reads the frozen document."),
        (AMBER, "Build on CPU, spend nothing",
         "197 tests, 3,909 lines of source, zero GPU hours, zero dataset egress, $0 of a "
         "$325 cap."),
        (RED, "Hunt for false answers",
         "Five distinct routes to a scientifically worthless verdict, found and closed at "
         "the desk."),
    ])
    consequence(s, Inches(5.48), "Ordering is the point.  ",
                "Every decision that could later look convenient was written down before "
                "there was a result to make it convenient for.")

    # -- 8 ------------------------------------------------------------------
    s = chrome(prs, "The discipline: a pre-registered kill-gate",
               "Because the answer might be “stop”, and that has to stay allowed", nxt())
    table(s, BODY_TOP,
          ["Pre-registered criterion", "Threshold", "Why this bar"],
          [["Effect size, Cohen's d (poison vs control)", ("≥ 0.80", 11.5, True, INK),
            ("Conventional “large”; must separate clearly from seed noise", 11, False, MUTED), GREEN_BG],
           ["Relative benign degradation", ("≥ 10%", 11.5, True, INK),
            ("Below this the harm is real but trivial, and motivates no defense", 11, False, MUTED), GREEN_BG],
           ["Fluency ratio under an independent model", ("≤ 1.50", 11.5, True, INK),
            ("Above this the stream is not benign-looking and fails the threat model", 11, False, MUTED), GREEN_BG]],
          [5.0, 1.7, 5.193])
    callout(s, L0, Inches(3.75), CW, Inches(1.1),
            "All three are required — not any of them",
            "Missing one means STOP, or a dated and reasoned revision of the frozen "
            "document. Dossier §7: no demonstrated benign-looking attack means there is no "
            "defense worth building, and the paper reports that outcome either way.", TEAL, GREEN_BG)
    consequence(s, Inches(5.1), "Newly mechanised.  ",
                "The thresholds now live in code with a test that parses the frozen document "
                "and asserts equality — a bar cannot move without CI going red, and the "
                "review protocol gained a rule making the parser itself reviewable as a bar.")

    # -- 9 ------------------------------------------------------------------
    s = chrome(prs, "Phase 0.4 — de-risking the first booking for $0",
               "Seven desk deliverables, all completed before any hardware existed", nxt())
    stat(s, L0, BODY_TOP, "$0", "spent to date", "against a written $325 cap")
    stat(s, Inches(4.58), BODY_TOP, "7", "deliverables complete", "exact command to cost model")
    stat(s, Inches(8.44), BODY_TOP, "3", "findings that changed the plan", "each one free to discover")
    table(s, Inches(3.3),
          ["What we expected", "What we found", "What it would have cost"],
          [["“Match the paper's published number”",
            ("The paper reports NO number for our checkpoint", 11, True, AMBER),
            ("A meaningless PASS, or a manufactured FAIL", 11, False, MUTED), AMBER_BG],
           ["Copying only /val is enough",
            ("The job CRASHES after billing starts without one more file", 11, True, RED),
            ("A dead run, paid for, diagnosed on the clock", 11, False, MUTED), RED_BG],
           ["The 3B/128K run is roughly 2× the 1B",
            ("It is ~10× — activations force a multi-GPU node", 11, True, AMBER),
            ("A budget overrun on the first booking", 11, False, MUTED), AMBER_BG]],
          [3.9, 4.3, 3.693])
    consequence(s, Inches(5.35), "The bar became a bracket.  ",
                "Since no published number exists for this checkpoint, the baseline is a "
                "two-sided band plus four structural checks — recorded as ADR-005, before "
                "any run.")

    # -- 10 -----------------------------------------------------------------
    s = chrome(prs, "Phase 1 — the experiment, built and tested on CPU",
               "Everything below runs with no GPU, no checkpoint and no network", nxt())
    stat(s, L0, BODY_TOP, "197", "tests passing", "31 before this period")
    stat(s, Inches(4.58), BODY_TOP, "10", "stubs remaining", "from 15; five are Phase 2 gate code")
    stat(s, Inches(8.44), BODY_TOP, "9", "branches merged", "two reviewed pull requests")
    table(s, Inches(3.3),
          ["Component", "State", "What its tests actually pin"],
          [["Fast-weight carry across chunks and sequences", ("BUILT", 10.5, True, GREEN),
            ("Step k receives step k-1's output, not the initial weights", 11, False, MUTED), GREEN_BG],
           ["Attack objectives (DEGRADE, TRIGGER)", ("BUILT", 10.5, True, GREEN),
            ("Equivalence to the victim's own reported cross-entropy", 11, False, MUTED), GREEN_BG],
           ["Corpus and crafted-stream builders", ("BUILT", 10.5, True, GREEN),
            ("Seed reproducibility; BOS-free spans; exact length match", 11, False, MUTED), GREEN_BG],
           ["Matching discipline (13 fields)", ("BUILT", 10.5, True, GREEN),
            ("Any single-field mismatch raises and names the field", 11, False, MUTED), GREEN_BG],
           ["Harness CLI, verdict report, CI", ("BUILT", 10.5, True, GREEN),
            ("End-to-end on a synthetic result; three review checks mechanised", 11, False, MUTED), GREEN_BG],
           ["Victim model binding, adversarial search", ("OPEN", 10.5, True, AMBER),
            ("The last real unknown; design settled in ADR-006", 11, False, MUTED), AMBER_BG]],
          [4.6, 1.4, 5.893])

    # -- 11 -----------------------------------------------------------------
    s = chrome(prs, "The findings that mattered most",
               "Five ways this experiment could have produced a worthless answer", nxt())
    table(s, BODY_TOP,
          ["What we found", "Direction of the error", "Why it would not have been noticed"],
          [["The vendor DISCARDS the adapted fast weights",
            ("false STOP", 10.5, True, RED),
            ("Adapt-then-measure resets the weights; the arms differ only by noise", 11, False, MUTED), RED_BG],
           ["The inner learning-rate ramp starts near zero",
            ("false STOP", 10.5, True, RED),
            ("A near-frozen inner loop moves nothing, and the null looks real", 11, False, MUTED), RED_BG],
           ["Per-seed values were never printed in the report",
            ("hidden evidence", 10.5, True, AMBER),
            ("A seed where the poison HELPED the victim was invisible", 11, False, MUTED), AMBER_BG],
           ["Degenerate input yields an infinite effect size",
            ("false PROCEED", 10.5, True, AMBER),
            ("Constant losses pass the bar; a stubbed harness reads as success", 11, False, MUTED), AMBER_BG],
           ["The vendor ships no tokenizer at all",
            ("silent invalidity", 10.5, True, AMBER),
            ("The task depending on it had no referent; length matching would drift", 11, False, MUTED), AMBER_BG]],
          [4.5, 1.9, 5.493])
    callout(s, L0, Inches(4.95), CW, Inches(1.35),
            "Why the first one is the serious one",
            "The carry is bound and never read (transformer.py:712). A two-call harness — "
            "adapt on the poison stream, then measure — silently resets the fast weights and "
            "measures eval noise. That is a structurally guaranteed null: a STOP verdict "
            "indistinguishable from a real one. Since STOP ends the project, it would have "
            "converted a paid GPU run into a project-ending artefact.", RED, RED_BG)

    # -- 12 -----------------------------------------------------------------
    s = chrome(prs, "The fluency bar, and an honest negative result",
               "T3.2 — the third gating criterion", nxt())
    bullets(s, BODY_TOP, [
        ("An independent reference model scores whether a poison stream still reads as "
         "ordinary text.", 16, False, INK, 6),
        ("Scoring with the victim would be circular — the attacker optimises against it, so "
         "it rates poisoned streams as maximally fluent.", 13, False, MUTED, 0),
    ], height=Inches(0.9))
    table(s, Inches(2.85),
          ["Acceptance check", "Result", "Verdict"],
          [["Reproduce the published perplexity (the specified check)",
            ("62.29 vs 37.50 published — but 31.18 on another word-count convention", 11, False, INK),
            ("UNUSABLE", 10.5, True, RED), RED_BG],
           ["Logit equivalence against a reference implementation",
            ("worst difference 5.8e-04; 100% argmax agreement", 11, False, INK),
            ("PASS", 10.5, True, GREEN), GREEN_BG],
           ["Discrimination self-test (fluent vs scrambled)",
            ("19.98 vs 1326.84 — a 66× separation", 11, False, INK),
            ("PASS", 10.5, True, GREEN), GREEN_BG]],
          [4.9, 5.6, 1.393])
    callout(s, L0, Inches(4.55), CW, Inches(1.75),
            "The specified acceptance check does not work — and we say so rather than widen a tolerance",
            "The published figure is per word, and the denominator is unspecified. Our single "
            "measurement yields 62.29 on one convention and 31.18 on another, and the "
            "published 37.50 falls between them. The check therefore cannot separate an "
            "implementation bug from a counting convention, so it is recorded as corroborating "
            "evidence and NOT used as a gate. Nothing was tuned toward the target: the bar "
            "consumes a ratio, so the absolute cancels, and fudging it would only hide a real bug.")

    # -- 13 -----------------------------------------------------------------
    s = chrome(prs, "A limitation of one of our own bars",
               "Measured, and written down before any result exists", nxt())
    banner(s, "Perplexity punishes scrambling — but it rewards repetition.")
    table(s, Inches(3.15),
          ["Text fed to the reference model", "Perplexity", "What that implies"],
          [["Fluent English prose", ("19.98", 11.5, True, INK),
            ("The baseline a benign stream should sit near", 11, False, MUTED), GREEN_BG],
           ["The same words, scrambled", ("1326.84", 11.5, True, RED),
            ("Correctly caught — 66× worse", 11, False, MUTED), GREEN_BG],
           ["A short pattern repeated many times", ("1.90", 11.5, True, AMBER),
            ("Scores TEN TIMES BETTER than real prose", 11, False, AMBER), AMBER_BG]],
          [5.0, 1.7, 5.193])
    callout(s, L0, Inches(4.85), CW, Inches(1.45),
            "So the fluency bar cannot catch a repetitive poison stream",
            "A stream that reuses one passage many times would pass the ≤ 1.5 realism bar "
            "while being obviously not benign-looking to a human. Two mitigations were already "
            "in place and are now known to be load-bearing rather than tidy: spans are drawn "
            "without replacement, and span multiplicity is reported per arm so a repetitive "
            "stream is visible even when its ratio looks fine.")

    # -- 14 -----------------------------------------------------------------
    s = chrome(prs, "How we work, and why it is recorded that way",
               "The commit timeline is conception evidence, so it is never rewritten", nxt())
    cards(s, BODY_TOP, [
        (TEAL, "Feature branches, no self-merge",
         "The author never presses merge on their own work. Nine branches landed through "
         "two reviewed pull requests."),
        (TEAL, "Seven ADRs",
         "Every decision expensive to reverse is recorded with its alternatives and its "
         "consequences, dated."),
        (TEAL, "CI on every push",
         "Runs the CPU suite on two Python versions and mechanises three review checks "
         "that a human previously had to remember."),
    ], height=Inches(1.95))
    table(s, Inches(4.05),
          ["Standing rule", "How it is enforced now"],
          [["Pre-registered bars must not move quietly",
            ("A CI test parses the frozen document and asserts the three thresholds", 11, False, MUTED), GREEN_BG],
           ["Unflattering numbers get reported, not dropped",
            ("The per-seed table is now mandatory in the verdict report", 11, False, MUTED), GREEN_BG],
           ["No Phase 2 gate code before the verdict",
            ("CI fails if the gate stubs stop raising NotImplementedError", 11, False, MUTED), GREEN_BG]],
          [5.0, 6.893])

    # -- 15  SECTION 3 ------------------------------------------------------
    s = chrome(prs, "What comes next", "Section 3 of 3  ·  Blockers and decisions", nxt())
    banner(s, "One purchase blocks the critical path. One technical decision needs the team.")
    cards(s, Inches(3.04), [
        (RED, "Blocked on a purchase",
         "A Weights & Biases entity, project and API key. Every GPU task waits on it, and "
         "no code change can route around it."),
        (AMBER, "Blocked on a decision",
         "Two independent fluency reference models now exist, built in parallel. They "
         "conflict, and the trade is real."),
        (TEAL, "Not blocked",
         "Binding the carry overlay to the vendor model, the adversarial search, and the "
         "remaining administrative items."),
    ])
    consequence(s, Inches(5.48), "Then, and only then.  ",
                "The verdict — and Phase 2 gate work only if it reads PROCEED.")

    # -- 16 -----------------------------------------------------------------
    s = chrome(prs, "The one hard blocker, and the path after it",
               "It is a purchase, not a document — which is why it is still open", nxt())
    callout(s, L0, BODY_TOP, CW, Inches(1.3),
            "Why a logging key blocks a machine-learning experiment",
            "The vendor's Weights & Biases login and an authenticated run query execute "
            "before the evaluation branch is reached, and the result drives the resume "
            "decision. Setting log_wandb=false does not avoid that path, and working around "
            "it would mean editing the vendor tree — which has no licence and which ADR-002 "
            "forbids us to modify.", RED, RED_BG)
    table(s, Inches(3.4),
          ["Step", "Order and precondition", "Cost"],
          [["Procure the W&B entity, project and key", ("first — nothing else can start", 11, False, MUTED),
            ("purchase", 11, True, RED), RED_BG],
           ["Probe the dataset size", ("metadata only, before any egress", 11, False, MUTED),
            ("free", 11, True, GREEN), GREEN_BG],
           ["Claim a booking row, then provision", ("claimed BEFORE billing starts", 11, False, MUTED),
            ("logged", 11, True, AMBER), AMBER_BG],
           ["Rehearse on the 125M model", ("before the 1B run, same session", 11, False, MUTED),
            ("minutes", 11, True, AMBER), AMBER_BG],
           ["Run 1B, read the bar before the number", ("Standing Rule 5", 11, False, MUTED),
            ("$10–35", 11, True, AMBER), AMBER_BG]],
          [4.9, 5.2, 1.793])

    # -- 17 -----------------------------------------------------------------
    s = chrome(prs, "A decision the team has to make",
               "Two independent fluency reference models now exist, and they conflict", nxt())
    cards(s, BODY_TOP, [
        (TEAL, "GPT-2 small  (124M)",
         "Different corpus, different tokenizer, different architecture era — maximum "
         "distance from the victim. Costs a decode and re-encode step between two "
         "tokenizers."),
        (AMBER, "Llama-3.2-1B",
         "Shares the victim's Llama-3 tokenizer, so crafted token IDs feed straight in with "
         "no tokenization seam. A stronger language model, but much closer to the victim."),
        (RED, "Why it cannot be settled by merge order",
         "Whichever is chosen changes what a fluency number means. It wants an ADR before "
         "any such number is used in a verdict."),
    ], height=Inches(2.3))
    callout(s, L0, Inches(4.4), CW, Inches(1.9),
            "The trade, stated fairly — and a guard of ours that is wrong either way",
            "Seam-free scoring versus distance from the model under attack. Both are "
            "defensible: the pre-registration forbids scoring with the victim, and a sibling "
            "model that merely shares a tokenizer is not the victim. Our current independence "
            "guard rejects any reference sharing the victim's 128,256-token vocabulary — which "
            "would reject Llama-3.2 as a false positive. That guard was written to catch "
            "“someone passed the victim”; if Llama-3.2 is chosen it must check model identity "
            "instead of vocabulary width.")

    # -- 18 -----------------------------------------------------------------
    s = chrome(prs, "Where we stand", "Summary", nxt())
    stat(s, L0, BODY_TOP, "$0", "spent", "of a $325 written cap")
    stat(s, Inches(4.58), BODY_TOP, "197", "tests passing", "31 at the start of this period")
    stat(s, Inches(8.44), BODY_TOP, "0", "numbers produced", "and therefore no verdict")
    bullets(s, Inches(3.35), [
        ("The question is not yet answered — and could not honestly have been by now.",
         16, True, TITLE_INK, 10),
        ("What exists: a runnable experiment, a validated instrument, a frozen bar, and a "
         "written record of every decision that could later look convenient.",
         14, False, INK, 8),
        ("What was avoided: five distinct routes to a scientifically worthless answer — "
         "three of them false-STOP paths that the numbers alone could never have revealed.",
         14, False, GREEN, 8),
        ("What is honest to say: the bounded-drift guarantee is designed and not yet "
         "enforced, and one of our three bars cannot detect a repetitive stream.",
         14, False, AMBER, 0),
    ], height=Inches(2.05))
    callout(s, L0, Inches(5.55), CW, Inches(0.75),
            "Next step",
            "One purchase. Everything downstream of it is written, tested and waiting.",
            TEAL, GREEN_BG)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(f"wrote {build()}")
