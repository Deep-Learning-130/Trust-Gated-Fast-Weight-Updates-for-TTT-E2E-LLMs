"""Build the 2026-09-22 project-update deck: short, plain, Times New Roman throughout.

    .venv/Scripts/python.exe scripts/make_update_deck.py
    -> docs/review/TrustGate_Update_2026-09-22_v2.pptx

Every number on a slide is read from a committed result or a pre-registered document:
  pilot effect size 0.122, noise floor 0.225, damage 0.04%  experiments/002-pilot-tiny-ttt/results/deep.json
  false-alarm rate 15% (3 of 20)                            experiments/002-pilot-tiny-ttt/results/null.json
  starting error within 0.03 of prediction, fall of 2.0     commit 548c0c0
  the three bars                                            experiments/001-attack-spike/PREREGISTERED.md
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "review" / "TrustGate_Update_2026-09-22_v2.pptx"
PILOT = ROOT / "experiments" / "002-pilot-tiny-ttt" / "results"

FONT = "Times New Roman"
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x59, 0x59, 0x59)
LINE = RGBColor(0xBF, 0xBF, 0xBF)

W, H = 13.333, 7.5
LEFT = 0.9
BODY_W = W - 2 * LEFT


def _run(p, text, *, size, color=INK, bold=False):
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold = FONT, Pt(size), bold
    r.font.color.rgb = color


def _bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.3)))
    pPr.set("indent", str(-Inches(0.3)))
    bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
    pPr.append(bu)


def slide(prs, heading, lines=(), *, size=22, top=1.7, bullets=True, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(LEFT), Inches(0.6), Inches(BODY_W), Inches(0.9))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = 0
    _run(tf.paragraphs[0], heading, size=36, bold=True)
    if lines:
        tb = s.shapes.add_textbox(Inches(LEFT), Inches(top), Inches(BODY_W), Inches(H - top - 0.6))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = 0
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(14)
            if bullets:
                _bullet(p)
            _run(p, line, size=size)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def table(s, top, rows, col_w, *, size=18, row_h=0.6):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(LEFT), Inches(top), Inches(BODY_W),
                               Inches(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.background()
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.1)
            _run(cell.text_frame.paragraphs[0], val, size=size, bold=(i == 0))
            cell.text_frame.word_wrap = True
            # thin grey rule under every row, nothing else
            tcPr = cell._tc.get_or_add_tcPr()
            for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = tcPr.makeelement(qn(edge), {"w": "9525" if edge == "a:lnB" else "0"})
                if edge == "a:lnB":
                    fill = ln.makeelement(qn("a:solidFill"), {})
                    clr = fill.makeelement(qn("a:srgbClr"), {"val": "BFBFBF"})
                    fill.append(clr)
                    ln.append(fill)
                else:
                    ln.append(ln.makeelement(qn("a:noFill"), {}))
                tcPr.append(ln)


def build() -> Presentation:
    deep = json.loads((PILOT / "deep.json").read_text(encoding="utf-8"))
    null = json.loads((PILOT / "null.json").read_text(encoding="utf-8"))
    d_attack = round(deep["effect_size"], 2)
    d_floor = round(deep["noise_floor_effect_size"], 2)
    damage_pct = deep["relative_degradation"] * 100
    fpr = null["false_positive_rate_at_bar"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    # 1 -- title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    tb = s.shapes.add_textbox(Inches(LEFT), Inches(2.5), Inches(BODY_W), Inches(2.5))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    _run(tf.paragraphs[0], "Trust-Gated Updates for Models That Learn While They Read", size=40, bold=True)
    for t in ("Project update, 22 September 2026", "Manas Maahir"):
        p = tf.add_paragraph()
        p.space_before = Pt(18)
        _run(p, t, size=22, color=MUTED)

    # 2 -- the idea
    slide(prs, "The idea", [
        "Most language models are fixed once they are built.",
        "A newer design, TTT-E2E, keeps adjusting a small part of itself while it reads. This is what "
        "lets it handle very long documents cheaply.",
        "That part is called the fast weights: a kind of short-term memory the model rewrites as it goes.",
        "The question: if a model learns from whatever it reads, can someone steer what it learns just by "
        "choosing what it reads?",
    ], notes="Ordinary models do not change after they are built. This one does, a little, every time it "
             "reads. That is a feature, and also a new kind of exposure.")

    # 3 -- the risk
    slide(prs, "The risk", [
        "The attacker never touches the model. They only choose which ordinary, harmless sentences it "
        "reads, and in what order.",
        "If that is enough to make the model worse, the damage would be quiet and gradual.",
        "Today's safety filters look for harmful text. Here there is none to find.",
        "This project first tests whether the attack works. A defence is only worth building if it does.",
    ])

    # 4 -- the rules
    s = slide(prs, "How we decide", [
        "The pass rules were written down on 28 July 2026, before any attack code existed, and cannot be "
        "changed after the results are in.",
    ], bullets=False)
    table(s, 3.0, [
        ["Test", "In plain words", "Must reach"],
        ["Size of the effect", "Is the damage clearly larger than normal run-to-run variation?", "0.8 or more"],
        ["Amount of damage", "How much worse does the model get on ordinary text?", "10% or more"],
        ["Naturalness", "Does the attack text still read like normal text?", "At most 1.5× less natural"],
    ], [2.7, 6.6, 2.23], row_h=0.7)
    tb = s.shapes.add_textbox(Inches(LEFT), Inches(6.1), Inches(BODY_W), Inches(0.6))
    tb.text_frame.margin_left = 0
    _run(tb.text_frame.paragraphs[0], "All three must pass. If any one fails, the answer is “no attack”.",
         size=20, bold=True)
    s.notes_slide.notes_text_frame.text = (
        "Fixing the rules first removes the temptation to move the line after seeing results. Naturalness is "
        "judged by a separate model, GPT-2, never by the model under attack.")

    # 5 -- the defence
    slide(prs, "The proposed defence: a trust gate", [
        "Before the model accepts a change to its fast weights, a gate checks it.",
        "It compares the model's answers with those of a frozen, trusted copy, on a few test passages the "
        "attacker cannot see.",
        "If the change pulls the answers too far away, it is refused.",
        "The gate is built and tested. It will only be measured on the real model if the attack works.",
    ])

    # 6 -- progress
    slide(prs, "What is done", [
        f"The full test pipeline is built and checked by {521} automatic tests.",
        f"A practice run on a tiny stand-in model found no attack: the model got {damage_pct:.2f}% worse, "
        "against the 10% required.",
        f"The measuring tool was checked against itself: with no attacker at all, it still crosses the "
        f"effect line about {fpr:.0%} of the time. A narrow pass will be read with caution.",
        "The real model's code runs on our pipeline, and learns while it reads, as designed.",
        "Paid GPU time used so far: none.",
    ], size=21, notes=(
        f"Practice run: effect size {deep['effect_size']:.3f} against a no-attacker noise level of "
        f"{deep['noise_floor_effect_size']:.3f}. The false-alarm check ran the measurement 20 times with no "
        "attacker; 3 crossed the line."))

    # 7 -- one chart
    s = slide(prs, "Practice run result", [
        "The attack's effect was smaller than the normal variation between repeat runs.",
    ], bullets=False)
    cd = CategoryChartData()
    cd.categories = ["Required to pass", "Normal variation, no attacker", "The attack"]
    cd.add_series("Effect size", (0.80, d_floor, d_attack))
    ch = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(LEFT), Inches(2.6), Inches(BODY_W),
                            Inches(3.8), cd).chart
    ch.font.name, ch.font.size = FONT, Pt(18)
    ch.has_legend = False
    plot = ch.plots[0]
    plot.gap_width = 70
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = "0.00", False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.name, dl.font.size, dl.font.bold = FONT, Pt(18), True
    for i, col in enumerate(("BFBFBF", "8C8C8C", "1A1A1A")):
        pt = plot.series[0].points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = RGBColor.from_string(col)
    va = ch.value_axis
    va.maximum_scale, va.minimum_scale = 1.0, 0.0
    va.has_major_gridlines = False
    va.visible = False
    ca = ch.category_axis
    ca.tick_labels.font.name, ca.tick_labels.font.size = FONT, Pt(18)
    ca.format.line.color.rgb = LINE
    s.notes_slide.notes_text_frame.text = (
        "This is a tiny stand-in, about 100,000 settings. It says nothing yet about the real model; that is "
        "the next step.")

    # 8 -- next step
    slide(prs, "Next step: the real test", [
        "Run the decisive test on the 125-million-parameter version of TTT-E2E, the official model released "
        "by its authors.",
        "It runs on free hardware: our own laptop GPU, or a free Kaggle GPU as a backup.",
        "Order: a quick memory check, a baseline check that the model behaves normally, then the attack "
        "test (five repeats).",
        "If all three rules pass, the attack is real, and we measure how well the trust gate protects.",
        "If any rule fails, the project stops and we publish the negative result.",
    ], size=21, notes=(
        "The outcome rules were written down before the test, on 22 September. A pass is a result for this "
        "model; we will not claim it for larger models. A fail ends the project."))

    # 9 -- demo
    s = slide(prs, "Live demonstration")
    demos = [
        ("The practice experiment, start to finish (about 35 seconds):", "bash scripts/demo.sh"),
        ("The pass rules are locked (ten automatic checks):",
         'uv run pytest tests/test_thresholds.py -v -o addopts=""'),
        ("When the rules were written:",
         'git log --format="%ad  %s" --date=short -- experiments/001-attack-spike/PREREGISTERED.md'),
    ]
    tf = s.shapes.add_textbox(Inches(LEFT), Inches(1.7), Inches(BODY_W), Inches(5)).text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    for i, (label, cmd) in enumerate(demos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(4)
        _run(p, label, size=20)
        p = tf.add_paragraph()
        p.space_after = Pt(24)
        _run(p, "    " + cmd, size=18, color=MUTED)
    s.notes_slide.notes_text_frame.text = (
        "Run from the project folder in Git Bash. Demo 1: at the end, compare the two effect sizes; the "
        "second compares normal text with normal text, so it is pure noise. Demo 2 prints ten PASSED lines. "
        "Demo 3 prints one line per dated change to the rules.")

    # 10 -- close
    slide(prs, "Questions", [
        "The tools are built and tested. The decisive test comes next, judged by rules fixed in advance.",
    ], bullets=False, size=22, top=1.8)
    return prs


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build().save(OUT)
    print(OUT)
