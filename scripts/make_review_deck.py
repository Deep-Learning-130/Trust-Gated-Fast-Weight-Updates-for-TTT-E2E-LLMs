#!/usr/bin/env python
"""Generate the project review deck.

Committed rather than hand-built so the deck is reproducible and its claims are
diffable. Every number in here is taken from a measurement recorded elsewhere in
the repo -- `docs/report.md`, `FLUENCY_REFERENCE.md`, the ADRs -- rather than
retyped from memory.

    uv pip install --python .venv python-pptx
    python scripts/make_review_deck.py

Writes `docs/review/TrustGate_Review_<date>.pptx`.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

DATE = "2026-08-27"
OUT = Path(__file__).resolve().parents[1] / "docs" / "review" / f"TrustGate_Review_{DATE}.pptx"

INK = RGBColor(0x1A, 0x1D, 0x23)
MUTED = RGBColor(0x5B, 0x63, 0x70)
ACCENT = RGBColor(0x1B, 0x5E, 0x8C)
WARN = RGBColor(0xA6, 0x3D, 0x1F)
GOOD = RGBColor(0x1E, 0x6B, 0x4A)
RULE = RGBColor(0xD5, 0xDA, 0xE0)
BG = RGBColor(0xFF, 0xFF, 0xFF)

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.75)


def _blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = BG
    return slide


def _text(slide, left, top, width, height, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.paragraphs[0].alignment = align
    return frame


def _line(slide, top, color=RULE, height=Pt(1.5)):
    from pptx.enum.shapes import MSO_SHAPE

    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, top, W - 2 * MARGIN, height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    bar.shadow.inherit = False
    return bar


def _run(paragraph, text, size, bold=False, color=INK, italic=False):
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = "Segoe UI"
    return run


def title_slide(prs, title, subtitle, footer):
    slide = _blank(prs)
    frame = _text(slide, MARGIN, Inches(2.4), W - 2 * MARGIN, Inches(1.4))
    _run(frame.paragraphs[0], title, 40, bold=True, color=INK)

    _line(slide, Inches(3.85), ACCENT, Pt(3))

    frame = _text(slide, MARGIN, Inches(4.05), W - 2 * MARGIN, Inches(1.2))
    _run(frame.paragraphs[0], subtitle, 19, color=MUTED)

    frame = _text(slide, MARGIN, Inches(6.4), W - 2 * MARGIN, Inches(0.6))
    _run(frame.paragraphs[0], footer, 13, color=MUTED)
    return slide


def section_slide(prs, number, title, blurb):
    slide = _blank(prs)
    frame = _text(slide, MARGIN, Inches(2.7), W - 2 * MARGIN, Inches(0.6))
    _run(frame.paragraphs[0], number, 16, bold=True, color=ACCENT)

    frame = _text(slide, MARGIN, Inches(3.2), W - 2 * MARGIN, Inches(1.0))
    _run(frame.paragraphs[0], title, 34, bold=True, color=INK)

    _line(slide, Inches(4.35), ACCENT, Pt(3))

    frame = _text(slide, MARGIN, Inches(4.6), Inches(9.5), Inches(1.2))
    _run(frame.paragraphs[0], blurb, 17, color=MUTED)
    return slide


def content_slide(prs, title, kicker, bullets, index=None):
    """bullets: list of (text, level, style) where style in {'', 'bold', 'warn', 'good', 'muted'}."""
    slide = _blank(prs)

    frame = _text(slide, MARGIN, Inches(0.55), W - 2 * MARGIN, Inches(0.7))
    _run(frame.paragraphs[0], title, 27, bold=True, color=INK)

    if kicker:
        frame = _text(slide, MARGIN, Inches(1.18), W - 2 * MARGIN, Inches(0.45))
        _run(frame.paragraphs[0], kicker, 14, color=MUTED, italic=True)

    _line(slide, Inches(1.72))

    frame = _text(slide, MARGIN, Inches(2.0), W - 2 * MARGIN, Inches(4.9))
    colors = {"": INK, "bold": INK, "warn": WARN, "good": GOOD, "muted": MUTED}

    for i, (text, level, style) in enumerate(bullets):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.level = level
        para.space_after = Pt(9 if level == 0 else 5)
        size = 17 if level == 0 else 14
        marker = "" if level == 0 else "— "
        _run(
            para,
            marker + text,
            size,
            bold=(style == "bold"),
            color=colors[style],
        )

    if index is not None:
        frame = _text(slide, W - Inches(1.4), Inches(6.85), Inches(0.7), Inches(0.4), PP_ALIGN.RIGHT)
        _run(frame.paragraphs[0], str(index), 12, color=MUTED)
    return slide


def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = [1]

    def idx():
        n[0] += 1
        return n[0]

    # 1 -------------------------------------------------------------- title
    title_slide(
        prs,
        "Trust-Gated Fast-Weight Updates for TTT-E2E LLMs",
        "Project review  ·  motivation, work completed, and what comes next",
        f"{DATE}  ·  no measurement taken yet  ·  total spend $0 of a $325 cap",
    )

    content_slide(
        prs,
        "What this review covers",
        "Three questions, in order",
        [
            ("1.  Why the work exists — what TTT-E2E changes about the threat model", 0, "bold"),
            ("2.  What we built, and what we found while building it", 0, "bold"),
            ("3.  What is blocked, and on what", 0, "bold"),
            ("", 0, ""),
            ("The honest headline: the experiment is now runnable and its instrument is", 0, "muted"),
            ("validated, but no number has been produced and no verdict has been reached.", 0, "muted"),
        ],
        idx(),
    )

    # 2 --------------------------------------------------------- motivation
    section_slide(
        prs,
        "SECTION 1 OF 3",
        "Motivation",
        "A model that learns while it serves can be poisoned while it serves.",
    )

    content_slide(
        prs,
        "The architecture we target",
        "TTT-E2E — End-to-End Test-Time Training for Long Context",
        [
            ("TTT-E2E compresses long context by updating 'fast weights' at inference time.", 0, ""),
            ("An inner SGD loop runs inside the forward pass, once per mini-batch chunk.", 1, "muted"),
            ("The adapted weights are a function of whatever text the model just read.", 1, "muted"),
            ("", 0, ""),
            ("That is the whole benefit — and the whole exposure.", 0, "bold"),
            ("Conventional LLM inference is stateless: a bad prompt harms one response.", 1, "muted"),
            ("Here, the context stream mutates model weights. Harm can persist.", 1, "warn"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "The attack surface this creates",
        "Dossier §1 — the threat model",
        [
            ("The adversary never touches the base weights, the training set, or the host.", 0, ""),
            ("They only supply context — the thing the system is designed to accept.", 1, "muted"),
            ("", 0, ""),
            ("Two distinct harms, kept separately switchable:", 0, "bold"),
            ("DEGRADE — steer fast weights so later benign inputs are handled worse.", 1, ""),
            ("TRIGGER — implant a latent association while benign quality stays flat.", 1, ""),
            ("", 0, ""),
            ("TRIGGER is the more alarming: it is silent under any monitor that watches", 0, "warn"),
            ("only aggregate quality.", 0, "warn"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Why this problem, and not another",
        "What the literature survey actually found",
        [
            ("Test-time adaptation defenses exist — but they are vision and BatchNorm methods.", 0, ""),
            ("MedBN and its relatives defend batch-norm statistics, not fast-weight updates.", 1, "muted"),
            ("They cannot be run as-is; the principle has to be ported, not reused.", 1, "muted"),
            ("", 0, ""),
            ("Nothing in the surveyed work gates fast-weight updates in an LLM under a", 0, "bold"),
            ("stated, testable bound.", 0, "bold"),
            ("", 0, ""),
            ("15 verified journal papers with resolved DOIs — docs/literature-survey.md", 0, "muted"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Our proposal — the trust gate",
        "Intercept every proposed update; admit it only if it survives a check",
        [
            ("Interceptor — no fast-weight update commits without passing through it.", 0, "bold"),
            ("Frozen-anchor consistency check as the primary gate signal (ADR-F1).", 1, "muted"),
            ("Bounded cumulative-drift budget — the headline claimable property.", 0, "bold"),
            ("Over any window, committed weights cannot travel more than ε from the anchor.", 1, "muted"),
            ("Checkpoint and rollback — O(1) revert via a versioned ring buffer.", 0, "bold"),
            ("", 0, ""),
            ("Stated honestly: the accumulator and store are built and unit-tested but", 0, "warn"),
            ("NOT yet wired into the interceptor. The bound is designed, not enforced (ADR-003).", 0, "warn"),
        ],
        idx(),
    )

    # 3 -------------------------------------------------------- what we did
    section_slide(
        prs,
        "SECTION 2 OF 3",
        "What we did",
        "Build the experiment that can answer the question — and find the ways it could lie.",
    )

    content_slide(
        prs,
        "The discipline first: a pre-registered kill-gate",
        "Because the answer might be 'stop', and that has to be allowed",
        [
            ("Before any attack code ran, we froze the go/no-go criterion in writing.", 0, ""),
            ("Effect size (Cohen's d) ≥ 0.8   ·   relative degradation ≥ 10%   ·   fluency ratio ≤ 1.5", 1, "bold"),
            ("All three required, not any. Missing one means STOP or a dated, reasoned revision.", 1, "muted"),
            ("", 0, ""),
            ("Dossier §7: no demonstrated benign-looking attack means no defense worth building.", 0, ""),
            ("STOP is a complete outcome, not a failure. The paper reports it either way.", 1, "muted"),
            ("", 0, ""),
            ("The thresholds now live in code with a CI test that parses the frozen document", 0, "good"),
            ("and asserts equality — a bar cannot move without CI going red.", 0, "good"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Phase 0.4 — de-risking the first GPU booking for $0",
        "Seven desk deliverables, completed before any hardware existed",
        [
            ("Exact eval command, pre-registered tolerance, cost model with a $325 cap,", 0, ""),
            ("idempotent cold-start bootstrap, GPU queue and booking ledger, review protocol.", 0, ""),
            ("", 0, ""),
            ("Three findings that changed what someone would otherwise have done:", 0, "bold"),
            ("The paper reports NO number for our checkpoint — the bar became a two-sided", 1, "warn"),
            ("bracket plus structural checks, not 'match the published number' (ADR-005).", 1, "warn"),
            ("A /val-only dataset copy CRASHES the job after billing starts — it needed one", 1, "warn"),
            ("extra metadata file, found for free.", 1, "warn"),
            ("The 3B/128K run is ~10× the 1B, not ~2× — it forces a multi-GPU node.", 1, "warn"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Phase 1 — the experiment, built and tested on CPU",
        "197 tests passing · 3,909 lines of source · 2,255 lines of tests",
        [
            ("Attack objectives optimise the victim's own reported metric, transcribed", 0, ""),
            ("from the vendor's cross-entropy so the attacker and the bar measure one thing.", 1, "muted"),
            ("Corpus and stream builders work in token-ID space — spans of real text, so", 0, ""),
            ("fluency is exact by construction and length matching is exact in model units.", 1, "muted"),
            ("Matching discipline: 13 fields that must agree between poison and control,", 0, ""),
            ("validated at construction rather than at comparison time.", 1, "muted"),
            ("Harness CLI, verdict report, and CI — none of which existed before.", 0, ""),
            ("", 0, ""),
            ("NotImplementedError sites: 15 → 10. Five of the ten are Phase 2 gate code,", 0, "muted"),
            ("deliberately forbidden until the verdict lands.", 0, "muted"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "The findings that mattered most",
        "Five ways the experiment could have produced a scientifically worthless answer",
        [
            ("The vendor DISCARDS the adapted fast weights.", 0, "bold"),
            ("transformer.py:712 binds the scan carry and never reads it. A two-call harness —", 1, "warn"),
            ("adapt, then measure — resets the weights and measures noise. A guaranteed null,", 1, "warn"),
            ("indistinguishable from a real STOP. Since STOP ends the project, this would have", 1, "warn"),
            ("converted a paid GPU run into a project-ending artefact. Silently.", 1, "warn"),
            ("A second, independent route to the same false STOP: the inner-LR ramp.", 0, "bold"),
            ("Left unsaturated, the inner loop is near-frozen and nothing moves.", 1, "warn"),
            ("Two live defects in shipped code: per-seed values were never printed (a seed where", 0, "bold"),
            ("poison HELPED was invisible), and degenerate input rendered PROCEED on an infinite", 1, "warn"),
            ("effect size.", 1, "warn"),
            ("The vendor has no tokenizer at all — the task that depended on it had no referent.", 0, "bold"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "The fluency bar — and an honest negative result",
        "T3.2: the third gating criterion",
        [
            ("An independent reference model scores whether a poison stream still looks benign.", 0, ""),
            ("Scoring with the victim would be circular: the attacker optimises against it.", 1, "muted"),
            ("", 0, ""),
            ("The acceptance check the plan specified DOES NOT WORK.", 0, "bold"),
            ("The published perplexity is per word, and the denominator is unspecified: our", 1, "warn"),
            ("measurement gives 62.29 on one convention and 31.18 on another — and the", 1, "warn"),
            ("published 37.50 falls between them. The check cannot separate a bug from a", 1, "warn"),
            ("counting convention, so it is recorded as evidence and NOT used as a gate.", 1, "warn"),
            ("", 0, ""),
            ("Replaced with a decisive check: logit-level equivalence against a reference", 0, "good"),
            ("implementation — worst difference 5.8e-04, 100% argmax agreement. PASS.", 0, "good"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "A limitation of one of our own bars",
        "Measured, and recorded before any result exists",
        [
            ("Perplexity punishes scrambling but REWARDS repetition.", 0, "bold"),
            ("Fluent English scores 19.98.   A repeated pattern scores 1.90.", 1, ""),
            ("", 0, ""),
            ("So the pre-registered fluency bar cannot catch a poison stream that reuses one", 0, "warn"),
            ("passage many times — something no human would call benign-looking.", 0, "warn"),
            ("", 0, ""),
            ("Mitigations already in place, and now known to be load-bearing rather than tidy:", 0, "good"),
            ("spans are drawn without replacement, and span multiplicity is reported per arm.", 1, "good"),
            ("", 0, ""),
            ("Written down before results exist. Afterwards it would read as an excuse.", 0, "muted"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Process and repository state",
        "The commit timeline is conception evidence, so it is never rewritten",
        [
            ("Feature branches only; the author never presses merge on their own work.", 0, ""),
            ("Nine branches reviewed and merged to the org repo via two pull requests.", 1, "good"),
            ("CI runs the CPU suite on every push and mechanises three review checks that", 0, ""),
            ("were previously enforced only by a human remembering to look.", 1, "muted"),
            ("Seven ADRs record every decision expensive to reverse.", 0, ""),
            ("", 0, ""),
            ("Zero spend. No GPU provisioned, no checkpoint fetched, no dataset egressed.", 0, "bold"),
        ],
        idx(),
    )

    # 4 ------------------------------------------------------- what's next
    section_slide(
        prs,
        "SECTION 3 OF 3",
        "What's next",
        "One purchase blocks the critical path. One decision needs the team.",
    )

    content_slide(
        prs,
        "The one hard blocker",
        "It is a purchase, not a document — which is why it is still open",
        [
            ("A Weights & Biases entity, project and API key (T1.2).", 0, "bold"),
            ("Every GPU task waits on it. The vendor's login and an authenticated query run", 1, "warn"),
            ("before the eval branch, so log_wandb=false does not avoid it, and working around", 1, "warn"),
            ("it means editing vendor code — which ADR-002 forbids (the tree has no licence).", 1, "warn"),
            ("", 0, ""),
            ("Then, in this order, all of it already written down:", 0, "bold"),
            ("Probe the dataset size for free  →  claim a booking row BEFORE billing starts", 1, ""),
            ("→  bootstrap  →  rehearse on the 125M model  →  run 1B  →  read the bar first.", 1, ""),
        ],
        idx(),
    )

    content_slide(
        prs,
        "A decision the team has to make",
        "Two independent implementations of the fluency reference model now exist",
        [
            ("Both were built in parallel, both are pushed, and they conflict.", 0, "bold"),
            ("", 0, ""),
            ("GPT-2 small — different corpus, different tokenizer, different architecture era.", 0, ""),
            ("Maximum independence; needs a decode/re-encode step between tokenizers.", 1, "muted"),
            ("Llama-3.2-1B — shares the victim's tokenizer, so token IDs feed in directly.", 0, ""),
            ("No tokenization seam; a stronger language model, but far closer to the victim.", 1, "muted"),
            ("", 0, ""),
            ("The trade is real: seam-free scoring versus distance from the model under attack.", 0, "warn"),
            ("Whichever is chosen should be recorded as an ADR before any fluency number is used.", 0, "warn"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Remaining work, in value order",
        "Everything below is desk work until the key exists",
        [
            ("1.  Bind the carry overlay to the real vendor model, plus benign evaluation.", 0, "bold"),
            ("The design is settled and tested; this is the adapter. It is the last real unknown.", 1, "muted"),
            ("2.  The adversarial search that crafts a poison stream (currently a proposal generator).", 0, ""),
            ("3.  PARAPHRASE and SOFT stream strategies — reported separately from SELECT.", 0, ""),
            ("4.  Administrative: inventorship, the signed acknowledgement table, the paper tree.", 0, ""),
            ("", 0, ""),
            ("Then the verdict — and Phase 2 only if it says PROCEED.", 0, "bold"),
            ("Gate code stays unwritten until then; building it early is sunk cost against STOP.", 1, "muted"),
        ],
        idx(),
    )

    content_slide(
        prs,
        "Where we stand",
        "Summary",
        [
            ("The question is not yet answered — and could not honestly have been by now.", 0, "bold"),
            ("", 0, ""),
            ("What exists: a runnable experiment, a validated instrument, a frozen bar, and a", 0, ""),
            ("written record of every decision that could later look convenient.", 0, ""),
            ("", 0, ""),
            ("What was avoided: five distinct routes to a scientifically worthless answer,", 0, "good"),
            ("three of them false-STOP paths the numbers alone could never have revealed.", 0, "good"),
            ("", 0, ""),
            ("Cost so far: $0 of a $325 cap.", 0, "bold"),
            ("Next step: one purchase.", 0, "bold"),
        ],
        idx(),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
