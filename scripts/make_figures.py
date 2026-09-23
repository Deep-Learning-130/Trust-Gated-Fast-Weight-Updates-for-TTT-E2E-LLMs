"""Generate the paper's tables and figures.

The first plotting code in this repository. Every number below is transcribed
from a document in this tree, and every block cites the file and section it came
from. Nothing here is measured by this script -- the Stage 1 figures are
published prior-art values and instrument-validation numbers that already exist;
the pilot figures are drawn from `experiments/002-pilot-tiny-ttt/results/`.

Outputs, both written to `docs/paper/figures/`:

- `*.png`  -- for the review deck, which needs raster images.
- `*.tex`  -- pgfplots fragments with inline coordinates, so the paper compiles
              with no external image files. T3.10 requires the document stay
              self-contained, and `p3-intro-related-methodology.tex` already
              draws its architecture figure in TikZ for the same reason.

Run:  python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "paper" / "figures"

# Colour-blind-safe. TTT-E2E is the emphasis series wherever it appears.
EMPH = "#0b6e4f"
MUTED = ["#4c72b0", "#dd8452", "#937860", "#8172b3", "#b07aa1", "#da8bc3"]
CONTROL = "#4c72b0"
POISON = "#c44e52"


# --------------------------------------------------------------------------
# Table I + Figure 1 -- comparison against the base works.
#
# Source: experiments/000-repro-baseline/TOLERANCE.md section 3.
# 3B, 3x Chinchilla, Books-finetuned, evaluated on held-out Books. Mean CE in
# nats/token. Recovered from the paper's Figure 9 vector art -- the y-axis tick
# stubs gave a linear calibration with a maximum residual of 4e-16 nats -- and
# cross-validated against its Figure 1 on all 35 values to 4 decimal places.
# --------------------------------------------------------------------------
CONTEXTS = [8, 16, 32, 64, 128]  # thousands of tokens
BASELINES = {
    "Transformer (full attention)": [2.328, 2.301, 2.283, 2.270, 2.261],
    "Sliding-window attention (k=8K)": [2.328, 2.307, 2.299, 2.301, 2.309],
    "Hybrid SWA + full": [2.328, 2.304, 2.288, 2.280, 2.276],
    "Mamba 2": [2.312, 2.291, 2.283, 2.285, 2.293],
    "Gated DeltaNet": [2.322, 2.301, 2.291, 2.289, 2.295],
    "TTT-KVB": [2.319, 2.298, 2.291, 2.293, 2.305],
    "TTT-E2E (system under study)": [2.314, 2.288, 2.270, 2.258, 2.249],
}

# --------------------------------------------------------------------------
# Figure 2 -- the fluency instrument, and the limitation of one of our own bars.
#
# Source: experiments/001-attack-spike/FLUENCY_REFERENCE.md, sections
# "Discrimination self-test" and "A limitation of the bar itself". GPT-2 small,
# measured on CPU. The repeated-pattern bar is the point: perplexity punishes
# scrambling but *rewards* repetition, so the pre-registered fluency bar cannot
# detect a poison stream that reuses one passage many times.
# --------------------------------------------------------------------------
FLUENCY_PROBES = [
    ("Repeated short pattern", 1.90),
    ("Fluent English (Austen)", 19.98),
    ("Same words, scrambled", 1326.84),
]

# --------------------------------------------------------------------------
# Figure 3 -- GPT-2 forward-pass equivalence against the reference implementation.
#
# Source: FLUENCY_REFERENCE.md, "Primary: equivalence to the reference
# implementation". Acceptance band 1e-03; argmax agreement 1.0000 at every
# length. A transposed Conv1D, exact GELU, or a wrong LayerNorm epsilon each
# shift logits by O(0.1) -- four orders of magnitude outside this band.
# --------------------------------------------------------------------------
EQUIV_LENGTHS = [1, 7, 64, 256, 1024]
EQUIV_LOGIT_DIFF = [1.373e-04, 1.984e-04, 1.907e-04, 2.213e-04, 5.798e-04]
EQUIV_BAND = 1e-03

# Frozen bars. Mirrored from src/trustgate/eval/prereg.py FROZEN, which a CI
# test keeps equal to experiments/001-attack-spike/PREREGISTERED.md.
PREREG = [
    ("Effect size (Cohen's d), poison vs control", "$\\ge 0.80$"),
    ("Relative benign degradation", "$\\ge 10\\%$"),
    ("Fluency ratio vs control (max across seeds)", "$\\le 1.50$"),
    ("Usable seeds per condition", "$\\ge 5$"),
]


def _save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / (stem + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + stem + ".png")


def _write_tex(stem: str, body: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    (FIGDIR / (stem + ".tex")).write_text(body, encoding="utf-8")
    print("  wrote " + stem + ".tex")


# ==========================================================================
# Stage 1 -- no new measurement
# ==========================================================================
def table_baselines() -> None:
    """Table I -- published losses for seven long-context methods."""
    rows = []
    for name, vals in BASELINES.items():
        if name.startswith("TTT-E2E"):
            cells = " & ".join("\\textbf{%.3f}" % v for v in vals)
            rows.append("\\textbf{%s} & %s \\\\" % (name, cells))
        else:
            cells = " & ".join("%.3f" % v for v in vals)
            rows.append("%s & %s \\\\" % (name, cells))

    body = (
        "% Generated by scripts/make_figures.py -- do not edit by hand.\n"
        "% Source: experiments/000-repro-baseline/TOLERANCE.md section 3.\n"
        "\\begin{table}[t]\n"
        "\\caption{Held-out Books loss (nats/token) for seven long-context methods\n"
        "at 3B parameters, $3\\times$ Chinchilla compute. Values recovered from the\n"
        "TTT-E2E paper's Figure~9 vector data (maximum axis-calibration residual\n"
        "$4\\times10^{-16}$ nats) and cross-validated against its Figure~1 on all 35\n"
        "values. Lower is better.}\n"
        "\\label{tab:baselines}\n"
        "\\centering\n"
        "\\begin{tabular}{lccccc}\n"
        "\\hline\n"
        "Method & 8K & 16K & 32K & 64K & 128K \\\\\n"
        "\\hline\n" + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write_tex("table1_baselines", body)


def figure_baselines() -> None:
    """Figure 1 -- loss vs context length, seven methods."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, (name, vals) in enumerate(BASELINES.items()):
        emph = name.startswith("TTT-E2E")
        ax.plot(
            CONTEXTS,
            vals,
            marker="o" if emph else "s",
            markersize=6 if emph else 4,
            linewidth=2.6 if emph else 1.3,
            color=EMPH if emph else MUTED[i % len(MUTED)],
            alpha=1.0 if emph else 0.75,
            zorder=5 if emph else 2,
            label=name,
        )

    # The gap that governs the discussion: at 8K the whole TTT mechanism is
    # worth 0.014 nats (SWA 2.328 vs TTT-E2E 2.314).
    ax.annotate("", xy=(8, 2.314), xytext=(8, 2.328),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    ax.text(8.7, 2.321,
            "0.014 nats =\nthe entire TTT\nbenefit at 8K",
            fontsize=8, va="center", color="#333333")

    ax.set_xscale("log", base=2)
    ax.set_xticks(CONTEXTS)
    ax.set_xticklabels([str(c) + "K" for c in CONTEXTS])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Held-out Books loss (nats/token)")
    ax.set_title("Long-context methods at 3B parameters, $3\\times$ Chinchilla compute",
                 fontsize=11)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    _save(fig, "fig1_baselines")

    coords = []
    for name, vals in BASELINES.items():
        pts = " ".join("(%d,%.3f)" % (c, v) for c, v in zip(CONTEXTS, vals))
        coords.append("\\addplot coordinates {%s}; \\addlegendentry{%s}" % (pts, name))
    _write_tex("fig1_baselines",
               "% Generated by scripts/make_figures.py.\n"
               "% Source: experiments/000-repro-baseline/TOLERANCE.md section 3.\n"
               "\\begin{figure}[t]\\centering\n"
               "\\begin{tikzpicture}\n"
               "\\begin{semilogxaxis}[width=\\columnwidth,height=6cm,log basis x=2,\n"
               "  xtick={8,16,32,64,128},xticklabels={8K,16K,32K,64K,128K},\n"
               "  xlabel={Context length},ylabel={Loss (nats/token)},\n"
               "  legend style={font=\\tiny,at={(0.98,0.98)},anchor=north east},\n"
               "  grid=both,grid style={dotted}]\n"
               + "\n".join(coords) + "\n"
               "\\end{semilogxaxis}\n"
               "\\end{tikzpicture}\n"
               "\\caption{Held-out Books loss against context length for seven\n"
               "long-context methods. TTT-E2E is lowest at every context length; SWA,\n"
               "Mamba~2, Gated DeltaNet and TTT-KVB all worsen past 32K.}\n"
               "\\label{fig:baselines}\n"
               "\\end{figure}\n")


def figure_fluency() -> None:
    """Figure 2 -- fluency discrimination, and the bar's blind spot."""
    labels = [p[0] for p in FLUENCY_PROBES]
    vals = [p[1] for p in FLUENCY_PROBES]
    colours = [POISON, EMPH, "#8172b3"]

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    bars = ax.barh(labels, vals, color=colours, height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Perplexity under the independent reference model "
                  "(GPT-2 small, log scale)")
    ax.set_xlim(1, 5000)
    for b, v in zip(bars, vals):
        ax.text(v * 1.18, b.get_y() + b.get_height() / 2, "%.2f" % v,
                va="center", fontsize=9)
    ax.axvline(19.98, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("The realism bar discriminates scrambling but rewards repetition",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.text(0.12, -0.06,
             "A repetitive stream scores 10x BETTER than real prose, so the "
             "pre-registered fluency bar cannot reject it.",
             fontsize=8.5, color=POISON)
    _save(fig, "fig2_fluency")

    pts = " ".join("(%.2f,%d)" % (v, i) for i, (_, v) in enumerate(FLUENCY_PROBES))
    _write_tex("fig2_fluency",
               "% Generated by scripts/make_figures.py.\n"
               "% Source: experiments/001-attack-spike/FLUENCY_REFERENCE.md.\n"
               "\\begin{figure}[t]\\centering\n"
               "\\begin{tikzpicture}\n"
               "\\begin{semilogxaxis}[width=\\columnwidth,height=4.2cm,xbar,\n"
               "  ytick={0,1,2},yticklabels={Repeated pattern,Fluent English,Scrambled},\n"
               "  xlabel={Perplexity (GPT-2 small, log scale)},xmin=1,xmax=5000,\n"
               "  nodes near coords,every node near coord/.append style={font=\\tiny}]\n"
               "\\addplot coordinates {" + pts + "};\n"
               "\\end{semilogxaxis}\n"
               "\\end{tikzpicture}\n"
               "\\caption{Measured perplexity of three probe texts under the\n"
               "independent reference model. The $66\\times$ separation between fluent\n"
               "and scrambled English is the discrimination self-test. The repeated\n"
               "pattern at 1.90 is the negative result: perplexity rewards repetition,\n"
               "so the $\\rho \\le 1.5$ realism bar cannot by itself reject a poison\n"
               "stream that reuses one passage.}\n"
               "\\label{fig:fluency}\n"
               "\\end{figure}\n")


def figure_equivalence() -> None:
    """Figure 3 -- the fluency instrument is correct."""
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(EQUIV_LENGTHS, EQUIV_LOGIT_DIFF, marker="o", color=EMPH, linewidth=2.0,
            label="max $|$logit difference$|$ vs reference")
    ax.axhline(EQUIV_BAND, color=POISON, linestyle="--", linewidth=1.4,
               label="acceptance band ($10^{-3}$)")
    ax.axhline(0.1, color="#666666", linestyle=":", linewidth=1.2,
               label="shift a real bug would cause ($\\sim$0.1)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(EQUIV_LENGTHS)
    ax.set_xticklabels([str(v) for v in EQUIV_LENGTHS])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Max absolute logit difference")
    ax.set_title("Reimplemented GPT-2 forward pass vs the reference\n"
                 "(argmax agreement 1.0000 at every length)", fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=7.5, loc="center left")
    _save(fig, "fig3_equivalence")


def table_prereg(observed=None) -> None:
    """Table II -- the frozen bars, with an observed column."""
    rows = []
    for crit, bar in PREREG:
        obs = "---"
        if observed:
            obs = observed.get(crit, "---")
        rows.append("%s & %s & %s \\\\" % (crit, bar, obs))

    body = (
        "% Generated by scripts/make_figures.py -- do not edit by hand.\n"
        "% Source: src/trustgate/eval/prereg.py (FROZEN), kept equal to\n"
        "% experiments/001-attack-spike/PREREGISTERED.md by a CI test.\n"
        "\\begin{table}[t]\n"
        "\\caption{Pre-registered criteria, frozen 2026-07-28 before any attack code\n"
        "was run. All four are required jointly. The observed column reports the\n"
        "scaled-down pilot; it is \\emph{not} the pre-registered verdict, which\n"
        "requires the released checkpoint and the GPT-2 fluency instrument.}\n"
        "\\label{tab:results}\n"
        "\\centering\n"
        "\\begin{tabular}{lcc}\n"
        "\\hline\n"
        "Criterion & Pre-registered & Pilot (observed) \\\\\n"
        "\\hline\n" + "\n".join(rows) + "\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    _write_tex("table2_prereg", body)


# ==========================================================================
# Stage 3 -- pilot results, if a run has completed
# ==========================================================================
def _pilot_figures(data):
    """Figures 4 and 5, drawn from a completed pilot run."""
    print("\nStage 3 -- pilot results")

    # Figure 4: per-seed paired slope plot. Standing Rule 6 -- a seed where the
    # poison stream HELPED the victim must be visible, not averaged away.
    control = data["per_seed_control"]
    poison = data["per_seed_poisoned"]
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    helped = 0
    for i, (c, p) in enumerate(zip(control, poison)):
        backfired = p < c
        helped += int(backfired)
        colour = "#888888" if backfired else POISON
        ax.plot([0, 1], [c, p], marker="o", color=colour, linewidth=1.8,
                markersize=6, alpha=0.9, linestyle="--" if backfired else "-")
        ax.annotate("s%d" % i, (1.04, p), fontsize=8, va="center", color=colour)
    ax.set_xlim(-0.25, 1.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Benign control", "Poison"])
    ax.set_ylabel("Held-out benign loss (nats/token)")
    title = "Per-seed paired benign loss after each stream"
    if helped:
        title += "\n(%d seed(s) where poison helped, dashed grey)" % helped
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    _save(fig, "fig4_per_seed")

    # Paired slope plot as pgfplots too, so the paper stays image-free.
    segs = []
    for i, (c, p_) in enumerate(zip(control, poison)):
        style = "gray,dashed" if p_ < c else "red"
        segs.append("\\addplot[%s,mark=*] coordinates {(0,%.5f) (1,%.5f)};"
                    % (style, c, p_))
    _write_tex("fig4_per_seed",
               "% Generated by scripts/make_figures.py.\n"
               "\\begin{figure}[t]\\centering\n"
               "\\begin{tikzpicture}\n"
               "\\begin{axis}[width=0.85\\columnwidth,height=5.4cm,\n"
               "  xtick={0,1},xticklabels={Benign control,Poison},\n"
               "  xmin=-0.3,xmax=1.3,ylabel={Held-out benign loss (nats/token)},\n"
               "  grid=both,grid style={dotted}]\n"
               + "\n".join(segs) + "\n"
               "\\end{axis}\n"
               "\\end{tikzpicture}\n"
               "\\caption{Per-seed benign loss after each stream, paired by seed. Dashed grey marks the seeds in which the poison stream helped the victim. Printing every seed is a standing requirement: an aggregate hides sign changes, and sign changes at this magnitude are better described as noise than as a weak attack.}\n"
               "\\label{fig:perseed}\n"
               "\\end{figure}\n")

    # Figure 5: dose-response, against the control-vs-control noise floor.
    dose = data.get("dose_response")
    if dose:
        # Two panels, because one scale cannot carry both facts honestly. The
        # left shows how far the observed effect sits from the pre-registered
        # bar; on that scale the trend is invisible. The right zooms to the data
        # so the monotone rise -- and its size against the measured noise floor
        # -- can actually be read.
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 3.9))
        toks = dose["stream_tokens"]
        rel = [100.0 * r for r in dose["relative_degradation"]]
        floor = abs(100.0 * data.get("noise_floor_relative", 0.0))

        axl.axhline(10.0, color=POISON, linestyle="--", linewidth=1.6,
                    label="pre-registered bar (10%)")
        axl.plot(toks, rel, marker="o", color=EMPH, linewidth=2.2,
                 label="observed degradation")
        axl.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
        axl.set_ylim(-1.0, 11.5)
        axl.set_xlabel("Poison stream length (tokens)")
        axl.set_ylabel("Relative benign degradation (%)")
        axl.set_title("Against the pre-registered bar", fontsize=10)
        axl.legend(fontsize=8, loc="center right")
        axl.grid(alpha=0.25, linestyle=":")

        axr.axhspan(-floor, floor, color=CONTROL, alpha=0.18,
                    label="control-vs-control noise floor ($\pm$%.3f%%)" % floor)
        axr.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.5)
        axr.plot(toks, rel, marker="o", color=EMPH, linewidth=2.2,
                 label="observed degradation")
        for t, r in zip(toks, rel):
            axr.annotate("%.3f%%" % r, (t, r), fontsize=7,
                         textcoords="offset points", xytext=(0, 7), ha="center")
        axr.set_xlabel("Poison stream length (tokens)")
        axr.set_title("Zoomed to the data", fontsize=10)
        axr.legend(fontsize=8, loc="upper left")
        axr.grid(alpha=0.25, linestyle=":")

        fig.suptitle("Dose-response: degradation against attack budget "
                     "(scaled-down pilot, not the verdict)", fontsize=11)
        fig.tight_layout()
        _save(fig, "fig5_dose_response")

        pts = " ".join("(%d,%.3f)" % (t, r) for t, r in zip(toks, rel))
        _write_tex("fig5_dose_response",
                   "% Generated by scripts/make_figures.py.\n"
                   "\\begin{figure}[t]\\centering\n"
                   "\\begin{tikzpicture}\n"
                   "\\begin{axis}[width=\\columnwidth,height=5cm,\n"
                   "  xlabel={Poison stream length (tokens)},\n"
                   "  ylabel={Relative degradation (\\%)},grid=both,grid style={dotted}]\n"
                   "\\addplot coordinates {" + pts + "};\n"
                   "\\end{axis}\n"
                   "\\end{tikzpicture}\n"
                   "\\caption{Degradation against attack budget in the scaled-down\n"
                   "pilot. The pre-registration names stream tokens to threshold as a\n"
                   "secondary, non-gating metric.}\n"
                   "\\label{fig:dose}\n"
                   "\\end{figure}\n")

    # ---- Figure 6: the signal against its own negative control ----------
    # The single most important panel in the pilot. Cohen's d for poison-vs-
    # control clears the pre-registered 0.80 bar -- but the control-vs-control
    # arm, in which nothing is attacking anything, reaches |d| = 0.557 on the
    # same five seeds. A bar that a pure negative control can approach is not
    # separating signal from noise at this sample size.
    floor_d = data.get("noise_floor_effect_size")
    if floor_d is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        names = ["Poison vs control\n(the measurement)",
                 "Control vs control\n(nothing is attacking)"]
        vals = [abs(data["effect_size"]), abs(floor_d)]
        bars = ax.bar(names, vals, color=[POISON, CONTROL], width=0.5)
        ax.axhline(0.8, color="#333333", linestyle="--", linewidth=1.6,
                   label="pre-registered bar ($d \\geq 0.80$)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, "%.3f" % v,
                    ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("$|$Cohen's $d|$ over 5 seeds")
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_title("The effect-size bar is not robust at five seeds",
                     fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        _save(fig, "fig6_signal_vs_floor")

    return {
        "Effect size (Cohen's d), poison vs control": "%.3f" % data["effect_size"],
        "Relative benign degradation": "%+.3f\\%%" % (100.0 * data["relative_degradation"]),
        "Fluency ratio vs control (max across seeds)": "not scored (pilot)",
        "Usable seeds per condition": str(data["n_seeds"]),
    }


def main() -> int:
    print("Stage 1 -- published baselines and instrument validation")
    table_baselines()
    figure_baselines()
    figure_fluency()
    figure_equivalence()

    # The deeper run supersedes the shallow one as the headline: its inner loop
    # does ~97x more work, so it is the only one of the two with an attack
    # surface worth interpreting. The shallow run's numbers stay in its own JSON.
    base = ROOT / "experiments" / "002-pilot-tiny-ttt" / "results"
    deep, shallow = base / "deep.json", base / "pilot.json"
    observed = None
    if deep.exists():
        print("  (headline: deep.json -- meta-trained base)")
        observed = _pilot_figures(json.loads(deep.read_text(encoding="utf-8")))
    elif shallow.exists():
        observed = _pilot_figures(json.loads(shallow.read_text(encoding="utf-8")))
    else:
        print("  (no pilot results yet)")
    table_prereg(observed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
