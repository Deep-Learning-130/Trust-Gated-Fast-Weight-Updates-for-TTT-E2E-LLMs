"""Assemble the complete paper into `main.tex`.

The paper is written in two files that each stand alone:

  p3-intro-related-methodology.tex   Sections I-III + the 35-entry bibliography
  p4-experiments-results.tex         Sections IV-V

Rather than duplicate either, this script splices them into a single compilable
document: p3's preamble (extended with `pgfplots`, which Section IV's figures
need), a new abstract and keyword block, p3's Sections I-III verbatim, p4's
Sections IV-V, and p3's bibliography verbatim.

Written as a script rather than done by hand so that editing a section file and
rebuilding cannot leave `main.tex` silently stale -- the same reason the figures
are generated rather than drawn.

Run:  python docs/paper/build_main.py      (then compile main.tex)
"""

from __future__ import annotations

import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
P3 = HERE / "p3-intro-related-methodology.tex"
P4 = HERE / "p4-experiments-results.tex"
OUT = HERE / "main.tex"

# Landmarks are located rather than hardcoded: line numbers in a 1,600-line
# source drift the moment anyone edits a section, and a silent off-by-one would
# splice the document in the wrong place.
def _find(lines, pred, what):
    for i, line in enumerate(lines):
        if pred(line):
            return i
    raise SystemExit("build_main: could not locate %s in p3" % what)


ABSTRACT = r"""
\begin{abstract}
Test-time training compresses long context into fast weights that a model
updates by next-token prediction as it reads, which turns the serving path into a
write path into model state. We formalise the resulting threat --- a
benign-looking input stream that steers those updates so later, unrelated
inputs are handled worse, with no write access to the pretrained weights --- and
specify a trust gate that admits each proposed update only if it is consistent
with a frozen anchor and within a bounded cumulative-drift budget, restoring a
checkpoint in $O(1)$ on breach; we prove a per-window drift bound for that gated
commit, and record which part of it our own overlay does not yet enforce.
Whether the gate is worth building is decided by a
criterion pre-registered before any attack code existed. In a scaled-down pilot,
the measured effect size clears that criterion --- yet a negative control with no
attacker clears it on 55\% of draws, and the observed value sits at the median of
that null. We therefore report no attack, and contribute a measured
false-positive rate for the criterion itself.
\end{abstract}

\begin{IEEEkeywords}
Test-time training, fast weights, data poisoning, long-context language models,
robust adaptation, inference-time security, pre-registration.
\end{IEEEkeywords}

"""

# Section IV's figures are pgfplots; p3 needed only tikz.
EXTRA_PACKAGES = r"""\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
"""


def read_lines(p: Path) -> list[str]:
    return io.open(p, encoding="utf-8").read().split("\n")


def main() -> int:
    p3 = read_lines(P3)

    i_title = _find(p3, lambda l: l.strip() == r"\maketitle",
                    r"\maketitle")
    i_intro = _find(p3, lambda l: l.strip() == r"\section{Introduction}",
                    "Section I")
    i_bib = _find(p3, lambda l: l.startswith(r"\begin{thebibliography}"),
                  "the bibliography")
    if not i_title < i_intro < i_bib:
        raise SystemExit("build_main: p3 landmarks are out of order")

    preamble = p3[:i_title + 1]
    body = p3[i_intro:i_bib]
    bib = p3[i_bib:]

    # pgfplots must load after tikz; \usetikzlibrary is the last package line.
    for i, line in enumerate(preamble):
        if line.startswith(r"\usetikzlibrary"):
            preamble.insert(i + 1, EXTRA_PACKAGES)
            break
    else:
        raise SystemExit("could not find \\usetikzlibrary in p3's preamble")

    p4 = io.open(P4, encoding="utf-8").read()

    # Merge p4's "Experimental Setup" and "Results and Discussion" into one
    # numbered section, so the paper reads I-V: Introduction, Related Work,
    # Proposed Methodology, Experimental Evaluation, Conclusion.
    p4 = p4.replace(r"\section{Experimental Setup}",
                    r"\section{Experimental Evaluation}")
    p4 = p4.replace("\\section{Results and Discussion}\n\\label{sec:results}\n\n", "")
    # p4 already labels its own "Validity checks" subsection; nothing to
    # inject here beyond dropping the now-redundant section heading.
    out = "\n".join(preamble) + "\n" + ABSTRACT + "\n" + "\n".join(body) \
        + "\n" + p4.strip() + "\n\n" + "\n".join(bib)

    io.open(OUT, "w", encoding="utf-8").write(out)

    n_fig = out.count(r"\begin{figure}") + out.count(r"\input{figures/fig")
    n_tab = out.count(r"\begin{table}") + out.count(r"\input{figures/table")
    print("wrote %s" % OUT)
    print("  sections   %d" % out.count("\n\\section{"))
    print("  bibitems   %d" % out.count(r"\bibitem"))
    print("  figures    ~%d   tables ~%d" % (n_fig, n_tab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
