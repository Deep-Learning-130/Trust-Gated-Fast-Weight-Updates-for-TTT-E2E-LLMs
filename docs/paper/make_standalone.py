import io, sys
sys.path.insert(0, 'docs/paper')
from pathlib import Path
from check_main import expand, MAIN

HERE = Path('docs/paper').resolve()
text, missing = expand(MAIN)
if missing:
    raise SystemExit("missing inputs: %s" % missing)

banner = (
    "%% =====================================================================\n"
    "%% Trust-Gated Fast-Weight Updates for TTT-E2E Large Language Models\n"
    "%%\n"
    "%% SELF-CONTAINED. Paste this whole file into Overleaf as main.tex and\n"
    "%% compile with pdfLaTeX. No .bib file, no image files, no uploads: every\n"
    "%% figure is TikZ/pgfplots with inline coordinates and the bibliography is\n"
    "%% a thebibliography block.\n"
    "%%\n"
    "%% GENERATED -- do not edit here. Edit the section sources and rebuild:\n"
    "%%   python docs/paper/build_main.py && python docs/paper/make_standalone.py\n"
    "%% =====================================================================\n"
)
out = HERE / "main-overleaf.tex"
out.write_text(banner + text, encoding="utf-8")
n = len(text.split("\n"))
print("wrote %s" % out)
print("  %d lines, %d KB" % (n, len(text) // 1024))
