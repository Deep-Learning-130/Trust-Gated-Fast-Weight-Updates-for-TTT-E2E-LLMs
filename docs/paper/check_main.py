"""Static checks on `main.tex`, standing in for a compiler we do not have here.

This machine has no LaTeX installation, so `main.tex` has never been run through
`pdflatex`. That is worth stating plainly rather than implying a clean build. What
*can* be checked without a compiler is checked here, over the assembled document
with every `\\input` expanded:

  1. every `\\begin{X}` has a matching `\\end{X}`, correctly nested
  2. every `\\ref` / `\\eqref` target is defined by some `\\label`
  3. every `\\cite` key is defined by some `\\bibitem`
  4. every `\\input` resolves to a file that exists
  5. no duplicate `\\label`
  6. braces balance, and `$` delimiters pair

These catch the failures that actually bite when a paper is assembled from
fragments -- a dangling reference, a stale citation key, a missing figure file.
They do not catch spacing, overfull boxes, or package conflicts.

Run:  python docs/paper/check_main.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE / "main.tex"


def expand(path: Path, seen=None) -> tuple[str, list[str]]:
    """Inline every \\input, returning the flattened source and any missing files."""
    seen = seen or set()
    missing: list[str] = []
    text = io.open(path, encoding="utf-8").read()

    def sub(m):
        target = m.group(1)
        p = HERE / (target if target.endswith(".tex") else target + ".tex")
        if not p.exists():
            missing.append(target)
            return ""
        if str(p) in seen:
            return ""
        seen.add(str(p))
        inner, inner_missing = expand(p, seen)
        missing.extend(inner_missing)
        return inner

    return re.sub(r"\\input\{([^}]+)\}", sub, text), missing


def strip_comments(s: str) -> str:
    """Drop LaTeX comments, keeping escaped percent signs."""
    out = []
    for line in s.split("\n"):
        i, esc = 0, False
        while i < len(line):
            c = line[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "%":
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def main() -> int:
    if not MAIN.exists():
        print("main.tex not found -- run build_main.py first")
        return 1

    raw, missing_inputs = expand(MAIN)
    s = strip_comments(raw)
    problems = 0

    # 1. environments
    stack, env_err = [], []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, name = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(name)
        elif not stack:
            env_err.append("stray \\end{%s}" % name)
        elif stack[-1] != name:
            env_err.append("\\end{%s} closes \\begin{%s}" % (name, stack[-1]))
            stack.pop()
        else:
            stack.pop()
    for name in stack:
        env_err.append("unclosed \\begin{%s}" % name)

    # 2/5. labels and references
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", s))
    undefined_refs = sorted(refs - set(labels))

    # 3. citations
    keys = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", s):
        cited.update(k.strip() for k in group.split(","))
    undefined_cites = sorted(cited - keys)
    uncited = sorted(keys - cited)

    # 6. braces and math delimiters
    depth, brace_err = 0, None
    esc = False
    for i, c in enumerate(s):
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0 and brace_err is None:
                brace_err = "unbalanced } near offset %d" % i
    if depth != 0 and brace_err is None:
        brace_err = "%d unclosed {" % depth
    n_math = len(re.findall(r"(?<!\\)\$", s.replace("$$", "")))

    def report(title, items, fatal=True):
        nonlocal problems
        if items:
            if fatal:
                problems += len(items)
            print("  %-24s %s" % (title, "FAIL" if fatal else "note"))
            for x in items[:12]:
                print("      - %s" % x)
            if len(items) > 12:
                print("      ... and %d more" % (len(items) - 12))
        else:
            print("  %-24s OK" % title)

    print("Static check of main.tex (no compiler available on this machine)\n")
    print("  %-24s %d chars, %d sections"
          % ("expanded source", len(s), s.count("\n\\section{")))
    report("\\input files resolve", sorted(set(missing_inputs)))
    report("environments balanced", env_err)
    report("references defined", undefined_refs)
    report("citation keys defined", undefined_cites)
    report("duplicate labels", dupes)
    report("braces balanced", [brace_err] if brace_err else [])
    report("$ delimiters paired",
           ["odd number of $ (%d)" % n_math] if n_math % 2 else [])
    report("uncited bibitems", uncited, fatal=False)

    figs = len(re.findall(r"\\begin\{figure\}", s))
    tabs = len(re.findall(r"\\begin\{table\}", s))
    algs = len(re.findall(r"\\begin\{algorithm\}", s))
    print("\n  figures %d   tables %d   algorithms %d   bibitems %d"
          % (figs, tabs, algs, len(keys)))

    print("\n%s" % ("PASS -- no structural problems found" if not problems
                    else "FAIL -- %d problem(s)" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
