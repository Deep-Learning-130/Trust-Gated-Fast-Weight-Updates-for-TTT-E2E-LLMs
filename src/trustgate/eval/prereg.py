"""The frozen decision thresholds, and a check that they still match the doc.

Why constants here plus a test, rather than parsing the doc at runtime
---------------------------------------------------------------------
Standing Rule 5 requires the pre-registered bar to be the single source of
truth, and task-allocation T3.3 says the thresholds must be "passed in from the
`PREREGISTERED.md` values -- never hard-coded inside the report code". Two ways
to honour that:

1. Parse `PREREGISTERED.md` at report time. Single source of truth, but it
   relocates the failure to the worst possible moment -- the end of a paid GPU
   run, when the report is generated and the markdown turns out to have been
   reformatted.
2. Constants here, plus `tests/test_thresholds.py` parsing the doc and asserting
   equality. Identical guarantee -- a bar cannot move without CI going red --
   but it fails at PR time.

(2). Same protection, better failure timing, and no runtime dependency on a
document's formatting.

This module is a new place a bar could be moved, which the review checklist's
"diff to `PREREGISTERED.md`" check does not cover. `docs/protocols/branch-and-review.md`
gains rule 5b for exactly that.

The percent trap
----------------
The doc writes the degradation bar as `**>= 10%**` while `metrics.py` returns a
*fraction* (`relative_degradation = (poisoned - control) / control`). A parser
yielding `10.0` instead of `0.10` would set the bar at 1000% and manufacture a
STOP from a genuine positive -- the project-killing direction. Both the value
and that failure direction are tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: `experiments/001-attack-spike/PREREGISTERED.md`, written 2026-07-28 before any
#: attack code ran.
PREREGISTERED_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "001-attack-spike"
    / "PREREGISTERED.md"
)


@dataclass(frozen=True)
class Thresholds:
    """The three gating bars. All three are required, not any of them."""

    min_effect_size: float
    """Cohen's *d*, poison vs control, across seeds. Conventional "large"."""

    min_relative_degradation: float
    """A **fraction**, not a percentage. Below this the harm is
    real-but-trivial and would not motivate a defense."""

    max_fluency_ratio: float
    """Poison perplexity / control perplexity under an independent reference
    model. Above this the stream is not benign-looking and fails the dossier §1
    threat model, whatever the corruption numbers say."""


#: Frozen 2026-07-28. Moving any of these is a dated, reasoned revision inside
#: `PREREGISTERED.md` -- never an edit here.
FROZEN = Thresholds(
    min_effect_size=0.8,
    min_relative_degradation=0.10,
    max_fluency_ratio=1.5,
)

#: Substring of the criterion cell -> field name. Substrings rather than exact
#: matches so a wording tweak in the doc does not read as a moved bar, while a
#: renamed *criterion* still fails the row-count check below.
_CRITERION_KEYS = {
    "Effect size": "min_effect_size",
    "Relative benign degradation": "min_relative_degradation",
    "Fluency ratio": "max_fluency_ratio",
}

_SECTION = "## Decision thresholds (frozen)"
_NUMBER = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(%?)")


def parse_preregistered(text: str) -> Thresholds:
    """Recover the three bars from the pre-registration's own table.

    Deliberately strict. Anything it cannot read with certainty is an error,
    because the alternative -- guessing -- is how a bar moves without anyone
    deciding to move it.
    """
    if _SECTION not in text:
        raise ValueError(f"{_SECTION!r} section not found in PREREGISTERED.md")

    section = text.split(_SECTION, 1)[1].split("\n## ", 1)[0]

    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if cells[0].startswith("---") or cells[0] == "Criterion":
            continue
        rows.append((cells[0], cells[1]))

    if len(rows) != 3:
        raise ValueError(
            f"expected exactly 3 criterion rows in the frozen thresholds table, "
            f"found {len(rows)}: {[r[0] for r in rows]}. A criterion was added, "
            f"removed or renamed -- treat this as a threshold change and review "
            f"it as one."
        )

    values: dict[str, float] = {}
    for criterion, cell in rows:
        field = next(
            (f for key, f in _CRITERION_KEYS.items() if key in criterion), None
        )
        if field is None:
            raise ValueError(
                f"unrecognised criterion {criterion!r}; cannot map it to a "
                f"threshold field without guessing"
            )

        match = _NUMBER.search(cell)
        if match is None:
            raise ValueError(f"no number in threshold cell {cell!r} for {criterion!r}")

        number = float(match.group(1))
        if match.group(2) == "%":
            # The doc says "10%"; the code compares against a fraction.
            number /= 100.0
        values[field] = number

    if set(values) != set(_CRITERION_KEYS.values()):
        raise ValueError(
            f"parsed {sorted(values)} but need "
            f"{sorted(set(_CRITERION_KEYS.values()))}"
        )

    return Thresholds(**values)


def load_preregistered(path: Path | None = None) -> Thresholds:
    """Parse the doc from disk. Used by the test that pins `FROZEN` to it."""
    path = PREREGISTERED_PATH if path is None else Path(path)
    return parse_preregistered(path.read_text(encoding="utf-8"))
