"""The frozen bars in code must equal the frozen bars in the document.

This is the mechanism that makes Standing Rule 5 enforceable: a threshold cannot
move in either place without CI going red. It runs at PR time rather than at the
end of a paid GPU run, which is the whole reason the constants are not parsed
from markdown at report time.
"""

import pytest

from trustgate.eval.prereg import (
    FROZEN,
    PREREGISTERED_PATH,
    Thresholds,
    load_preregistered,
    parse_preregistered,
)

TABLE = """
## Decision thresholds (frozen)

| Criterion | Threshold | Rationale |
|---|---|---|
| Effect size (Cohen's *d*, poison vs control) | **>= 0.8** | large |
| Relative benign degradation | **>= 10%** | not trivial |
| Fluency ratio (poison perplexity / control perplexity) | **<= 1.5** | benign-looking |

## Secondary
"""


def test_frozen_constants_equal_the_preregistered_document():
    # The one assertion this whole module exists for.
    assert load_preregistered() == FROZEN


def test_the_preregistration_actually_exists_where_we_look_for_it():
    assert PREREGISTERED_PATH.is_file(), PREREGISTERED_PATH


def test_frozen_values_are_the_ones_written_on_2026_07_28():
    assert FROZEN == Thresholds(
        min_effect_size=0.8,
        min_relative_degradation=0.10,
        max_fluency_ratio=1.5,
    )


def test_relative_degradation_bar_is_a_fraction_not_a_percent():
    # The doc says "10%"; metrics.py returns a fraction. A parser yielding 10.0
    # would set the bar at 1000% and manufacture a STOP from a genuine positive
    # -- the project-killing direction.
    assert FROZEN.min_relative_degradation < 1.0
    assert load_preregistered().min_relative_degradation == pytest.approx(0.10)


def test_parser_reads_a_well_formed_table():
    parsed = parse_preregistered(TABLE)
    assert parsed == Thresholds(0.8, 0.10, 1.5)


def test_parser_requires_exactly_three_criterion_rows():
    extra = TABLE.replace(
        "| Fluency ratio (poison perplexity / control perplexity) | **<= 1.5** | benign-looking |",
        "| Fluency ratio (poison perplexity / control perplexity) | **<= 1.5** | benign-looking |\n"
        "| Some new criterion | **>= 2.0** | added later |",
    )
    with pytest.raises(ValueError, match="exactly 3 criterion rows"):
        parse_preregistered(extra)


def test_parser_rejects_a_removed_criterion():
    fewer = TABLE.replace(
        "| Relative benign degradation | **>= 10%** | not trivial |\n", ""
    )
    with pytest.raises(ValueError, match="exactly 3 criterion rows"):
        parse_preregistered(fewer)


def test_parser_rejects_a_renamed_criterion():
    renamed = TABLE.replace("Relative benign degradation", "Relative wobble")
    with pytest.raises(ValueError, match="unrecognised criterion"):
        parse_preregistered(renamed)


def test_parser_rejects_a_missing_section():
    with pytest.raises(ValueError, match="section not found"):
        parse_preregistered("# Some other document\n")


def test_parser_rejects_a_threshold_cell_with_no_number():
    broken = TABLE.replace("**>= 0.8**", "**TBD**")
    with pytest.raises(ValueError, match="no number in threshold cell"):
        parse_preregistered(broken)
