"""Measure what the gate costs, against ADR-F1 section 5.1's ~10% budget.

`gate/anchor.py` says of the anchor forward pass: "That is the single largest
overhead in the design and the reason the section 5.1 budget is stated as
'<= ~10% of the per-window update cost'. Measure it; do not assume it."
Nothing measured it. `metrics.py` has `clean_regression` for the *accuracy* cost
of gating and no counterpart for the *compute* cost.

Two things this is careful about, because a naive timing would report a number
that is wrong in a flattering direction.

**Compilation is not overhead.** JAX traces and compiles on first call, and a
gated step compiles a bigger graph. Timing that first call would charge the gate
for a one-time cost that a served model pays once and amortises over every
window after. Hence `warmup`, which is not optional.

**Asynchronous dispatch is not speed.** JAX returns before the device is
finished, so timing without `block_until_ready` measures how fast Python can
queue work. Both arms are blocked here; a benchmark that forgets this reports
overhead near zero, which is exactly the answer nobody should believe.

The median is reported rather than the mean: one descheduled iteration on a
shared box moves a mean and does not move a median, and the question asked is
what a typical window costs, not what the worst one did.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

#: ADR-F1 section 5.1, from dossier section 5.1: "Gate compute overhead <= ~10%
#: of the per-window update cost (target; measure and report)." A target, not a
#: frozen bar -- unlike `PREREGISTERED.md`'s three, this one can move if the
#: measurement justifies it, and the justification has to be written down.
BUDGET_FRACTION = 0.10


@dataclass(frozen=True)
class OverheadResult:
    """Per-window cost of running the gate, in seconds and as a fraction."""

    ungated_seconds: float
    """Median wall-clock of one ungated update step."""

    gated_seconds: float
    """Median wall-clock of the same step with the gate installed."""

    repeats: int
    ungated_samples: tuple[float, ...]
    gated_samples: tuple[float, ...]

    @property
    def overhead_fraction(self) -> float:
        """Gate cost as a fraction of the ungated per-window update cost."""
        if self.ungated_seconds <= 0:
            raise ValueError(
                f"ungated step measured at {self.ungated_seconds}s, which is "
                f"not a usable denominator -- the timer resolution is larger "
                f"than the work. Raise the repeat count or the window size."
            )
        return (self.gated_seconds - self.ungated_seconds) / self.ungated_seconds

    def within_budget(self, budget: float = BUDGET_FRACTION) -> bool:
        """Exact comparison, deliberately -- with one caveat for the reader.

        A result within a fraction of a percent of the bar is inside the noise
        of any wall-clock measurement, and the budget itself is written "~10%".
        Do not report a marginal pass as a pass without quoting
        `overhead_fraction`; the number is the finding, this bool is a
        convenience.
        """
        return self.overhead_fraction <= budget

    def summary(self) -> str:
        verdict = "within" if self.within_budget() else "OVER"
        return (
            f"ungated {self.ungated_seconds * 1e3:.3f} ms/window, "
            f"gated {self.gated_seconds * 1e3:.3f} ms/window, "
            f"overhead {self.overhead_fraction * 100:+.1f}% "
            f"({verdict} the {BUDGET_FRACTION * 100:.0f}% budget), "
            f"median of {self.repeats}"
        )


def _block(value):
    """Wait for the device, so a timing measures work and not dispatch."""
    import jax

    return jax.block_until_ready(value)


def _time_one(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    _block(fn())
    return time.perf_counter() - start


def measure_overhead(
    ungated_step: Callable[[], object],
    gated_step: Callable[[], object],
    *,
    repeats: int = 20,
    warmup: int = 3,
) -> OverheadResult:
    """Time one window with and without the gate.

    Both callables must run exactly one window's worth of work and return
    something the device can be blocked on. They are interleaved rather than run
    in two blocks, so a thermal or contention drift partway through the
    measurement lands on both arms instead of only the second.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if warmup < 1:
        raise ValueError(
            f"warmup must be at least 1, got {warmup}: the first call to a "
            f"jitted function traces and compiles it, and charging the gate "
            f"for a one-time compilation reports an overhead no served model "
            f"would ever pay"
        )

    for _ in range(warmup):
        _block(ungated_step())
        _block(gated_step())

    ungated: list[float] = []
    gated: list[float] = []
    for _ in range(repeats):
        ungated.append(_time_one(ungated_step))
        gated.append(_time_one(gated_step))

    return OverheadResult(
        ungated_seconds=statistics.median(ungated),
        gated_seconds=statistics.median(gated),
        repeats=repeats,
        ungated_samples=tuple(ungated),
        gated_samples=tuple(gated),
    )
