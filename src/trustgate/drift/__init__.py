"""Cumulative-drift budgeting. Home of the bounded-drift guarantee."""

from trustgate.drift.accumulator import accumulate, init_drift, reset_window, would_breach

__all__ = ["accumulate", "init_drift", "reset_window", "would_breach"]
