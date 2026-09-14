"""TRIGGER objective tests.

CPU only, no model required. Tests verify the interface and error messages.
"""

import numpy as np
import pytest

from trustgate.attack.trigger import (
    TriggerSpec,
    build_trigger_stream,
    evaluate_trigger,
    run_trigger_objective,
)


def test_trigger_spec_creation():
    """TriggerSpec holds trigger and target tokens."""
    trigger = np.array([100, 200, 300], dtype=np.int32)
    target = np.array([400, 500], dtype=np.int32)
    spec = TriggerSpec(trigger_tokens=trigger, target_tokens=target, trigger_count=2)
    assert np.array_equal(spec.trigger_tokens, trigger)
    assert np.array_equal(spec.target_tokens, target)
    assert spec.trigger_count == 2


def test_trigger_spec_default_count():
    """TriggerSpec defaults trigger_count to 1."""
    trigger = np.array([100], dtype=np.int32)
    target = np.array([200], dtype=np.int32)
    spec = TriggerSpec(trigger_tokens=trigger, target_tokens=target)
    assert spec.trigger_count == 1


def test_build_trigger_stream_raises_not_implemented():
    """T2.8: build_trigger_stream blocked on T1.10 (GPU)."""
    corpus = np.arange(1000, dtype=np.int32)
    trigger = np.array([100, 200], dtype=np.int32)
    target = np.array([300, 400], dtype=np.int32)
    spec = TriggerSpec(trigger_tokens=trigger, target_tokens=target)

    with pytest.raises(NotImplementedError, match="TRIGGER stream building blocked on T1.10"):
        build_trigger_stream(corpus, 256, 0, spec)


def test_evaluate_trigger_raises_not_implemented():
    """T2.8: evaluate_trigger blocked on T1.10 (GPU)."""
    trigger = np.array([100, 200], dtype=np.int32)
    target = np.array([300, 400], dtype=np.int32)
    spec = TriggerSpec(trigger_tokens=trigger, target_tokens=target)

    with pytest.raises(NotImplementedError, match="TRIGGER evaluation blocked on T1.10"):
        evaluate_trigger(None, None, spec, None)


def test_run_trigger_objective_raises_not_implemented():
    """T2.8: run_trigger_objective blocked on T1.10 (GPU)."""
    corpus = np.arange(1000, dtype=np.int32)
    trigger = np.array([100, 200], dtype=np.int32)
    target = np.array([300, 400], dtype=np.int32)
    spec = TriggerSpec(trigger_tokens=trigger, target_tokens=target)

    with pytest.raises(NotImplementedError, match="TRIGGER objective run blocked on T1.10"):
        run_trigger_objective(corpus, 256, [0, 1, 2], spec)


def test_run_trigger_objective_empty_seeds_raises():
    """T2.8: no seeds -> error."""
    corpus = np.arange(1000, dtype=np.int32)
    trigger = np.array([100], dtype=np.int32)
    target = np.array([200], dtype=np.int32)
    spec = TriggerSpec(trigger_tokens=trigger, target_tokens=target)

    with pytest.raises(ValueError, match="no seeds"):
        run_trigger_objective(corpus, 256, [], spec)
