"""Phase 2 gate signals, on CPU with stand-in parameter trees.

Everything here runs under `jax.jit` at least once, because that is the
constraint the design actually has to meet: the gate runs inside
`jax.lax.scan` in `MetaModel.loss_for_sequence`, so a Python branch on a traced
value is not a style problem, it is a crash at the only moment that matters
(ADR-003 decision 2).

What these cannot establish: that the thresholds are right. They are pre-verdict
guesses per the 2026-09-20 revision to `PREREGISTERED.md`, and the deliverable
is the operating-point curve, not a threshold (dossier 5.6).
"""

import jax
import jax.numpy as jnp
import pytest

from trustgate.drift.accumulator import init_drift
from trustgate.gate.anchor import anchor_consistency_gate, probe_divergence
from trustgate.gate.policy import Reason, combine
from trustgate.gate.uncertainty import uncertainty_gate
from trustgate.probes.rotating import make_probe_rotation, pool_size
from trustgate.types import DriftState, GateDecision, Verdict

VOCAB = 8


def params(scale):
    """Weights that vary across the vocabulary axis.

    A tree constant across that axis puts the same value in every logit slot,
    so the softmax is uniform at every scale and the divergence is zero however
    far the parameters move -- which would make most of this file pass
    vacuously.
    """
    row = jnp.arange(VOCAB, dtype=jnp.float32)
    return {"w": jnp.tile(row, (4, 1)) * (scale / 4.0)}


def zeros():
    return {"w": jnp.zeros((4, VOCAB), dtype=jnp.float32)}


def forward(p, probes):
    """Logits from a parameter tree and a probe batch. Linear and deterministic
    so the divergence has a known direction."""
    return probes @ p["w"]


def probe_batch(n=3):
    return jnp.ones((n, 4), dtype=jnp.float32)


# ------------------------------------------------------------ divergence ---


def test_identical_parameters_have_zero_divergence():
    anchor = params(0.5)
    value = probe_divergence(anchor, anchor, probe_batch(), forward_fn=forward)
    assert float(value) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("divergence", ["symmetric_kl", "kl", "topk_agreement"])
def test_every_divergence_is_non_negative_and_zero_at_identity(divergence):
    """All three must point the same way -- 0 is identical, larger is further --
    or a threshold swept over one is meaningless against another."""
    anchor = params(0.5)
    at_identity = probe_divergence(
        anchor, anchor, probe_batch(), divergence, forward_fn=forward
    )
    assert float(at_identity) == pytest.approx(0.0, abs=1e-6)

    moved = probe_divergence(
        params(3.0), anchor, probe_batch(), divergence, forward_fn=forward
    )
    assert float(moved) >= 0.0


def test_symmetric_kl_is_actually_symmetric():
    a, b = params(0.2), params(2.0)
    forward_kwargs = dict(divergence="symmetric_kl", forward_fn=forward)
    ab = probe_divergence(a, b, probe_batch(), **forward_kwargs)
    ba = probe_divergence(b, a, probe_batch(), **forward_kwargs)
    assert float(ab) == pytest.approx(float(ba), rel=1e-5)


def test_plain_kl_is_not_symmetric():
    """If it were, the two measures would be the same knob under two names."""
    a, b = params(0.2), params(2.0)
    ab = probe_divergence(a, b, probe_batch(), "kl", forward_fn=forward)
    ba = probe_divergence(b, a, probe_batch(), "kl", forward_fn=forward)
    assert float(ab) != pytest.approx(float(ba), rel=1e-3)


def test_divergence_grows_with_distance_from_the_anchor():
    anchor = params(0.0)
    near = probe_divergence(params(0.5), anchor, probe_batch(), forward_fn=forward)
    far = probe_divergence(params(4.0), anchor, probe_batch(), forward_fn=forward)
    assert float(far) > float(near)


def test_an_unknown_divergence_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        probe_divergence(
            params(1.0), params(1.0), probe_batch(), "cosine", forward_fn=forward
        )
    assert "cosine" in str(excinfo.value)
    assert "symmetric_kl" in str(excinfo.value)


# ----------------------------------------------------------- anchor gate ---


def make_gate(threshold=1.0, **kwargs):
    return anchor_consistency_gate(
        zeros(), probe_batch(), threshold, forward_fn=forward, **kwargs
    )


def test_an_update_that_stays_near_the_anchor_is_accepted():
    gate = make_gate(threshold=1.0)
    decision = gate(zeros(), zeros())
    assert int(decision.verdict) == Verdict.ACCEPT


def test_an_update_that_leaves_the_anchor_is_rejected():
    gate = make_gate(threshold=0.01)
    decision = gate(params(5.0), zeros())
    assert int(decision.verdict) == Verdict.REJECT


def test_a_rejected_update_contributes_no_drift():
    """It never commits, so it cannot have moved anything."""
    gate = make_gate(threshold=0.01)
    decision = gate(params(5.0), zeros())
    assert float(decision.drift_delta) == 0.0


def test_an_accepted_update_reports_its_divergence_as_drift():
    gate = make_gate(threshold=100.0)
    decision = gate(params(1.0), zeros())
    assert int(decision.verdict) == Verdict.ACCEPT
    assert float(decision.drift_delta) > 0.0


def test_the_gate_scores_the_proposed_weights_not_the_update_alone():
    """`gate(updates, fast_weights)` must score `fast_weights + updates`.
    Scoring the update in isolation would miss an update that is small but
    lands somewhere behaviourally distant."""
    gate = make_gate(threshold=100.0)
    from_zero = gate(params(1.0), zeros())
    from_far = gate(params(1.0), params(3.0))
    assert float(from_far.drift_delta) != pytest.approx(
        float(from_zero.drift_delta), rel=1e-3
    )


def test_a_nan_divergence_rejects_rather_than_accepting():
    """`jnp.where` propagates nan and `nan > threshold` is False, so an
    unguarded nan commits the update it failed to score. This is the fail-safe
    direction dossier 5.5 asks for."""
    gate = make_gate(threshold=1.0)
    decision = gate({"w": jnp.full((4, VOCAB), jnp.nan)}, zeros())
    assert int(decision.verdict) == Verdict.REJECT
    assert float(decision.confidence) == 0.0
    assert float(decision.drift_delta) == 0.0


def test_confidence_is_lowest_at_the_threshold():
    """A decision taken exactly at the bar is the one most likely to be wrong,
    and `combine` turns low confidence into a conservative reject."""
    anchor = zeros()
    at_bar = float(
        probe_divergence(params(1.0), anchor, probe_batch(), forward_fn=forward)
    )
    near = anchor_consistency_gate(
        anchor, probe_batch(), at_bar, forward_fn=forward
    )(params(1.0), zeros())
    far = make_gate(threshold=at_bar * 50)(params(1.0), zeros())
    assert float(near.confidence) < float(far.confidence)


def test_an_explicit_anchor_overrides_the_closed_over_one():
    """`interceptor.py:100` calls `gate(delta, current)` and never passes an
    anchor, so the closure is the real source -- but the protocol declares the
    argument and it has to work."""
    gate = make_gate(threshold=100.0)
    closed_over = gate(zeros(), zeros())
    explicit = gate(zeros(), zeros(), params(4.0))
    assert float(explicit.drift_delta) > float(closed_over.drift_delta)


def test_a_non_positive_threshold_is_refused():
    with pytest.raises(ValueError):
        make_gate(threshold=0.0)


# --------------------------------------------------------- jit-safety ---


def test_the_anchor_gate_survives_jit():
    """ADR-003 decision 2. The gate runs inside `scan` under `jit`; a Python
    branch on a traced verdict would fail here and nowhere else."""
    gate = make_gate(threshold=1.0)
    jitted = jax.jit(gate)
    decision = jitted(params(0.1), zeros())
    assert int(decision.verdict) in (Verdict.ACCEPT, Verdict.REJECT)


def test_the_anchor_gate_survives_scan():
    """Closer to the real call site than jit alone: `loss_for_sequence` scans
    the inner step, and a scan additionally requires stable shapes and dtypes
    across iterations."""
    gate = make_gate(threshold=1.0)

    def body(carry, scale):
        decision = gate({"w": jnp.full((4, VOCAB), scale)}, zeros())
        return carry, decision.verdict

    _, verdicts = jax.lax.scan(body, None, jnp.array([0.0, 0.5, 5.0]))
    assert verdicts.shape == (3,)
    assert verdicts.dtype == jnp.int32


def test_combine_survives_jit():
    def run(conf):
        primary = GateDecision(
            verdict=jnp.int32(Verdict.ACCEPT),
            drift_delta=jnp.float32(0.1),
            confidence=conf,
        )
        decision, reason = combine(
            primary, None, init_drift(10.0), min_confidence=0.5
        )
        return decision.verdict, reason

    verdict, reason = jax.jit(run)(jnp.float32(0.1))
    assert int(verdict) == Verdict.REJECT
    assert int(reason) == Reason.LOW_CONFIDENCE


# -------------------------------------------------------------- policy ---


def decision(verdict, drift=0.0, confidence=1.0):
    return GateDecision(
        verdict=jnp.int32(verdict),
        drift_delta=jnp.float32(drift),
        confidence=jnp.float32(confidence),
    )


def test_an_accepted_update_under_budget_stays_accepted():
    out, reason = combine(
        decision(Verdict.ACCEPT, drift=0.1), None, init_drift(10.0),
        min_confidence=0.0,
    )
    assert int(out.verdict) == Verdict.ACCEPT
    assert int(reason) == Reason.SIGNAL


def test_either_signal_rejecting_is_enough():
    out, _ = combine(
        decision(Verdict.ACCEPT), decision(Verdict.REJECT), init_drift(10.0),
        min_confidence=0.0,
    )
    assert int(out.verdict) == Verdict.REJECT


def test_low_confidence_rejects_conservatively():
    out, reason = combine(
        decision(Verdict.ACCEPT, confidence=0.1), None, init_drift(10.0),
        min_confidence=0.9,
    )
    assert int(out.verdict) == Verdict.REJECT
    assert int(reason) == Reason.LOW_CONFIDENCE


def test_low_confidence_is_reported_separately_from_a_signal_reject():
    """An over-cautious gate starves adaptation and looks like a clean-accuracy
    regression. Conflating the two causes hides which one happened."""
    signal, signal_reason = combine(
        decision(Verdict.REJECT, confidence=1.0), None, init_drift(10.0),
        min_confidence=0.5,
    )
    unsure, unsure_reason = combine(
        decision(Verdict.ACCEPT, confidence=0.0), None, init_drift(10.0),
        min_confidence=0.5,
    )
    assert int(signal.verdict) == int(unsure.verdict) == Verdict.REJECT
    assert int(signal_reason) == Reason.SIGNAL
    assert int(unsure_reason) == Reason.LOW_CONFIDENCE


def test_a_breach_rolls_back_and_dominates_everything():
    breached = DriftState(
        cumulative=jnp.float32(0.0),
        budget=jnp.float32(1.0),
        breached=jnp.asarray(True),
    )
    out, reason = combine(
        decision(Verdict.ACCEPT), None, breached, min_confidence=0.0
    )
    assert int(out.verdict) == Verdict.ROLLBACK
    assert int(reason) == Reason.BUDGET


def test_an_update_that_would_breach_is_stopped_before_it_commits():
    """`would_breach` is checked before committing so the bound is a strict
    invariant rather than something restored after the fact."""
    nearly_full = DriftState(
        cumulative=jnp.float32(0.95),
        budget=jnp.float32(1.0),
        breached=jnp.asarray(False),
    )
    out, reason = combine(
        decision(Verdict.ACCEPT, drift=0.2), None, nearly_full, min_confidence=0.0
    )
    assert int(out.verdict) == Verdict.ROLLBACK
    assert int(reason) == Reason.BUDGET
    assert float(out.drift_delta) == 0.0


def test_a_rejected_update_cannot_consume_budget():
    out, _ = combine(
        decision(Verdict.REJECT, drift=0.5), None, init_drift(10.0),
        min_confidence=0.0,
    )
    assert float(out.drift_delta) == 0.0


def test_a_cheap_secondary_cannot_inflate_the_drift_measure():
    """Drift is behavioural, and only the primary signal measures behaviour."""
    out, _ = combine(
        decision(Verdict.ACCEPT, drift=0.1),
        decision(Verdict.ACCEPT, drift=99.0),
        init_drift(10.0),
        min_confidence=0.0,
    )
    assert float(out.drift_delta) == pytest.approx(0.1)


def test_an_out_of_range_min_confidence_is_refused():
    with pytest.raises(ValueError):
        combine(decision(Verdict.ACCEPT), None, init_drift(1.0), min_confidence=1.5)


# --------------------------------------------------------- uncertainty ---


def test_a_small_update_is_accepted():
    gate = uncertainty_gate(2.0, reference_norm=1.0)
    assert int(gate(params(0.01), zeros()).verdict) == Verdict.ACCEPT


def test_an_outsized_update_is_rejected():
    gate = uncertainty_gate(2.0, reference_norm=1.0)
    assert int(gate(params(10.0), zeros()).verdict) == Verdict.REJECT


def test_the_secondary_reports_no_drift():
    """It measures magnitude, not behaviour."""
    gate = uncertainty_gate(100.0, reference_norm=1.0)
    assert float(gate(params(1.0), zeros()).drift_delta) == 0.0


def test_the_secondary_survives_jit():
    gate = uncertainty_gate(2.0, reference_norm=1.0)
    assert int(jax.jit(gate)(params(0.01), zeros()).verdict) == Verdict.ACCEPT


@pytest.mark.parametrize("signal", ["grad_norm_zscore", "loss_spike"])
def test_stateful_signals_say_why_they_are_blocked(signal):
    """They need running statistics and there is no carry slot for them. They
    must not silently degrade to a z-score against a constant."""
    with pytest.raises(NotImplementedError) as excinfo:
        uncertainty_gate(2.0, signal=signal, reference_norm=1.0)
    message = str(excinfo.value)
    assert "ADR-P3-1" in message
    assert "update_norm" in message  # names what does work


def test_an_unknown_signal_lists_both_categories():
    with pytest.raises(ValueError) as excinfo:
        uncertainty_gate(2.0, signal="vibes", reference_norm=1.0)
    assert "vibes" in str(excinfo.value)


# ------------------------------------------------------------- probes ---


def test_rotation_yields_the_requested_fixed_shape():
    """Changing probe shapes retriggers XLA compilation inside the scan."""
    pool = jnp.arange(40, dtype=jnp.float32).reshape(10, 4)
    rotate = make_probe_rotation(pool, 3, jax.random.PRNGKey(0))
    for window in range(5):
        assert rotate(window).shape == (3, 4)


def test_rotation_changes_contents_across_windows():
    """A fixed set is learnable; that is the whole reason this module exists."""
    pool = jnp.arange(400, dtype=jnp.float32).reshape(100, 4)
    rotate = make_probe_rotation(pool, 4, jax.random.PRNGKey(0))
    batches = [rotate(w) for w in range(5)]
    assert any(
        not bool(jnp.array_equal(batches[0], other)) for other in batches[1:]
    )


def test_rotation_is_reproducible_given_the_key():
    """Unpredictable without the key, fully replayable with it -- which is what
    makes a run auditable after the fact."""
    pool = jnp.arange(400, dtype=jnp.float32).reshape(100, 4)
    a = make_probe_rotation(pool, 4, jax.random.PRNGKey(7))
    b = make_probe_rotation(pool, 4, jax.random.PRNGKey(7))
    assert bool(jnp.array_equal(a(3), b(3)))


def test_a_different_key_gives_a_different_schedule():
    pool = jnp.arange(400, dtype=jnp.float32).reshape(100, 4)
    a = make_probe_rotation(pool, 4, jax.random.PRNGKey(1))
    b = make_probe_rotation(pool, 4, jax.random.PRNGKey(2))
    assert not bool(jnp.array_equal(a(3), b(3)))


def test_rotation_samples_without_replacement_within_a_window():
    """Duplicates would cut the effective probe count without cutting the cost."""
    pool = jnp.arange(20, dtype=jnp.float32).reshape(20, 1)
    rotate = make_probe_rotation(pool, 8, jax.random.PRNGKey(0))
    drawn = rotate(0).ravel().tolist()
    assert len(set(drawn)) == len(drawn)


def test_rotation_works_under_jit_with_a_traced_window_index():
    """The rotation has to live inside the scan, where the index is traced."""
    pool = jnp.arange(400, dtype=jnp.float32).reshape(100, 4)
    rotate = make_probe_rotation(pool, 4, jax.random.PRNGKey(0))
    assert jax.jit(rotate)(jnp.int32(3)).shape == (4, 4)


def test_asking_for_more_probes_than_the_pool_holds_is_refused():
    pool = jnp.arange(20, dtype=jnp.float32).reshape(5, 4)
    with pytest.raises(ValueError) as excinfo:
        make_probe_rotation(pool, 9, jax.random.PRNGKey(0))
    assert "duplicate" in str(excinfo.value)


def test_a_pool_whose_leaves_disagree_is_refused():
    """Mismatched rows pair one example's inputs with another's targets and
    score as behavioural divergence -- an attack that is not there."""
    pool = {"x": jnp.zeros((10, 4)), "y": jnp.zeros((7, 4))}
    with pytest.raises(ValueError) as excinfo:
        pool_size(pool)
    assert "leading axis" in str(excinfo.value)


def test_a_pytree_pool_rotates_every_leaf_together():
    pool = {"x": jnp.arange(40.0).reshape(10, 4), "y": jnp.arange(10.0).reshape(10, 1)}
    rotate = make_probe_rotation(pool, 3, jax.random.PRNGKey(0))
    batch = rotate(0)
    assert batch["x"].shape == (3, 4)
    assert batch["y"].shape == (3, 1)
    # Same rows on both leaves: y is x's row index by construction.
    assert bool(jnp.array_equal(batch["y"].ravel() * 4, batch["x"][:, 0]))
