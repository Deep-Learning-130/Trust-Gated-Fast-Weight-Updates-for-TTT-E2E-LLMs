"""Interceptor behaviour, tested against lightweight stand-ins for the vendor
MetaModel / InnerLoopStepResult so no checkpoint or GPU is needed.

The stand-ins reproduce exactly the surface the interceptor touches:
`self.inner_parameters()`, `result.new_model`, `result.new_optimizer_state`,
`result.new_state`, `result.metrics`, and `type(result)(...)`. If the vendor
call site drifts from this shape after a submodule bump, these tests should be
the first thing that fails -- that is the tripwire promised in vendor_patch.py.
"""

import equinox as eqx
import jax.numpy as jnp

from trustgate.interceptor import (
    make_gated_inner_loop_step,
    norm_threshold_gate,
    passthrough_gate,
)


# eqx.Module, not a plain dataclass: the real MetaModel is a registered pytree,
# and `tree_select` in the interceptor only descends into pytree leaves. A plain
# dataclass would be an opaque leaf and the select would silently no-op -- which
# is exactly the bug an earlier version of this test hid.
class FakeModel(eqx.Module):
    params: dict

    def inner_parameters(self):
        return self.params


class FakeResult(eqx.Module):
    new_model: FakeModel
    new_optimizer_state: object
    new_state: object
    metrics: dict


def make_original(delta_scale):
    """A fake vendor step that adds `delta_scale` to every inner weight."""

    def original(self, opt_state, state_tuple, seq, prefix_outputs):
        new_params = {k: v + delta_scale for k, v in self.params.items()}
        return FakeResult(
            new_model=FakeModel(new_params),
            new_optimizer_state=opt_state,
            new_state=state_tuple,
            metrics={"loss": jnp.float32(1.0)},
        )

    return original


def _model():
    return FakeModel({"w": jnp.ones((2, 2), dtype=jnp.float32)})


def test_passthrough_commits_the_update():
    step = make_gated_inner_loop_step(make_original(0.5), passthrough_gate)
    out = step(_model(), None, None, None, None)
    # accept-all: committed weights equal the proposed update
    assert jnp.allclose(out.new_model.params["w"], 1.5)


def test_passthrough_is_a_noop_versus_ungated():
    original = make_original(0.5)
    gated = make_gated_inner_loop_step(original, passthrough_gate)
    ungated_out = original(_model(), None, None, None, None)
    gated_out = gated(_model(), None, None, None, None)
    assert jnp.allclose(gated_out.new_model.params["w"], ungated_out.new_model.params["w"])


def test_reject_reverts_to_pre_update_weights():
    # Update norm is ||0.5 * ones(2,2)|| = 1.0; threshold below that rejects.
    step = make_gated_inner_loop_step(make_original(0.5), norm_threshold_gate(max_norm=0.5))
    out = step(_model(), None, None, None, None)
    assert jnp.allclose(out.new_model.params["w"], 1.0)  # reverted
    assert float(out.metrics["gate/accepted"]) == 0.0


def test_accept_when_under_threshold():
    step = make_gated_inner_loop_step(make_original(0.5), norm_threshold_gate(max_norm=2.0))
    out = step(_model(), None, None, None, None)
    assert jnp.allclose(out.new_model.params["w"], 1.5)  # committed
    assert float(out.metrics["gate/accepted"]) == 1.0


def test_metrics_are_populated():
    step = make_gated_inner_loop_step(make_original(0.5), passthrough_gate)
    out = step(_model(), None, None, None, None)
    for key in ("gate/accepted", "gate/drift_delta", "gate/confidence", "gate/update_norm"):
        assert key in out.metrics
    assert float(out.metrics["gate/update_norm"]) == 1.0


def test_result_type_is_preserved():
    step = make_gated_inner_loop_step(make_original(0.5), passthrough_gate)
    out = step(_model(), None, None, None, None)
    assert isinstance(out, FakeResult)
