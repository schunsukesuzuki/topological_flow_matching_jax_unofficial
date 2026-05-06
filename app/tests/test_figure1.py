import jax.numpy as jnp
from backend.app.tfm.figure1 import build_figure1_payload


def test_figure1_payload_has_five_examples_and_bounded_visual_range():
    payload = build_figure1_payload(kappa=2.0, n=18)
    assert len(payload["labels"]) == 5
    assert len(payload["graph_signals"]) == 5
    assert len(payload["edge_signals"]) == 5

    graph_zero = jnp.asarray(payload["graph_signals"][2])
    # Constant zero-eigenfunction should no longer be fully saturated.
    assert jnp.max(jnp.abs(graph_zero)) <= 0.6

    edge_high = jnp.asarray(payload["edge_signals"][4])
    assert jnp.max(jnp.abs(edge_high)) <= 0.85
