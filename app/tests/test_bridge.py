import jax.numpy as jnp
from backend.app.tfm.simplicial_complex import annulus_complex
from backend.app.tfm.boundary import build_boundary_matrix
from backend.app.tfm.hodge import hodge_laplacian
from backend.app.tfm.bridge import cfm_path, cfm_transport_cost, tfm_path_spectral


def test_tfm_equals_cfm_when_kappa_zero_on_annulus():
    nodes, edges, faces = annulus_complex(n=18, with_faces=True)
    B1 = build_boundary_matrix(nodes, edges)
    B2 = build_boundary_matrix(edges, faces)
    L1 = hodge_laplacian(B1, B2)
    x0 = jnp.linspace(-1.0, 1.0, len(edges))
    x1 = jnp.cos(jnp.arange(len(edges), dtype=jnp.float32))
    assert jnp.allclose(cfm_path(x0, x1, 0.4), tfm_path_spectral(x0, x1, L1, 0.0, 0.4), atol=1e-5)


def test_cfm_transport_cost_matches_squared_l2_displacement():
    x0 = jnp.array([1.0, -1.0, 1.0])
    x1 = jnp.array([0.5, 0.5, -0.5])
    expected = jnp.sum((x1 - x0) ** 2)
    assert jnp.allclose(cfm_transport_cost(x0, x1), expected)
