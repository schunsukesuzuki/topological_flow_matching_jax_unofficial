import jax.numpy as jnp
from backend.app.tfm.simplicial_complex import annulus_complex
from backend.app.tfm.boundary import build_boundary_matrix
from backend.app.tfm.hodge import hodge_laplacian


def test_l1_is_psd_on_annulus():
    nodes, edges, faces = annulus_complex(n=18, with_faces=True)
    B1 = build_boundary_matrix(nodes, edges)
    B2 = build_boundary_matrix(edges, faces)
    L1 = hodge_laplacian(B1, B2)
    evals = jnp.linalg.eigvalsh(L1)
    assert jnp.min(evals) > -1e-5
