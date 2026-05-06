import jax.numpy as jnp
from backend.app.tfm.simplicial_complex import triangle_complex, annulus_complex
from backend.app.tfm.boundary import build_boundary_matrix


def test_triangle_boundary_of_boundary_is_zero():
    nodes, edges, faces = triangle_complex(with_face=True)
    B1 = build_boundary_matrix(nodes, edges)
    B2 = build_boundary_matrix(edges, faces)
    assert jnp.allclose(B1 @ B2, 0.0)


def test_annulus_boundary_of_boundary_is_zero():
    nodes, edges, faces = annulus_complex(n=18, with_faces=True)
    B1 = build_boundary_matrix(nodes, edges)
    B2 = build_boundary_matrix(edges, faces)
    assert jnp.allclose(B1 @ B2, 0.0)
