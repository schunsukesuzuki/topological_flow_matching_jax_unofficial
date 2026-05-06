import jax.numpy as jnp
from .simplicial_complex import annulus_complex, annulus_layout
from .boundary import build_boundary_matrix
from .hodge import hodge_laplacian


def _to_list(x):
    return jnp.asarray(x).tolist()


def _deterministic_coeffs(n: int):
    i = jnp.arange(n, dtype=jnp.float32)
    coeffs = (
        1.15 * jnp.sin(0.73 * (i + 1.0))
        + 0.85 * jnp.cos(1.37 * (i + 2.0))
        + 0.35 * jnp.sin(2.11 * (i + 0.5))
    )
    return coeffs


def _visual_scale(v, target: float = 0.78, clip_std: float = 1.8):
    """Make figure panels visually comparable without saturating extremes.

    The previous implementation normalized by max(|v|), which pushed almost all
    panels to very dark red/blue extremes and made the 5 examples difficult to
    distinguish. Here we center, scale by standard deviation, clip softly, and
    then map into a narrower visual range.
    """
    v = jnp.asarray(v, dtype=jnp.float32)
    mean = jnp.mean(v)
    std = jnp.std(v)

    # Constant / near-constant mode: keep it lightly tinted instead of fully saturated.
    def constant_case():
        sign = jnp.where(mean >= 0.0, 1.0, -1.0)
        return jnp.ones_like(v) * (0.45 * sign)

    def variable_case():
        z = (v - mean) / (std + 1e-6)
        z = jnp.clip(z / clip_std, -1.0, 1.0)
        return target * z

    return jnp.where(std < 1e-5, constant_case(), variable_case())


def _sample_from_spectrum(evals, U, kappa: float, heat: bool):
    coeffs = _deterministic_coeffs(evals.shape[0])
    if heat:
        coeffs = jnp.exp(-0.5 * kappa * evals) * coeffs
    signal = U @ coeffs
    return _visual_scale(signal)


def _mode(evals, U, which: str):
    if which == "zero":
        idx = int(jnp.argmin(jnp.abs(evals)))
    elif which == "low":
        positive = jnp.where(evals > 1e-5, evals, jnp.inf)
        idx = int(jnp.argmin(positive))
    elif which == "high":
        idx = int(jnp.argmax(evals))
    else:
        idx = 0

    v = jnp.asarray(U[:, idx], dtype=jnp.float32)
    # Stabilize sign so the visualization doesn't flip unpredictably.
    pivot = v[jnp.argmax(jnp.abs(v))]
    v = jnp.where(pivot < 0.0, -v, v)

    # The graph zero mode is constant on a connected graph; keep it lightly tinted.
    if which == "zero" and jnp.std(v) < 1e-5:
        return jnp.ones_like(v) * 0.45

    return _visual_scale(v, target=0.82, clip_std=1.6)


def build_figure1_payload(kappa: float = 2.0, n: int = 18):
    nodes, edges, faces = annulus_complex(n=n, with_faces=True)
    layout = annulus_layout(n=n)

    B1 = build_boundary_matrix(nodes, edges)
    B2 = build_boundary_matrix(edges, faces)
    L0 = B1 @ B1.T
    L1 = hodge_laplacian(B1, B2)

    evals0, U0 = jnp.linalg.eigh(L0)
    evals1, U1 = jnp.linalg.eigh(L1)

    labels = [
        "Normal sample",
        "Heat GP sample",
        "Zero eigenfunction",
        "Low-frequency eigenfunction",
        "High-frequency eigenfunction",
    ]

    graph_signals = [
        _sample_from_spectrum(evals0, U0, kappa, heat=False),
        _sample_from_spectrum(evals0, U0, kappa, heat=True),
        _mode(evals0, U0, "zero"),
        _mode(evals0, U0, "low"),
        _mode(evals0, U0, "high"),
    ]
    edge_signals = [
        _sample_from_spectrum(evals1, U1, kappa, heat=False),
        _sample_from_spectrum(evals1, U1, kappa, heat=True),
        _mode(evals1, U1, "zero"),
        _mode(evals1, U1, "low"),
        _mode(evals1, U1, "high"),
    ]

    return {
        "labels": labels,
        "nodes": layout,
        "edges": [list(e) for e in edges],
        "faces": [list(f) for f in faces],
        "graph_signals": [_to_list(s) for s in graph_signals],
        "edge_signals": [_to_list(s) for s in edge_signals],
        "graph_eigenvalues": _to_list(evals0),
        "edge_eigenvalues": _to_list(evals1),
    }
