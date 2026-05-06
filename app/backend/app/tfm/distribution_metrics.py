import jax
import jax.numpy as jnp


def _pairwise_sq_dists(x: jnp.ndarray, y: jnp.ndarray):
    """Pairwise squared Euclidean distances between two batches."""
    diff = x[:, None, :] - y[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def mmd_rbf(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    bandwidth: float | None = None,
):
    """Compute biased RBF-kernel MMD^2 between two empirical distributions.

    This is a compact diagnostic distance:

        MMD^2(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]

    If bandwidth is not provided, a median-distance heuristic is used over the
    cross-distance matrix.
    """
    d_xx = _pairwise_sq_dists(x, x)
    d_yy = _pairwise_sq_dists(y, y)
    d_xy = _pairwise_sq_dists(x, y)

    if bandwidth is None:
        # Stable median heuristic. The +1e-6 prevents a zero bandwidth when
        # samples are nearly identical.
        bandwidth = jnp.sqrt(jnp.median(d_xy) + 1e-6)

    gamma = 1.0 / (2.0 * bandwidth * bandwidth + 1e-8)

    k_xx = jnp.exp(-gamma * d_xx)
    k_yy = jnp.exp(-gamma * d_yy)
    k_xy = jnp.exp(-gamma * d_xy)

    mmd2 = jnp.mean(k_xx) + jnp.mean(k_yy) - 2.0 * jnp.mean(k_xy)
    return float(jnp.maximum(mmd2, 0.0))


def _wasserstein_1d_sorted(a: jnp.ndarray, b: jnp.ndarray):
    """Empirical 1D W2-like distance using sorted samples.

    For equal-size empirical samples, the optimal 1D matching pairs sorted
    order statistics. We return mean squared gap as a stable scalar.
    """
    a_sorted = jnp.sort(a)
    b_sorted = jnp.sort(b)
    return jnp.mean((a_sorted - b_sorted) ** 2)


def spectral_sliced_wasserstein(
    x: jnp.ndarray,
    y: jnp.ndarray,
    U: jnp.ndarray,
    *,
    n_projections: int = 64,
    seed: int = 0,
):
    """Approximate distribution distance in Hodge spectral coordinates.

    Steps:
        1. Project signals into the Hodge eigenbasis.
        2. Draw random 1D projection directions in spectral space.
        3. Compute 1D sorted empirical transport cost along each direction.
        4. Average over projections.

    This is not meant to be a formal benchmark metric. It is a useful app-level
    diagnostic for comparing generated and reference distributions while still
    respecting the Hodge spectral representation.
    """
    x_spec = x @ U
    y_spec = y @ U

    key = jax.random.PRNGKey(seed)
    dirs = jax.random.normal(key, (n_projections, x_spec.shape[1]))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

    x_proj = x_spec @ dirs.T
    y_proj = y_spec @ dirs.T

    costs = jax.vmap(lambda a, b: _wasserstein_1d_sorted(a, b), in_axes=(1, 1))(x_proj, y_proj)
    return float(jnp.mean(costs))


def spectral_coefficient_wasserstein(
    x: jnp.ndarray,
    y: jnp.ndarray,
    U: jnp.ndarray,
    *,
    n_modes: int = 12,
):
    """Mean 1D sorted transport cost over the first Hodge spectral modes.

    This is more interpretable than random sliced projections because it compares
    mode-by-mode coefficient distributions.
    """
    x_spec = x @ U
    y_spec = y @ U
    n = min(n_modes, x_spec.shape[1])

    costs = []
    for i in range(n):
        costs.append(_wasserstein_1d_sorted(x_spec[:, i], y_spec[:, i]))

    return float(jnp.mean(jnp.asarray(costs)))


def distribution_distance_summary(
    generated: jnp.ndarray,
    reference: jnp.ndarray,
    U: jnp.ndarray,
    *,
    seed: int = 0,
):
    """Return compact generated-vs-reference distribution distances."""
    return {
        "mmd_rbf": mmd_rbf(generated, reference),
        "spectral_sliced_wasserstein": spectral_sliced_wasserstein(
            generated,
            reference,
            U,
            n_projections=64,
            seed=seed,
        ),
        "spectral_mode_wasserstein": spectral_coefficient_wasserstein(
            generated,
            reference,
            U,
            n_modes=12,
        ),
    }
