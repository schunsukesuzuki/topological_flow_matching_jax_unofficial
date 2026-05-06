import jax
import jax.numpy as jnp


def _batch_to_signal(y_batch: jnp.ndarray, U: jnp.ndarray):
    """Convert batched spectral coefficients y into signal coordinates x.

    For one sample, x = U @ y. For row-batched data, this becomes:

        x_batch = y_batch @ U.T
    """
    return y_batch @ U.T


def sample_mu0_standard_noise(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
):
    """Sample the topology-unaware source distribution μ0.

    This is a standard Gaussian in Hodge spectral coordinates. Since U is
    orthonormal, this is equivalent to standard Gaussian edge noise in signal
    coordinates, but writing it in spectral coordinates makes the contrast with
    heat GP initialization explicit.
    """
    n_edges = evals.shape[0]
    y = jax.random.normal(key, (n_samples, n_edges))
    return _batch_to_signal(y, U)


def sample_mu0_heat_noise(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    heat_kappa: float = 0.35,
):
    """Sample topology-aware μ0 as heat-smoothed Gaussian edge noise.

    In spectral coordinates, high-frequency modes are damped by:

        exp(-0.5 * heat_kappa * lambda_i)

    This produces a topology-aware Gaussian process on the edge space.
    """
    n_edges = evals.shape[0]
    coeff_scale = jnp.exp(-0.5 * heat_kappa * evals)
    y = jax.random.normal(key, (n_samples, n_edges)) * coeff_scale[None, :]
    return _batch_to_signal(y, U)


def sample_mu0(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    mode: str = "heat_gp",
    heat_kappa: float = 0.35,
):
    """Sample μ0 for initial-distribution ablations.

    Modes:
        standard:
            topology-unaware standard Gaussian noise.

        heat_gp:
            topology-aware heat Gaussian process.
    """
    if mode == "standard":
        return sample_mu0_standard_noise(key, n_samples, evals, U)
    if mode == "heat_gp":
        return sample_mu0_heat_noise(key, n_samples, evals, U, heat_kappa=heat_kappa)
    raise ValueError(f"Unknown mu0 mode: {mode}")


def sample_mu1_harmonic_target(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    low_modes: int = 6,
    noise_scale: float = 0.05,
    harmonic_mean: float = 1.15,
    harmonic_std: float = 0.18,
):
    """Sample target distribution μ1.

    μ1 is synthetic but topology-aware:
      - harmonic / zero-eigenvalue circulation,
      - several low-frequency structural modes,
      - small heat-smoothed noise.
    """
    n_edges = evals.shape[0]
    key_h, key_l, key_n = jax.random.split(key, 3)

    y = jnp.zeros((n_samples, n_edges), dtype=jnp.float32)

    zero_indices = jnp.where(evals < 1e-5, size=n_edges, fill_value=0)[0]
    harmonic_idx = zero_indices[0]
    harmonic_amp = harmonic_mean + harmonic_std * jax.random.normal(key_h, (n_samples,))
    y = y.at[:, harmonic_idx].set(harmonic_amp)

    positive_indices = jnp.where(evals > 1e-5, size=n_edges, fill_value=0)[0]
    low_idx = positive_indices[:low_modes]
    low_coeff = 0.30 * jax.random.normal(key_l, (n_samples, low_modes))
    y = y.at[:, low_idx].set(low_coeff)

    noise = (
        noise_scale
        * jax.random.normal(key_n, (n_samples, n_edges))
        * jnp.exp(-0.6 * evals)[None, :]
    )
    y = y + noise
    return _batch_to_signal(y, U)


def sample_itfm_pair_batch(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    mu0_mode: str = "heat_gp",
):
    """Sample independent I-TFM pairs.

    I-TFM uses the independent coupling:

        (X0, X1) ~ μ0 ⊗ μ1
    """
    key0, key1 = jax.random.split(key)
    x0 = sample_mu0(key0, n_samples, evals, U, mode=mu0_mode)
    x1 = sample_mu1_harmonic_target(key1, n_samples, evals, U)
    return x0, x1
