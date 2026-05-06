import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Hodge spectral diagnostics: x, y, and x^T L x
# -----------------------------------------------------------------------------
#
# This file evaluates generated edge signals using compact Hodge spectral
# diagnostics.
#
# Notation:
#
#   x:
#       The original edge signal in the edge basis.
#
#       If the simplicial complex has n_edges edges, then:
#
#           x ∈ R^{n_edges}
#
#       Each component x_i is the scalar value assigned to edge i.
#
#   L:
#       The L1 Hodge Laplacian acting on edge signals:
#
#           L ∈ R^{n_edges × n_edges}
#
#       It measures how rough, inconsistent, or high-frequency an edge signal is
#       with respect to the topology / incidence structure of the complex.
#
#   U, evals:
#       The eigenvectors and eigenvalues of L:
#
#           L = U diag(λ) U^T
#
#       Here evals[i] = λ_i and U[:, i] is the corresponding Hodge eigenmode.
#
#   y:
#       The Hodge spectral coefficients of x:
#
#           y = U^T x
#
#       x and y represent the same signal, but in different coordinates.
#
#       x:
#           signal in the original edge basis
#
#       y:
#           signal decomposed into Hodge spectral modes
#
#   x^T L x:
#       The Hodge energy / Dirichlet-type energy of x:
#
#           x^T L x
#           = x^T U diag(λ) U^T x
#           = y^T diag(λ) y
#           = Σ_i λ_i y_i^2
#
#       This is what hodge_energy computes below.
#
# Interpretation of energy values:
#
#   high_frequency_energy:
#       Smaller is generally better when the target distribution is smooth.
#       It measures rough / noisy / locally oscillatory modes.
#
#   hodge_energy = x^T L x:
#       Smaller means the signal is smoother under the Hodge Laplacian.
#       This is the most direct energy-minimization diagnostic.
#
#   harmonic_energy:
#       Not necessarily smaller-is-better.
#       Harmonic modes have λ = 0, so they are not penalized by x^T L x.
#       On an annulus, this can represent circulation around the hole.
#       It should match the target μ1 distribution rather than be minimized
#       blindly.
#
#   low_frequency_energy:
#       Also not blindly smaller-is-better.
#       It captures smooth large-scale structural modes. Too much can indicate
#       overshoot, but too little may mean the generated distribution lost the
#       target structure.
#
#   l2_norm:
#       Measures total signal amplitude. Smaller is not always better; it should
#       be close to the reference μ1 norm. A trivial zero signal would have small
#       norm and small Hodge energy, but would not be a good generated sample.
#
# Therefore these metrics should be read as:
#
#   1. hodge_energy and high_frequency_energy should generally decrease,
#      because TFM is solving a topology-aware smoothing / energy-control
#      problem.
#
#   2. harmonic_energy, low_frequency_energy, and l2_norm should be compared
#      against the reference μ1 distribution, not simply minimized.
#
#   3. The best generated distribution is not the one with all energies equal
#      to zero. It is the one that suppresses rough modes while preserving the
#      target spectral profile.


METRIC_KEYS = (
    "harmonic_energy",
    "low_frequency_energy",
    "high_frequency_energy",
    "hodge_energy",
    "l2_norm",
)


def spectral_energy_metrics(x: jnp.ndarray, evals: jnp.ndarray, U: jnp.ndarray):
    """Compute compact spectral diagnostics for one edge signal.

    Args:
        x:
            One edge signal in the original edge basis.
            Shape: (n_edges,)

        evals:
            Eigenvalues of the L1 Hodge Laplacian.
            Shape: (n_edges,)

        U:
            Eigenvector matrix of the L1 Hodge Laplacian.
            Shape: (n_edges, n_edges)

    Returns:
        A vector of five diagnostics, ordered according to METRIC_KEYS:

            0. harmonic_energy
            1. low_frequency_energy
            2. high_frequency_energy
            3. hodge_energy = x^T L x
            4. l2_norm = ||x||^2

    The function computes these values in Hodge spectral coordinates. This makes
    it possible to distinguish harmonic, low-frequency, and high-frequency
    components of the same edge signal.
    """
    # Project the original edge signal x into the Hodge eigenbasis.
    #
    # If:
    #
    #     L = U diag(λ) U^T
    #
    # then:
    #
    #     y = U^T x
    #
    # y[i] is the coefficient of x along Hodge eigenmode i.
    y = U.T @ x

    # Squared spectral coefficients represent per-mode energy:
    #
    #     y_i^2
    #
    # These are later grouped into harmonic / low-frequency / high-frequency
    # energies.
    y2 = y**2

    # Harmonic modes are the zero-eigenvalue modes:
    #
    #     λ_i ≈ 0
    #
    # These lie in ker L. Since λ_i = 0, they do not contribute to:
    #
    #     x^T L x = Σ_i λ_i y_i^2
    #
    # On an annulus, these modes can represent circulation around the hole.
    zero_mask = evals < 1e-5

    # Positive eigenvalue modes are non-harmonic modes. jnp.where is called with
    # a fixed size so that the output shape is static and JAX-friendly.
    #
    # Entries beyond the number of true positives are filled with 0. Since this
    # code uses only the first few positive modes, this is acceptable as long as
    # the complex has enough positive eigenvalues.
    positive_idx = jnp.where(evals > 1e-5, size=evals.shape[0], fill_value=0)[0]

    # Define the first few positive-eigenvalue modes as low-frequency structural
    # modes. These are smooth non-harmonic modes.
    #
    # This cutoff is a compact diagnostic choice, not a universal mathematical
    # definition.
    low_idx = positive_idx[:6]

    # Define the upper half of the spectrum as high-frequency modes.
    #
    # Since jnp.linalg.eigh returns eigenvalues in ascending order, later modes
    # correspond to larger λ_i, hence rougher / more oscillatory components.
    n = evals.shape[0]
    high_start = n // 2
    high_idx = jnp.arange(high_start, n)

    # Energy in the harmonic subspace.
    #
    # This should generally match the target μ1 harmonic profile. It is not
    # penalized by x^T L x because the corresponding eigenvalues are zero.
    harmonic_energy = jnp.sum(jnp.where(zero_mask, y2, 0.0))

    # Energy in the first few positive-eigenvalue modes.
    #
    # These modes are smooth large-scale structural modes. They are often
    # important for matching the target distribution, so this metric should be
    # compared against reference μ1 rather than blindly minimized.
    low_frequency_energy = jnp.sum(y2[low_idx])

    # Energy in high-frequency modes.
    #
    # This is generally expected to be small for smooth target distributions.
    # The Hodge heat drift -κLx suppresses these modes strongly because their
    # eigenvalues λ_i are large.
    high_frequency_energy = jnp.sum(y2[high_idx])

    # Hodge energy:
    #
    #     x^T L x = Σ_i λ_i y_i^2
    #
    # This is the main roughness / Dirichlet-type energy. Large eigenvalue modes
    # are penalized more heavily. Harmonic modes have λ_i = 0 and therefore do
    # not contribute.
    #
    # This could also be computed as:
    #
    #     x.T @ L @ x
    #
    # but using evals and y is more direct once the spectral decomposition is
    # available.
    hodge_energy = jnp.sum(evals * y2)

    # Total signal amplitude in the original edge basis:
    #
    #     ||x||^2
    #
    # If U is orthonormal, this is equal to Σ_i y_i^2. It is computed directly
    # in x-space for clarity.
    #
    # Smaller is not always better: the generated norm should match the target
    # μ1 norm rather than collapse to zero.
    l2_norm = jnp.sum(x**2)

    # Return values in the fixed order defined by METRIC_KEYS.
    return jnp.asarray([
        harmonic_energy,
        low_frequency_energy,
        high_frequency_energy,
        hodge_energy,
        l2_norm,
    ])


def _batch_metric_summary(x_batch: jnp.ndarray, evals: jnp.ndarray, U: jnp.ndarray):
    """Return mean/std spectral diagnostics for a batch.

    Args:
        x_batch:
            Batch of edge signals.
            Shape: (batch_size, n_edges)

        evals, U:
            Hodge Laplacian spectral decomposition.

    Returns:
        A JSON-friendly dictionary:

            {
              metric_name: {
                "mean": ...,
                "std": ...
              },
              ...
            }

    This is used for distribution-level evaluation. Since flow matching maps a
    source distribution μ0 to a target distribution μ1, batch mean/std is more
    informative than a single generated sample versus a single reference sample.
    """
    # Apply the single-sample spectral diagnostics to every sample in the batch.
    #
    # values.shape = (batch_size, len(METRIC_KEYS))
    values = jax.vmap(lambda x: spectral_energy_metrics(x, evals, U))(x_batch)

    # Mean and standard deviation across samples, metric by metric.
    means = jnp.mean(values, axis=0)
    stds = jnp.std(values, axis=0)

    # Convert to regular Python floats so FastAPI can serialize the values into
    # JSON responses without JAX DeviceArray serialization issues.
    return {
        key: {
            "mean": float(means[i]),
            "std": float(stds[i]),
        }
        for i, key in enumerate(METRIC_KEYS)
    }


def compare_generated_to_reference(source, generated, reference, evals, U):
    """Return single-sample spectral diagnostics for backward compatibility.

    This function evaluates one source sample, one generated sample, and one
    reference sample.

    It is useful for UI panels that display one triplet:

        source μ0
        generated sample
        reference μ1 sample

    However, for I-TFM / OT-TFM, the preferred diagnostic is the batch-level
    comparison below, because the model targets a distribution rather than a
    particular reference sample.
    """
    def as_dict(vec):
        # Compute the five spectral metrics for one vector and map them to
        # human-readable names using METRIC_KEYS.
        vals = spectral_energy_metrics(vec, evals, U)
        return {key: float(vals[i]) for i, key in enumerate(METRIC_KEYS)}

    return {
        "source": as_dict(source),
        "generated": as_dict(generated),
        "reference": as_dict(reference),
    }


def compare_generated_batches_to_reference(sources, generated, references, evals, U):
    """Return distribution-level mean/std diagnostics.

    This is the preferred diagnostic for I-TFM / OT-TFM because these models
    target μ1 as a distribution, not a particular reference sample.

    Args:
        sources:
            Batch sampled from μ0.

        generated:
            Batch obtained by rolling source samples through the learned vector
            field.

        references:
            Batch sampled independently from μ1.

        evals, U:
            Hodge Laplacian spectral decomposition.

    Returns:
        A dictionary of batch summaries for source / generated / reference:

            {
              "source": ...,
              "generated": ...,
              "reference": ...
            }

    Interpretation:
        generated should be compared to reference. For hodge_energy and
        high_frequency_energy, lower is usually better if the reference is
        smooth. For harmonic_energy, low_frequency_energy, and l2_norm, closeness
        to the reference distribution is more important than minimizing the
        metric itself.
    """
    return {
        "source": _batch_metric_summary(sources, evals, U),
        "generated": _batch_metric_summary(generated, evals, U),
        "reference": _batch_metric_summary(references, evals, U),
    }
