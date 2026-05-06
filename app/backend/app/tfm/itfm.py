from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from .datasets import (
    sample_itfm_pair_batch,
    sample_mu0,
    sample_mu1_harmonic_target,
)
from .flax_model import BridgeMLP


@dataclass
class ITFMTrainState:
    """In-memory trained I-TFM state used by the FastAPI demo app.

    This state stores everything needed for distribution-level I-TFM generation:

        1. the trained Flax MLP parameters,
        2. feature / target normalization statistics,
        3. Hodge spectral data of the L1 Laplacian,
        4. rollout metadata such as κ and μ0 mode.

    The important modeling point is that the trained model represents:

        u_θ(t, x_t, κ)

    not a pair-conditioned control u(t, x_t, x_1). During generation, x_1 is
    unknown, so the learned vector field must depend only on time, current state,
    and the heat-drift strength κ.
    """

    params: dict
    model: BridgeMLP
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    target_mean: jnp.ndarray
    target_std: jnp.ndarray
    evals: jnp.ndarray
    U: jnp.ndarray
    L: jnp.ndarray
    kappa: float
    final_loss: float
    unnormalized_mse: float
    mu0_mode: str = "heat_gp"


def _normalize_batch(x: jnp.ndarray, eps: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Normalize a batch feature-wise.

    Returns:
        normalized_x:
            (x - mean) / std

        mean:
            feature-wise batch mean

        std:
            feature-wise batch standard deviation

    Very small standard deviations are replaced with 1.0 to avoid numerical
    explosion in nearly constant dimensions. This is used for both model inputs
    and control targets so that the MLP training problem is well-conditioned.
    """
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    std = jnp.where(std < eps, 1.0, std)
    return (x - mean) / std, mean, std


def _adam_init(params):
    """Initialize Adam first and second moment buffers.

    Flax parameters are PyTrees. jax.tree_util.tree_map creates zero arrays with
    exactly the same tree structure and tensor shapes as params.
    """
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros


def _adam_update(
    params,
    grads,
    m,
    v,
    step,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    """One manual Adam update step.

    This is the standard Adam update:

        m_t = β1 m_{t-1} + (1 - β1) g_t
        v_t = β2 v_{t-1} + (1 - β2) g_t^2

        m_hat = m_t / (1 - β1^t)
        v_hat = v_t / (1 - β2^t)

        θ_t = θ_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

    Optax would normally handle this, but the app keeps the optimizer minimal
    and self-contained.
    """
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1.0 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1.0 - beta2) * (g * g), v, grads)

    step_f = jnp.asarray(step, dtype=jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - beta1 ** step_f), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - beta2 ** step_f), v)

    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, m, v


def _tfm_path_from_eig(
    x0_batch,
    x1_batch,
    evals,
    U,
    kappa: float,
    t_batch,
    eps: float = 1e-8,
):
    """Closed-form TFM bridge path in the Hodge eigenbasis.

    This function computes x_t on the TFM bridge between paired samples x0 and
    x1. The computation is performed mode-by-mode in the eigenbasis of the
    Hodge Laplacian:

        L = U diag(λ) U^T

    In spectral coordinates, each eigenmode evolves independently.

    Shape convention:
        x0_batch, x1_batch:
            (batch_size, n_edges)

        evals:
            (n_edges,)

        U:
            (n_edges, n_edges)

        t_batch:
            (batch_size,)

    The resulting path is computed in spectral coordinates and converted back
    to signal coordinates.
    """
    # Move edge signals into the Hodge spectral basis.
    #
    # y0[b, i] is the coefficient of sample b on eigenmode i.
    y0 = x0_batch @ U
    y1 = x1_batch @ U

    # Each mode has heat-drift strength:
    #
    #     ω_i = κ λ_i
    #
    # where λ_i is the Hodge Laplacian eigenvalue.
    omega = kappa * evals

    # t_batch has shape:
    #
    #     (batch_size,)
    #
    # We need it to interact with spectral vectors of shape:
    #
    #     (n_edges,)
    #
    # Therefore t_batch[:, None] inserts a new axis and makes:
    #
    #     t.shape = (batch_size, 1)
    #
    # This allows broadcasting against omega[None, :] of shape:
    #
    #     (1, n_edges)
    #
    # The final broadcasted shape becomes:
    #
    #     (batch_size, n_edges)
    #
    # In other words, t varies by sample, while omega varies by eigenmode.
    t = t_batch[:, None]

    # Linear interpolation is the limiting case for zero eigenvalues
    # / harmonic modes where ω_i ≈ 0.
    linear = (1.0 - t) * y0 + t * y1

    # denom is mode-dependent only:
    #
    #     denom_i = sinh(ω_i)
    #
    # jnp.sinh(omega) has shape:
    #
    #     (n_edges,)
    #
    # [None, :] changes it to:
    #
    #     (1, n_edges)
    #
    # so it broadcasts across the batch dimension when divided into coeff0
    # and coeff1. This is the same broadcasting idea as omega[None, :].
    denom = jnp.sinh(omega)[None, :]

    # For nonzero modes, the TFM bridge path is:
    #
    #     y_t,i =
    #         sinh(ω_i(1 - t)) / sinh(ω_i) * y0_i
    #       + sinh(ω_i t)       / sinh(ω_i) * y1_i
    #
    # omega[None, :] has shape (1, n_edges), while t has shape
    # (batch_size, 1), so the product has shape (batch_size, n_edges).
    coeff0 = jnp.sinh(omega[None, :] * (1.0 - t)) / (denom + eps)
    coeff1 = jnp.sinh(omega[None, :] * t) / (denom + eps)
    sinh_path = coeff0 * y0 + coeff1 * y1

    # Zero eigenvalue modes correspond to the harmonic subspace. Since the heat
    # drift -κLx vanishes on ker L, the bridge should reduce to ordinary linear
    # interpolation. The mask has shape (1, n_edges) and is broadcast across the
    # batch dimension.
    use_linear = (jnp.abs(omega) < eps)[None, :]
    y_t = jnp.where(use_linear, linear, sinh_path)

    # Convert from spectral coordinates back to edge-signal coordinates.
    return y_t @ U.T


def _tfm_control_from_eig(
    x0_batch,
    x1_batch,
    evals,
    U,
    kappa: float,
    t_batch,
    eps: float = 1e-8,
):
    """Closed-form TFM bridge control in the Hodge eigenbasis.

    This computes the supervised target u_t used to train the neural vector
    field u_θ(t, x_t, κ).

    The reference drift is the Hodge heat drift:

        dx/dt = -κ L x

    The corresponding heat semigroup is:

        exp(-κ t L)

    In the Hodge eigenbasis L u_i = λ_i u_i, this semigroup acts diagonally:

        y_i(t) = exp(-κ λ_i t) y_i(0)

    Code-level correspondence:
        omega = κ * evals
            ω_i = κ λ_i

        exp_full = exp(-omega)
            exp(-κ λ_i), i.e. the full interval [0, 1] heat semigroup

        exp_remaining = exp(-omega * (1 - t))
            exp(-κ λ_i (1 - t)), i.e. the remaining interval [t, 1]
            heat semigroup

        residual = y1 - exp_full * y0
            the part of the endpoint y1 not reached by uncontrolled heat flow

    Thus the control target explicitly accounts for the amount that the heat
    semigroup would naturally decay from y0 before adding the learned control.
    """
    # Move edge signals into Hodge spectral coordinates.
    y0 = x0_batch @ U
    y1 = x1_batch @ U

    # Per-mode heat-drift strength:
    #
    #     ω_i = κ λ_i
    omega = kappa * evals

    # Insert a singleton spectral-broadcast axis for time:
    #
    #     t_batch: (batch_size,)
    #     t:       (batch_size, 1)
    #
    # This broadcasts against omega[None, :] with shape (1, n_edges).
    t = t_batch[:, None]

    # Zero-eigenvalue / harmonic modes have no heat drift, so the control is
    # the ordinary CFM-like constant velocity in spectral coordinates.
    linear_u = y1 - y0

    # Full-interval heat semigroup:
    #
    #     exp_full_i = exp(-ω_i) = exp(-κ λ_i)
    #
    # This represents how much mode i of y0 would remain at time 1 if the
    # system evolved only under dx/dt = -κLx with no learned control.
    #
    # Shape:
    #     exp_full: (1, n_edges)
    #
    # The [None, :] is used so exp_full broadcasts across the batch dimension.
    exp_full = jnp.exp(-omega)[None, :]

    # Remaining-interval heat semigroup:
    #
    #     exp_remaining_i(t) = exp(-ω_i (1 - t))
    #                         = exp(-κ λ_i (1 - t))
    #
    # This is the heat semigroup from the current time t to terminal time 1.
    #
    # Shape:
    #     omega[None, :]       -> (1, n_edges)
    #     (1.0 - t)            -> (batch_size, 1)
    #     exp_remaining        -> (batch_size, n_edges)
    #
    # Again, None is used to make the batch axis and spectral-mode axis
    # explicit for broadcasting.
    exp_remaining = jnp.exp(-omega[None, :] * (1.0 - t))

    # Control coefficient for each batch/time and eigenmode:
    #
    #     coeff_i(t) =
    #         2ω_i exp(-ω_i(1 - t)) / (1 - exp(-2ω_i))
    #
    # The denominator also comes from the heat semigroup / controllability
    # factor over the unit interval. The exp(-2ω_i) term corresponds to the
    # squared heat decay over [0, 1].
    numerator = 2.0 * omega[None, :] * exp_remaining
    denominator = 1.0 - jnp.exp(-2.0 * omega)[None, :]
    coeff = numerator / (denominator + eps)

    # Natural heat-flow endpoint:
    #
    #     exp_full_i * y0_i = exp(-κλ_i) y0_i
    #
    # residual is therefore the gap between the actual target y1_i and what the
    # heat semigroup would reach without control:
    #
    #     residual_i = y1_i - exp(-κλ_i) y0_i
    #
    # This is the most explicit place in the code where the phrase
    # "accounting for natural decay under the heat semigroup" appears.
    residual = y1 - exp_full * y0

    # Nonzero-mode TFM control:
    #
    #     u_i(t) = coeff_i(t) * residual_i
    spectral_u = coeff * residual

    # For harmonic modes, use the linear control because the heat drift is zero
    # and the nonzero-mode formula becomes numerically singular as ω -> 0.
    use_linear = (jnp.abs(omega) < eps)[None, :]
    u_y = jnp.where(use_linear, linear_u, spectral_u)

    # Return to edge-signal coordinates.
    return u_y @ U.T


def make_itfm_features(t_batch: jnp.ndarray, x_t_batch: jnp.ndarray, kappa: float):
    """Build MLP features for distribution-level I-TFM.

    Feature structure:

        [t, κ, x_t]

    If the edge-signal dimension is d, then the feature dimension is d + 2.

    Note that x1 is intentionally not included. At generation time, x1 is
    unknown; the model must learn a distribution-level vector field:

        u_θ(t, x_t, κ)
    """
    t_col = t_batch[:, None]
    kappa_col = jnp.ones_like(t_col) * jnp.asarray(kappa, dtype=x_t_batch.dtype)
    return jnp.concatenate([t_col, kappa_col, x_t_batch], axis=1)


def build_itfm_training_set(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    kappa: float,
    *,
    mu0_mode: str = "heat_gp",
):
    """Construct supervised training data for I-TFM.

    I-TFM uses independent coupling:

        (X0, X1) ~ μ0 ⊗ μ1

    For each sampled pair and random time t:
        1. compute the closed-form TFM bridge point x_t,
        2. compute the closed-form TFM control target u_t,
        3. create the feature vector [t, κ, x_t].

    Returns:
        features:
            input to the MLP

        target_u:
            supervised target for the MLP
    """
    key_pair, key_t = jax.random.split(key)

    # Independent coupling: x0 and x1 are sampled independently.
    x0, x1 = sample_itfm_pair_batch(
        key_pair,
        n_samples,
        evals,
        U,
        mu0_mode=mu0_mode,
    )

    # Avoid exact endpoints because bridge controls can be numerically more
    # extreme near t = 0 or t = 1.
    t = jax.random.uniform(key_t, (n_samples,), minval=0.02, maxval=0.98)

    # Closed-form bridge point and closed-form bridge control target.
    x_t = _tfm_path_from_eig(x0, x1, evals, U, kappa, t)
    target_u = _tfm_control_from_eig(x0, x1, evals, U, kappa, t)

    # Neural model input: [t, κ, x_t].
    features = make_itfm_features(t, x_t, kappa)

    return features, target_u


def train_itfm_vector_field(
    key,
    L: jnp.ndarray,
    kappa: float,
    *,
    n_samples: int = 2048,
    n_steps: int = 2000,
    learning_rate: float = 2e-3,
    hidden_dims: Sequence[int] = (256, 256, 128),
    mu0_mode: str = "heat_gp",
):
    """Train distribution-level I-TFM vector field u_theta(t, x_t, kappa).

    Training pipeline:
        1. eigendecompose the Hodge Laplacian,
        2. sample independent pairs (X0, X1) ~ μ0 ⊗ μ1,
        3. compute closed-form TFM bridge points and controls,
        4. train a Flax MLP to predict the control from [t, κ, x_t],
        5. store the trained model and normalization statistics.

    This is simulation-free training: the targets are generated from the
    closed-form bridge formulas, not by numerically simulating SDE/ODE paths.
    """
    # Hodge spectral decomposition:
    #
    #     L = U diag(evals) U^T
    #
    # This is needed for the closed-form path and control formulas.
    evals, U = jnp.linalg.eigh(L)

    features, targets = build_itfm_training_set(
        key,
        n_samples,
        evals,
        U,
        kappa,
        mu0_mode=mu0_mode,
    )

    # Normalize inputs and targets for stable MLP training.
    features_n, feature_mean, feature_std = _normalize_batch(features)
    targets_n, target_mean, target_std = _normalize_batch(targets)

    output_dim = int(targets.shape[1])
    model = BridgeMLP(hidden_dims=hidden_dims, output_dim=output_dim)
    params = model.init(jax.random.fold_in(key, 123), features_n[0])
    m, v = _adam_init(params)

    def predict_norm(params, batch_x):
        # Apply the single-sample MLP to a batch using vmap.
        return jax.vmap(lambda z: model.apply(params, z))(batch_x)

    def loss_fn(params):
        # MSE in normalized target space.
        pred = predict_norm(params, features_n)
        return jnp.mean((pred - targets_n) ** 2)

    def step_fn(carry, step_idx):
        params, m, v = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, m, v = _adam_update(
            params,
            grads,
            m,
            v,
            step_idx + 1,
            learning_rate=learning_rate,
        )
        return (params, m, v), loss

    # JAX-friendly loop over optimizer steps.
    (params, m, v), loss_history = jax.lax.scan(
        step_fn,
        (params, m, v),
        jnp.arange(n_steps),
    )

    # Report MSE in original signal/control scale for interpretability.
    pred_n = predict_norm(params, features_n)
    pred = pred_n * target_std + target_mean
    unnormalized_mse = jnp.mean((pred - targets) ** 2)

    return ITFMTrainState(
        params=params,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        evals=evals,
        U=U,
        L=L,
        kappa=float(kappa),
        final_loss=float(loss_history[-1]),
        unnormalized_mse=float(unnormalized_mse),
        mu0_mode=mu0_mode,
    )


def predict_itfm_u(
    state: ITFMTrainState,
    t: float,
    x_t: jnp.ndarray,
    *,
    control_scale: float = 0.92,
    clip_quantile: float = 3.0,
):
    """Predict learned I-TFM control u_theta(t, x_t, kappa).

    Steps:
        1. build feature [t, κ, x_t],
        2. normalize with training feature statistics,
        3. apply the Flax MLP,
        4. unnormalize the predicted control,
        5. clip the control mildly for rollout stability,
        6. apply control_scale.

    The clipping and control_scale are practical stabilizers for Euler rollout.
    They help prevent the learned control from overshooting in regions slightly
    outside the training distribution.
    """
    t_batch = jnp.asarray([t], dtype=x_t.dtype)
    x_batch = x_t[None, :]

    features = make_itfm_features(t_batch, x_batch, state.kappa)
    features_n = (features - state.feature_mean) / state.feature_std

    pred_n = state.model.apply(state.params, features_n[0])
    pred = pred_n * state.target_std + state.target_mean

    # Mild component-wise clipping based on the training target scale.
    clip_value = clip_quantile * state.target_std
    pred = jnp.clip(pred, -clip_value, clip_value)

    return control_scale * pred


def generate_itfm_samples(
    key,
    state: ITFMTrainState,
    *,
    n_eval: int = 32,
    n_steps: int = 160,
    control_scale: float = 0.92,
):
    """Generate a batch of I-TFM samples by Euler rollout.

    Source samples:
        x0 ~ μ0

    Reference samples:
        x1 ~ μ1
        These are used only for evaluation, not for generation.

    Generation dynamics:

        dx/dt = -κ L x + u_θ(t, x)

    The term -κLx is the Hodge heat drift. It suppresses high-frequency
    roughness and makes the generation dynamics topology-aware.
    """
    key0, key_ref = jax.random.split(key)

    sources = sample_mu0(key0, n_eval, state.evals, state.U, mode=state.mu0_mode)
    references = sample_mu1_harmonic_target(key_ref, n_eval, state.evals, state.U)

    dt = 1.0 / float(n_steps)

    def rollout_one(x0):
        def body(i, x_cur):
            t = jnp.asarray(i, dtype=x_cur.dtype) * dt

            # Learned control u_θ(t, x).
            u = predict_itfm_u(state, t, x_cur, control_scale=control_scale)

            # Hodge heat drift:
            #
            #     -κ L x
            #
            # This is the generator of the heat semigroup exp(-κtL).
            drift = -state.kappa * (state.L @ x_cur)

            # Explicit Euler step:
            #
            #     x_{n+1} = x_n + dt * (-κLx_n + u_θ(t_n, x_n))
            return x_cur + dt * (drift + u)

        return jax.lax.fori_loop(0, n_steps, body, x0)

    generated = jax.vmap(rollout_one)(sources)

    return sources, generated, references


def generate_itfm_sample(
    key,
    state: ITFMTrainState,
    *,
    n_steps: int = 160,
    control_scale: float = 0.92,
):
    """Generate one display sample.

    This is a thin helper around generate_itfm_samples(..., n_eval=1). It is
    useful for UI panels that show one source / generated / reference triplet.
    """
    sources, generated, references = generate_itfm_samples(
        key,
        state,
        n_eval=1,
        n_steps=n_steps,
        control_scale=control_scale,
    )
    return sources[0], generated[0], references[0]
