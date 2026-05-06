from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from .datasets import sample_mu0, sample_mu1_harmonic_target
from .flax_model import BridgeMLP


# -----------------------------------------------------------------------------
# OT-TFM overview
# -----------------------------------------------------------------------------
#
# This file implements OT-TFM:
#
#     Optimal-Transport-coupled Topological Flow Matching
#
# I-TFM and OT-TFM share the same topology-aware generation dynamics:
#
#     dx/dt = -κ L x + u_θ(t, x)
#
# where L is the L1 Hodge Laplacian and -κLx is the Hodge heat drift.
# The difference is the way training pairs (x0, x1) are selected:
#
#     I-TFM:
#         (X0, X1) ~ μ0 ⊗ μ1
#         source and target samples are paired independently.
#
#     OT-TFM:
#         x0_pool ~ μ0, x1_pool ~ μ1
#         C_ij = TFMTransportCost(x0_i, x1_j)
#         P = Sinkhorn(C)
#         (x0, x1) pairs are sampled from the soft OT plan P.
#
# Thus OT-TFM adds topology-aware coupling on top of topology-aware dynamics.
#
# -----------------------------------------------------------------------------
# Sinkhorn algorithm and sinkhorn_epsilon
# -----------------------------------------------------------------------------
#
# Given a cost matrix C_ij between source samples and target samples, optimal
# transport searches for a nonnegative transport plan P_ij with prescribed row
# and column sums:
#
#     Σ_j P_ij = a_i
#     Σ_i P_ij = b_j
#
# With uniform minibatches:
#
#     a_i = 1 / n_source
#     b_j = 1 / n_target
#
# Classical OT solves:
#
#     min_P Σ_ij P_ij C_ij
#
# Entropic OT solves:
#
#     min_P Σ_ij P_ij C_ij - ε H(P)
#
# where:
#
#     H(P) = -Σ_ij P_ij log P_ij
#
# The entropy term makes the plan soft and numerically stable. The coefficient
# ε is sinkhorn_epsilon. It behaves like a temperature in:
#
#     K_ij = exp(-C_ij / ε)
#
# Small sinkhorn_epsilon:
#     - sharper / more hard-matching-like plan
#     - low-cost pairs receive much more mass
#     - less pair diversity, potentially less stable
#
# Large sinkhorn_epsilon:
#     - more diffuse plan
#     - closer to independent coupling
#     - stable, but weaker OT effect
#
# Sinkhorn repeatedly rescales rows and columns of K:
#
#     P = diag(u) K diag(v)
#     u = a / (K v)
#     v = b / (K^T u)
#
# until the row and column marginals are approximately satisfied.
#
# In this app:
#     OT-CFM uses Euclidean cost: ||x0_i - x1_j||^2
#     OT-TFM uses topology-aware TFM transport cost.
#


@dataclass
class OTTFMTrainState:
    """In-memory trained OT-TFM state used by the FastAPI demo app.

    This stores the trained Flax MLP, normalization statistics, the Hodge
    spectral decomposition, rollout parameters, and OT-coupling diagnostics.

    Compared with ITFMTrainState, this state additionally stores:
        mean_pair_cost:
            average TFM transport cost of sampled OT training pairs.

        sinkhorn_epsilon:
            entropy / temperature parameter used in Sinkhorn coupling.

        ot_batch_size:
            source / target minibatch size used for each OT coupling problem.
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
    mean_pair_cost: float
    sinkhorn_epsilon: float
    ot_batch_size: int
    mu0_mode: str = "heat_gp"


def _normalize_batch(x: jnp.ndarray, eps: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Normalize a batch feature-wise.

    This is the same stabilization step as in itfm.py. It returns normalized
    data plus mean/std so inference can use the same scaling.
    """
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    std = jnp.where(std < eps, 1.0, std)
    return (x - mean) / std, mean, std


def _adam_init(params):
    """Initialize Adam first and second moment buffers as zero PyTrees."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros


def _adam_update(params, grads, m, v, step, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    """One manual Adam update step.

    This avoids adding an Optax dependency to the demo app. The implementation
    is the standard Adam update with bias correction.
    """
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1.0 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1.0 - beta2) * (g * g), v, grads)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda m_: m_ / (1.0 - beta1 ** step_f), m)
    v_hat = jax.tree_util.tree_map(lambda v_: v_ / (1.0 - beta2 ** step_f), v)
    params = jax.tree_util.tree_map(lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, m, v


def _tfm_path_from_eig(x0_batch, x1_batch, evals, U, kappa: float, t_batch, eps: float = 1e-8):
    """Closed-form TFM bridge path in the Hodge eigenbasis.

    Same mathematical role as in itfm.py.

    Broadcasting convention:
        t_batch[:, None] changes (batch,) to (batch, 1).
        omega[None, :] changes (n_modes,) to (1, n_modes).

    This allows computations to broadcast to (batch, n_modes), where t varies
    by sample and omega varies by Hodge spectral mode.
    """
    y0 = x0_batch @ U
    y1 = x1_batch @ U
    omega = kappa * evals
    t = t_batch[:, None]
    linear = (1.0 - t) * y0 + t * y1
    denom = jnp.sinh(omega)[None, :]
    coeff0 = jnp.sinh(omega[None, :] * (1.0 - t)) / (denom + eps)
    coeff1 = jnp.sinh(omega[None, :] * t) / (denom + eps)
    sinh_path = coeff0 * y0 + coeff1 * y1
    use_linear = (jnp.abs(omega) < eps)[None, :]
    return jnp.where(use_linear, linear, sinh_path) @ U.T


def _tfm_control_from_eig(x0_batch, x1_batch, evals, U, kappa: float, t_batch, eps: float = 1e-8):
    """Closed-form TFM bridge control in the Hodge eigenbasis.

    Heat semigroup correspondence:
        omega = κ * evals

        exp_full = exp(-omega)
            full-interval heat semigroup exp(-κL) in spectral coordinates.

        exp_remaining = exp(-omega * (1 - t))
            remaining-time heat semigroup exp(-κ(1-t)L).

        residual = y1 - exp_full * y0
            target endpoint minus the endpoint reached by uncontrolled heat flow.
    """
    y0 = x0_batch @ U
    y1 = x1_batch @ U
    omega = kappa * evals
    t = t_batch[:, None]
    linear_u = y1 - y0
    exp_full = jnp.exp(-omega)[None, :]
    exp_remaining = jnp.exp(-omega[None, :] * (1.0 - t))
    coeff = (2.0 * omega[None, :] * exp_remaining) / (1.0 - jnp.exp(-2.0 * omega)[None, :] + eps)
    residual = y1 - exp_full * y0
    spectral_u = coeff * residual
    use_linear = (jnp.abs(omega) < eps)[None, :]
    return jnp.where(use_linear, linear_u, spectral_u) @ U.T


def _tfm_cost_matrix_from_eig(x0_pool, x1_pool, evals, U, kappa: float, eps: float = 1e-8):
    """Compute pairwise TFM transport costs for OT-TFM coupling.

    This is the key OT-TFM-specific cost constructor.

    Input shapes:
        x0_pool: (n_source, n_edges)
        x1_pool: (n_target, n_edges)

    Output:
        cost: (n_source, n_target)

    The cost is evaluated mode-by-mode in the Hodge spectral basis and accounts
    for the heat semigroup induced by -κL.
    """
    y0 = x0_pool @ U
    y1 = x1_pool @ U
    omega = kappa * evals

    # Full-interval heat semigroup exp(-κλ_i) for each mode.
    exp_full = jnp.exp(-omega)

    # Nonzero-mode TFM transport-cost coefficient.
    coeff = 2.0 * omega / (1.0 - jnp.exp(-2.0 * omega) + eps)

    # Harmonic / zero-mode fallback: ordinary squared difference.
    # Broadcasting creates shape (n_source, n_target, n_modes).
    linear_cost = (y1[None, :, :] - y0[:, None, :]) ** 2

    # Heat-aware residual for all source-target-mode triples:
    #     y1_j - exp(-κλ) y0_i
    residual = y1[None, :, :] - exp_full[None, None, :] * y0[:, None, :]
    spectral_cost = coeff[None, None, :] * residual**2

    # Use linear cost on harmonic modes and TFM spectral cost otherwise.
    per_mode = jnp.where((jnp.abs(omega) < eps)[None, None, :], linear_cost, spectral_cost)
    return jnp.sum(per_mode, axis=-1)


def _sinkhorn_plan(cost: jnp.ndarray, epsilon: float = 0.75, n_iter: int = 80):
    """Compute an entropic Sinkhorn transport plan.

    Args:
        cost:
            Pairwise cost matrix C_ij.

        epsilon:
            Entropic regularization strength / temperature.

            small epsilon -> sharp plan, strong low-cost preference
            large epsilon -> diffuse plan, closer to independent coupling

        n_iter:
            Number of row/column scaling iterations.

    Returns:
        P:
            Soft transport plan with approximately uniform row/column marginals.

    Mathematical structure:
        K_ij = exp(-C_ij / epsilon)
        P = diag(u) K diag(v)
    """
    n, m = cost.shape

    # Shift cost for numerical stability before exponentiation.
    c = cost - jnp.min(cost)

    # Entropic OT Gibbs kernel. epsilon controls softness / temperature.
    K = jnp.exp(-c / epsilon) + 1e-8

    # Uniform empirical marginals.
    a = jnp.ones((n,), dtype=cost.dtype) / n
    b = jnp.ones((m,), dtype=cost.dtype) / m

    # Row and column scaling vectors.
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)

    def body(_, carry):
        u, v = carry

        # If P = diag(u) K diag(v), row sums are u ⊙ (K v).
        # To enforce row sums a, set u = a / (K v).
        u = a / (K @ v + 1e-8)

        # Column sums are v ⊙ (K^T u).
        # To enforce column sums b, set v = b / (K^T u).
        v = b / (K.T @ u + 1e-8)
        return u, v

    u, v = jax.lax.fori_loop(0, n_iter, body, (u, v))

    # Final plan: P = diag(u) K diag(v).
    P = (u[:, None] * K) * v[None, :]
    return P / (jnp.sum(P) + 1e-8)


def make_ottfm_features(t_batch: jnp.ndarray, x_t_batch: jnp.ndarray, kappa: float):
    """Build MLP features [t, κ, x_t].

    Same as I-TFM. x1 is not included because generation is distribution-level;
    target samples are unknown at inference time.
    """
    t_col = t_batch[:, None]
    kappa_col = jnp.ones_like(t_col) * jnp.asarray(kappa, dtype=x_t_batch.dtype)
    return jnp.concatenate([t_col, kappa_col, x_t_batch], axis=1)


def _sample_ot_pairs_one_batch(key, evals: jnp.ndarray, U: jnp.ndarray, kappa: float, *, ot_batch_size: int, sinkhorn_epsilon: float, mu0_mode: str = "heat_gp"):
    """Sample one minibatch of OT-coupled training pairs.

    Steps:
        1. sample source pool x0_pool ~ μ0,
        2. sample target pool x1_pool ~ μ1,
        3. compute TFM transport-cost matrix,
        4. compute Sinkhorn soft transport plan,
        5. sample source-target pairs from the flattened plan.
    """
    key0, key1, key_pair = jax.random.split(key, 3)
    x0_pool = sample_mu0(key0, ot_batch_size, evals, U, mode=mu0_mode)
    x1_pool = sample_mu1_harmonic_target(key1, ot_batch_size, evals, U)

    cost = _tfm_cost_matrix_from_eig(x0_pool, x1_pool, evals, U, kappa)
    plan = _sinkhorn_plan(cost, epsilon=sinkhorn_epsilon)
    probs = jnp.reshape(plan, (-1,))

    # Sample flat indices from P_ij treated as a categorical distribution.
    flat_idx = jax.random.choice(key_pair, probs.shape[0], shape=(ot_batch_size,), replace=True, p=probs)
    row = flat_idx // ot_batch_size
    col = flat_idx % ot_batch_size
    return x0_pool[row], x1_pool[col], cost[row, col]


def build_ottfm_training_set(key, n_samples: int, evals: jnp.ndarray, U: jnp.ndarray, kappa: float, *, ot_batch_size: int = 64, sinkhorn_epsilon: float = 0.75, mu0_mode: str = "heat_gp"):
    """Build the supervised OT-TFM training set.

    The pair selection is OT-coupled. After pairs are selected, training target
    construction is the same as I-TFM:
        - sample t,
        - compute closed-form TFM bridge path x_t,
        - compute closed-form TFM control target u_t,
        - create feature [t, κ, x_t].
    """
    n_batches = int((n_samples + ot_batch_size - 1) // ot_batch_size)
    x0_chunks, x1_chunks, cost_chunks = [], [], []

    for i in range(n_batches):
        x0_b, x1_b, c_b = _sample_ot_pairs_one_batch(
            jax.random.fold_in(key, i),
            evals,
            U,
            kappa,
            ot_batch_size=ot_batch_size,
            sinkhorn_epsilon=sinkhorn_epsilon,
            mu0_mode=mu0_mode,
        )
        x0_chunks.append(x0_b)
        x1_chunks.append(x1_b)
        cost_chunks.append(c_b)

    x0 = jnp.concatenate(x0_chunks, axis=0)[:n_samples]
    x1 = jnp.concatenate(x1_chunks, axis=0)[:n_samples]
    pair_cost = jnp.concatenate(cost_chunks, axis=0)[:n_samples]

    t = jax.random.uniform(jax.random.fold_in(key, 9999), (n_samples,), minval=0.02, maxval=0.98)
    x_t = _tfm_path_from_eig(x0, x1, evals, U, kappa, t)
    target_u = _tfm_control_from_eig(x0, x1, evals, U, kappa, t)
    features = make_ottfm_features(t, x_t, kappa)
    return features, target_u, pair_cost


def train_ottfm_vector_field(
    key,
    L: jnp.ndarray,
    kappa: float,
    *,
    n_samples: int = 2048,
    n_steps: int = 2000,
    learning_rate: float = 2e-3,
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
    hidden_dims: Sequence[int] = (256, 256, 128),
    mu0_mode: str = "heat_gp",
):
    """Train the OT-TFM neural control field u_θ(t, x, κ).

    Pipeline:
        1. eigendecompose the Hodge Laplacian,
        2. build OT-coupled training pairs using Sinkhorn,
        3. compute closed-form TFM bridge controls,
        4. train a Flax MLP with normalized MSE,
        5. return OTTFMTrainState.
    """
    evals, U = jnp.linalg.eigh(L)
    features, targets, pair_cost = build_ottfm_training_set(
        key,
        n_samples,
        evals,
        U,
        kappa,
        ot_batch_size=ot_batch_size,
        sinkhorn_epsilon=sinkhorn_epsilon,
        mu0_mode=mu0_mode,
    )

    features_n, feature_mean, feature_std = _normalize_batch(features)
    targets_n, target_mean, target_std = _normalize_batch(targets)

    model = BridgeMLP(hidden_dims=hidden_dims, output_dim=int(targets.shape[1]))
    params = model.init(jax.random.fold_in(key, 321), features_n[0])
    m, v = _adam_init(params)

    def predict_norm(params, batch_x):
        return jax.vmap(lambda z: model.apply(params, z))(batch_x)

    def loss_fn(params):
        return jnp.mean((predict_norm(params, features_n) - targets_n) ** 2)

    def step_fn(carry, step_idx):
        params, m, v = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, m, v = _adam_update(params, grads, m, v, step_idx + 1, learning_rate=learning_rate)
        return (params, m, v), loss

    (params, m, v), loss_history = jax.lax.scan(step_fn, (params, m, v), jnp.arange(n_steps))
    pred = predict_norm(params, features_n) * target_std + target_mean
    unnormalized_mse = jnp.mean((pred - targets) ** 2)

    return OTTFMTrainState(
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
        mean_pair_cost=float(jnp.mean(pair_cost)),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        ot_batch_size=int(ot_batch_size),
        mu0_mode=mu0_mode,
    )


def predict_ottfm_u(state: OTTFMTrainState, t: float, x_t: jnp.ndarray, *, control_scale: float = 0.92, clip_quantile: float = 3.0):
    """Predict the learned OT-TFM control u_θ(t, x_t, κ)."""
    t_batch = jnp.asarray([t], dtype=x_t.dtype)
    x_batch = x_t[None, :]
    features = make_ottfm_features(t_batch, x_batch, state.kappa)
    features_n = (features - state.feature_mean) / state.feature_std
    pred_n = state.model.apply(state.params, features_n[0])
    pred = pred_n * state.target_std + state.target_mean
    clip_value = clip_quantile * state.target_std
    pred = jnp.clip(pred, -clip_value, clip_value)
    return control_scale * pred


def generate_ottfm_samples(key, state: OTTFMTrainState, *, n_eval: int = 32, n_steps: int = 160, control_scale: float = 0.92):
    """Generate OT-TFM samples by Euler rollout.

    OT coupling is used during training pair construction only. Generation does
    not know target samples x1.

    Generation dynamics:
        dx/dt = -κLx + u_θ(t, x)
    """
    key0, key_ref = jax.random.split(key)
    sources = sample_mu0(key0, n_eval, state.evals, state.U, mode=state.mu0_mode)
    references = sample_mu1_harmonic_target(key_ref, n_eval, state.evals, state.U)

    dt = 1.0 / float(n_steps)

    def rollout_one(x0):
        def body(i, x_cur):
            t = jnp.asarray(i, dtype=x_cur.dtype) * dt
            u = predict_ottfm_u(state, t, x_cur, control_scale=control_scale)
            drift = -state.kappa * (state.L @ x_cur)
            return x_cur + dt * (drift + u)

        return jax.lax.fori_loop(0, n_steps, body, x0)

    generated = jax.vmap(rollout_one)(sources)
    return sources, generated, references


def generate_ottfm_sample(key, state: OTTFMTrainState, *, n_steps: int = 160, control_scale: float = 0.92):
    """Generate one display sample using generate_ottfm_samples(..., n_eval=1)."""
    sources, generated, references = generate_ottfm_samples(key, state, n_eval=1, n_steps=n_steps, control_scale=control_scale)
    return sources[0], generated[0], references[0]
