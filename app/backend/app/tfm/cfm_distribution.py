from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from .datasets import sample_mu0_heat_noise, sample_mu1_harmonic_target
from .flax_model import BridgeMLP


@dataclass
class CFMTrainState:
    """In-memory trained distribution-level CFM state for the demo app.

    This state is used for both I-CFM and OT-CFM. The difference is only how
    training pairs are sampled:

        I-CFM:
            independent coupling μ0 ⊗ μ1

        OT-CFM:
            entropic minibatch OT under Euclidean signal-space cost

    Unlike TFM, generation uses:

        dx/dt = uθ(t, x)

    with no Hodge heat drift term.
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
    coupling: str
    mean_pair_cost: float
    sinkhorn_epsilon: float
    ot_batch_size: int


def _normalize_batch(x: jnp.ndarray, eps: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    std = jnp.where(std < eps, 1.0, std)
    return (x - mean) / std, mean, std


def _adam_init(params):
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


def _sinkhorn_plan(cost: jnp.ndarray, epsilon: float = 0.75, n_iter: int = 80):
    """Entropic minibatch OT plan with uniform marginals."""
    n, m = cost.shape
    K = jnp.exp(-(cost - jnp.min(cost)) / epsilon) + 1e-8

    a = jnp.ones((n,), dtype=cost.dtype) / n
    b = jnp.ones((m,), dtype=cost.dtype) / m

    u = jnp.ones_like(a)
    v = jnp.ones_like(b)

    def body(_, carry):
        u, v = carry
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)
        return u, v

    u, v = jax.lax.fori_loop(0, n_iter, body, (u, v))
    P = u[:, None] * K * v[None, :]
    return P / (jnp.sum(P) + 1e-8)


def _euclidean_cost_matrix(x0_pool: jnp.ndarray, x1_pool: jnp.ndarray):
    """Pairwise squared Euclidean cost for OT-CFM coupling."""
    diff = x0_pool[:, None, :] - x1_pool[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def _sample_icfm_pairs(key, n_samples: int, evals: jnp.ndarray, U: jnp.ndarray):
    """Sample independent I-CFM training pairs (X0, X1) ~ μ0 ⊗ μ1."""
    key0, key1 = jax.random.split(key)
    x0 = sample_mu0_heat_noise(key0, n_samples, evals, U)
    x1 = sample_mu1_harmonic_target(key1, n_samples, evals, U)
    pair_cost = jnp.mean((x1 - x0) ** 2, axis=1)
    return x0, x1, pair_cost


def _sample_otcfm_pairs_one_batch(
    key,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    ot_batch_size: int,
    sinkhorn_epsilon: float,
):
    """Sample one minibatch of Euclidean OT-biased CFM pairs."""
    key0, key1, key_pair = jax.random.split(key, 3)
    x0_pool = sample_mu0_heat_noise(key0, ot_batch_size, evals, U)
    x1_pool = sample_mu1_harmonic_target(key1, ot_batch_size, evals, U)

    cost = _euclidean_cost_matrix(x0_pool, x1_pool)
    plan = _sinkhorn_plan(cost, epsilon=sinkhorn_epsilon)
    probs = jnp.reshape(plan, (-1,))

    flat_idx = jax.random.choice(
        key_pair,
        probs.shape[0],
        shape=(ot_batch_size,),
        replace=True,
        p=probs,
    )

    row = flat_idx // ot_batch_size
    col = flat_idx % ot_batch_size

    return x0_pool[row], x1_pool[col], cost[row, col]


def _sample_otcfm_pairs(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
):
    """Sample OT-CFM pairs using entropic minibatch Euclidean OT coupling."""
    n_batches = int((n_samples + ot_batch_size - 1) // ot_batch_size)

    x0_chunks = []
    x1_chunks = []
    cost_chunks = []

    for i in range(n_batches):
        x0_b, x1_b, c_b = _sample_otcfm_pairs_one_batch(
            jax.random.fold_in(key, i),
            evals,
            U,
            ot_batch_size=ot_batch_size,
            sinkhorn_epsilon=sinkhorn_epsilon,
        )
        x0_chunks.append(x0_b)
        x1_chunks.append(x1_b)
        cost_chunks.append(c_b)

    x0 = jnp.concatenate(x0_chunks, axis=0)[:n_samples]
    x1 = jnp.concatenate(x1_chunks, axis=0)[:n_samples]
    pair_cost = jnp.concatenate(cost_chunks, axis=0)[:n_samples]
    return x0, x1, pair_cost


def make_cfm_features(t_batch: jnp.ndarray, x_t_batch: jnp.ndarray, kappa: float):
    """Feature map for distribution-level CFM.

    κ is included for interface consistency with TFM panels, but CFM rollout
    does not use a Hodge heat drift term.
    """
    t_col = t_batch[:, None]
    kappa_col = jnp.ones_like(t_col) * jnp.asarray(kappa, dtype=x_t_batch.dtype)
    return jnp.concatenate([t_col, kappa_col, x_t_batch], axis=1)


def build_cfm_training_set(
    key,
    n_samples: int,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    kappa: float,
    *,
    coupling: str = "independent",
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
):
    """Build distribution-level CFM supervised training data.

    CFM bridge:

        x_t = (1 - t)x0 + t x1
        target_u = x1 - x0

    Coupling options:

        independent:
            I-CFM, pairs sampled from μ0 ⊗ μ1

        ot:
            OT-CFM, pairs sampled from Euclidean-cost Sinkhorn plan
    """
    if coupling == "independent":
        x0, x1, pair_cost = _sample_icfm_pairs(key, n_samples, evals, U)
    elif coupling == "ot":
        x0, x1, pair_cost = _sample_otcfm_pairs(
            key,
            n_samples,
            evals,
            U,
            ot_batch_size=ot_batch_size,
            sinkhorn_epsilon=sinkhorn_epsilon,
        )
    else:
        raise ValueError(f"Unknown CFM coupling: {coupling}")

    t = jax.random.uniform(jax.random.fold_in(key, 9999), (n_samples,), minval=0.02, maxval=0.98)

    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
    target_u = x1 - x0
    features = make_cfm_features(t, x_t, kappa)

    return features, target_u, pair_cost


def train_cfm_vector_field(
    key,
    L: jnp.ndarray,
    kappa: float,
    *,
    coupling: str = "independent",
    n_samples: int = 2048,
    n_steps: int = 2000,
    learning_rate: float = 2e-3,
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
    hidden_dims: Sequence[int] = (256, 256, 128),
):
    """Train I-CFM or OT-CFM distribution-level vector field."""
    evals, U = jnp.linalg.eigh(L)
    features, targets, pair_cost = build_cfm_training_set(
        key,
        n_samples,
        evals,
        U,
        kappa,
        coupling=coupling,
        ot_batch_size=ot_batch_size,
        sinkhorn_epsilon=sinkhorn_epsilon,
    )

    features_n, feature_mean, feature_std = _normalize_batch(features)
    targets_n, target_mean, target_std = _normalize_batch(targets)

    output_dim = int(targets.shape[1])
    model = BridgeMLP(hidden_dims=hidden_dims, output_dim=output_dim)

    params = model.init(jax.random.fold_in(key, 777), features_n[0])
    m, v = _adam_init(params)

    def predict_norm(params, batch_x):
        return jax.vmap(lambda z: model.apply(params, z))(batch_x)

    def loss_fn(params):
        pred = predict_norm(params, features_n)
        return jnp.mean((pred - targets_n) ** 2)

    def step_fn(carry, step_idx):
        params, m, v = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, m, v = _adam_update(params, grads, m, v, step_idx + 1, learning_rate=learning_rate)
        return (params, m, v), loss

    (params, m, v), loss_history = jax.lax.scan(
        step_fn,
        (params, m, v),
        jnp.arange(n_steps),
    )

    pred_n = predict_norm(params, features_n)
    pred = pred_n * target_std + target_mean
    unnormalized_mse = jnp.mean((pred - targets) ** 2)

    return CFMTrainState(
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
        coupling=coupling,
        mean_pair_cost=float(jnp.mean(pair_cost)),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        ot_batch_size=int(ot_batch_size),
    )


def predict_cfm_u(
    state: CFMTrainState,
    t: float,
    x_t: jnp.ndarray,
    *,
    control_scale: float = 1.0,
    clip_quantile: float = 3.0,
):
    """Predict CFM vector field uθ(t, x_t)."""
    t_batch = jnp.asarray([t], dtype=x_t.dtype)
    x_batch = x_t[None, :]
    features = make_cfm_features(t_batch, x_batch, state.kappa)
    features_n = (features - state.feature_mean) / state.feature_std

    pred_n = state.model.apply(state.params, features_n[0])
    pred = pred_n * state.target_std + state.target_mean

    clip_value = clip_quantile * state.target_std
    pred = jnp.clip(pred, -clip_value, clip_value)
    return control_scale * pred


def generate_cfm_samples(
    key,
    state: CFMTrainState,
    *,
    n_eval: int = 32,
    n_steps: int = 160,
    control_scale: float = 1.0,
):
    """Generate a batch of CFM samples by integrating dx/dt = uθ(t, x)."""
    key0, key_ref = jax.random.split(key)

    sources = sample_mu0_heat_noise(key0, n_eval, state.evals, state.U)
    references = sample_mu1_harmonic_target(key_ref, n_eval, state.evals, state.U)

    dt = 1.0 / float(n_steps)

    def rollout_one(x0):
        def body(i, x_cur):
            t = jnp.asarray(i, dtype=x_cur.dtype) * dt
            u = predict_cfm_u(state, t, x_cur, control_scale=control_scale)
            return x_cur + dt * u

        return jax.lax.fori_loop(0, n_steps, body, x0)

    generated = jax.vmap(rollout_one)(sources)
    return sources, generated, references


def generate_cfm_sample(
    key,
    state: CFMTrainState,
    *,
    n_steps: int = 160,
    control_scale: float = 1.0,
):
    """Single-sample helper for display."""
    sources, generated, references = generate_cfm_samples(
        key,
        state,
        n_eval=1,
        n_steps=n_steps,
        control_scale=control_scale,
    )
    return sources[0], generated[0], references[0]
