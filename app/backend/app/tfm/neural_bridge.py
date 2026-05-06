from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp

from .bridge import tfm_path_spectral, tfm_bridge_control_spectral
from .flax_model import BridgeMLP, make_bridge_features


@dataclass
class NeuralBridgeResult:
    """Result returned by the neural bridge-control fitting routine."""

    pred_u: jnp.ndarray
    target_u: jnp.ndarray
    loss_history: jnp.ndarray
    final_loss: float
    unnormalized_final_mse: float


def _normalize_batch(x: jnp.ndarray, eps: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Normalize a batch feature-wise and return normalized data plus stats."""
    mean = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0)
    std = jnp.where(std < eps, 1.0, std)
    return (x - mean) / std, mean, std


def _adam_init(params):
    """Create zero-valued Adam moment pytrees."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return zeros, zeros


def _adam_update(params, grads, m, v, step, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    """Apply one Adam update without adding an external optax dependency."""
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


def build_training_batch(x0, x1, L, kappa: float, times: jnp.ndarray):
    """Build supervised training pairs for u_theta(t, x_t).

    Teacher signal:
        x_t      = tfm_path_spectral(x0, x1, L, kappa, t)
        target_u = tfm_bridge_control_spectral(x0, x1, L, kappa, t)

    Improved features:
        concat([t, kappa], x_t, x0, x1, x1 - x0)

    This keeps the demo deterministic while making the MLP fitting much more
    stable than the first minimal concat([t], x_t) version.
    """
    def one_example(t):
        x_t = tfm_path_spectral(x0, x1, L, kappa, t)
        target_u = tfm_bridge_control_spectral(x0, x1, L, kappa, t)
        features = make_bridge_features(t, x_t, x0=x0, x1=x1, kappa=kappa)
        return features, target_u

    return jax.vmap(one_example)(times)


def fit_bridge_control_mlp(
    x0,
    x1,
    L,
    kappa: float,
    query_t: float,
    *,
    n_steps: int = 1500,
    learning_rate: float = 3e-3,
    n_time_samples: int = 33,
    hidden_dims: Sequence[int] = (256, 256, 128),
    seed: int = 0,
):
    """Fit a Flax MLP to the analytical TFM bridge-control target.

    Accuracy-oriented changes from the initial neural patch:

    1. Uses Adam instead of plain SGD.
    2. Uses more deterministic time samples.
    3. Normalizes both features and targets.
    4. Uses richer features: concat([t, kappa], x_t, x0, x1, x1 - x0).
    5. Uses a slightly larger MLP.

    The goal is not to train a reusable generative model here. The goal is to
    make the app's neural mode visibly approximate the closed-form analytical
    TFM bridge control for the current displayed pair.
    """
    output_dim = int(x0.shape[0])
    model = BridgeMLP(hidden_dims=hidden_dims, output_dim=output_dim)

    # Use interior samples plus the exact query time to improve the displayed
    # prediction at the current slider position.
    base_times = jnp.linspace(0.02, 0.98, n_time_samples, dtype=x0.dtype)
    query_time = jnp.asarray([query_t], dtype=x0.dtype)
    times = jnp.sort(jnp.concatenate([base_times, query_time], axis=0))

    features, targets = build_training_batch(x0, x1, L, kappa, times)

    features_n, feat_mean, feat_std = _normalize_batch(features)
    targets_n, target_mean, target_std = _normalize_batch(targets)

    rng = jax.random.PRNGKey(seed)
    params = model.init(rng, features_n[0])
    m, v = _adam_init(params)

    def predict_norm(params, batch_x):
        return jax.vmap(lambda z: model.apply(params, z))(batch_x)

    def loss_fn(params):
        pred_n = predict_norm(params, features_n)
        return jnp.mean((pred_n - targets_n) ** 2)

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

    (params, m, v), loss_history = jax.lax.scan(
        step_fn,
        (params, m, v),
        jnp.arange(n_steps),
    )

    query_x_t = tfm_path_spectral(x0, x1, L, kappa, query_t)
    query_features = make_bridge_features(query_t, query_x_t, x0=x0, x1=x1, kappa=kappa)
    query_features_n = (query_features - feat_mean) / feat_std

    pred_u_n = model.apply(params, query_features_n)
    pred_u = pred_u_n * target_std + target_mean
    target_u = tfm_bridge_control_spectral(x0, x1, L, kappa, query_t)

    unnormalized_final_mse = jnp.mean((pred_u - target_u) ** 2)

    return NeuralBridgeResult(
        pred_u=pred_u,
        target_u=target_u,
        loss_history=loss_history,
        final_loss=float(loss_history[-1]),
        unnormalized_final_mse=float(unnormalized_final_mse),
    )
