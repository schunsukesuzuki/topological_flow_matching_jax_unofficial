import jax
import jax.numpy as jnp

from .datasets import sample_mu0, sample_mu1_harmonic_target


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
    plan = u[:, None] * K * v[None, :]
    return plan / (jnp.sum(plan) + 1e-8)


def _euclidean_cost_matrix(x0_pool: jnp.ndarray, x1_pool: jnp.ndarray):
    """Pairwise squared Euclidean cost used by OT-CFM."""
    diff = x0_pool[:, None, :] - x1_pool[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def _tfm_cost_matrix_from_eig(
    x0_pool: jnp.ndarray,
    x1_pool: jnp.ndarray,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    kappa: float,
    eps: float = 1e-8,
):
    """Pairwise TFM transport cost used by OT-TFM.

    The cost is evaluated in Hodge spectral coordinates. Each eigenmode
    contributes independently, and nonzero modes are weighted by the TFM
    reference-process geometry.
    """
    y0 = x0_pool @ U
    y1 = x1_pool @ U

    omega = kappa * evals
    exp_full = jnp.exp(-omega)
    coeff = 2.0 * omega / (1.0 - jnp.exp(-2.0 * omega) + eps)

    linear_cost = (y1[None, :, :] - y0[:, None, :]) ** 2
    residual = y1[None, :, :] - exp_full[None, None, :] * y0[:, None, :]
    spectral_cost = coeff[None, None, :] * residual**2

    use_linear = (jnp.abs(omega) < eps)[None, None, :]
    per_mode = jnp.where(use_linear, linear_cost, spectral_cost)

    return jnp.sum(per_mode, axis=-1)


def _sample_plan_pairs(key, plan: jnp.ndarray, n_pairs: int):
    """Sample pair indices from a flattened transport plan."""
    n, m = plan.shape
    probs = jnp.reshape(plan, (-1,))
    flat = jax.random.choice(key, probs.shape[0], shape=(n_pairs,), replace=True, p=probs)
    rows = flat // m
    cols = flat % m
    return rows, cols


def _top_plan_entries(plan: jnp.ndarray, cost: jnp.ndarray, top_k: int = 12):
    """Return the largest plan entries for readable UI display."""
    flat_plan = jnp.reshape(plan, (-1,))
    flat_cost = jnp.reshape(cost, (-1,))
    k = min(top_k, flat_plan.shape[0])

    values, indices = jax.lax.top_k(flat_plan, k)
    n_cols = plan.shape[1]
    rows = indices // n_cols
    cols = indices % n_cols
    costs = flat_cost[indices]

    return [
        {
            "source_index": int(rows[i]),
            "target_index": int(cols[i]),
            "plan_mass": float(values[i]),
            "cost": float(costs[i]),
        }
        for i in range(k)
    ]


def _row_argmax_pairs(plan: jnp.ndarray, cost: jnp.ndarray):
    """Deterministic most-likely target per source row."""
    cols = jnp.argmax(plan, axis=1)
    rows = jnp.arange(plan.shape[0])
    masses = plan[rows, cols]
    costs = cost[rows, cols]
    return [
        {
            "source_index": int(rows[i]),
            "target_index": int(cols[i]),
            "plan_mass": float(masses[i]),
            "cost": float(costs[i]),
        }
        for i in range(plan.shape[0])
    ]


def _summarize_plan(plan: jnp.ndarray, cost: jnp.ndarray):
    """Compact diagnostics for the coupling plan."""
    row_entropy = -jnp.sum(plan * jnp.log(plan + 1e-8), axis=1)
    expected_cost = jnp.sum(plan * cost)

    row_max = jnp.max(plan, axis=1)
    col_max = jnp.max(plan, axis=0)

    return {
        "expected_cost": float(expected_cost),
        "plan_entropy": float(-jnp.sum(plan * jnp.log(plan + 1e-8))),
        "mean_row_entropy": float(jnp.mean(row_entropy)),
        "mean_row_max_mass": float(jnp.mean(row_max)),
        "mean_col_max_mass": float(jnp.mean(col_max)),
        "min_cost": float(jnp.min(cost)),
        "max_cost": float(jnp.max(cost)),
        "mean_cost": float(jnp.mean(cost)),
    }


def build_ot_coupling_visualization(
    key,
    evals: jnp.ndarray,
    U: jnp.ndarray,
    *,
    kappa: float,
    method: str = "ot_tfm",
    batch_size: int = 16,
    sinkhorn_epsilon: float = 0.75,
    mu0_mode: str = "heat_gp",
    top_k: int = 12,
):
    """Build a small OT coupling payload for UI visualization.

    method:
        ot_cfm:
            Euclidean squared cost.

        ot_tfm:
            TFM transport cost in Hodge spectral coordinates.

    The returned data is intentionally small enough to render directly as
    heatmaps in React.
    """
    key0, key1, key_sample = jax.random.split(key, 3)

    x0_pool = sample_mu0(key0, batch_size, evals, U, mode=mu0_mode)
    x1_pool = sample_mu1_harmonic_target(key1, batch_size, evals, U)

    if method == "ot_cfm":
        cost = _euclidean_cost_matrix(x0_pool, x1_pool)
        label = "OT-CFM / Euclidean cost"
    elif method == "ot_tfm":
        cost = _tfm_cost_matrix_from_eig(x0_pool, x1_pool, evals, U, kappa)
        label = "OT-TFM / TFM transport cost"
    else:
        raise ValueError(f"Unknown coupling visualization method: {method}")

    plan = _sinkhorn_plan(cost, epsilon=sinkhorn_epsilon)

    sample_rows, sample_cols = _sample_plan_pairs(
        key_sample,
        plan,
        n_pairs=min(batch_size, 24),
    )

    sampled_pairs = [
        {
            "source_index": int(sample_rows[i]),
            "target_index": int(sample_cols[i]),
            "plan_mass": float(plan[sample_rows[i], sample_cols[i]]),
            "cost": float(cost[sample_rows[i], sample_cols[i]]),
        }
        for i in range(sample_rows.shape[0])
    ]

    return {
        "method": method,
        "label": label,
        "batch_size": int(batch_size),
        "sinkhorn_epsilon": float(sinkhorn_epsilon),
        "mu0_mode": mu0_mode,
        "kappa": float(kappa),
        "cost_matrix": jnp.asarray(cost).tolist(),
        "plan_matrix": jnp.asarray(plan).tolist(),
        "sampled_pairs": sampled_pairs,
        "top_pairs": _top_plan_entries(plan, cost, top_k=top_k),
        "row_argmax_pairs": _row_argmax_pairs(plan, cost),
        "summary": _summarize_plan(plan, cost),
    }
