import jax

from .itfm import train_itfm_vector_field, generate_itfm_samples
from .ottfm import train_ottfm_vector_field, generate_ottfm_samples
from .itfm_metrics import compare_generated_batches_to_reference
from .distribution_metrics import distribution_distance_summary


def run_kappa_sweep(
    key,
    L,
    kappas,
    *,
    n_samples: int = 768,
    n_steps: int = 900,
    learning_rate: float = 2e-3,
    n_eval: int = 32,
    rollout_steps: int = 160,
    control_scale: float = 0.92,
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
    mu0_mode: str = "heat_gp",
):
    """Evaluate I-TFM / OT-TFM across different Hodge heat-drift strengths.

    The sweep is designed to answer:

        How does κ in dx/dt = -κLx + uθ(t, x)
        affect generated distribution quality?

    For each κ, the function trains:
        I-TFM
        OT-TFM

    and returns:
        training MSE
        mean pair cost for OT-TFM
        generated spectral/Hodge energy summaries
        distribution distances to reference μ1
    """
    rows = []

    for k_idx, kappa in enumerate(kappas):
        for m_idx, method in enumerate(["I-TFM", "OT-TFM"]):
            train_key = jax.random.fold_in(key, 1000 + 10 * k_idx + m_idx)
            gen_key = jax.random.fold_in(key, 2000 + 10 * k_idx + m_idx)

            if method == "I-TFM":
                state = train_itfm_vector_field(
                    train_key,
                    L,
                    float(kappa),
                    n_samples=n_samples,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    mu0_mode=mu0_mode,
                )
                sources, generated, references = generate_itfm_samples(
                    gen_key,
                    state,
                    n_eval=n_eval,
                    n_steps=rollout_steps,
                    control_scale=control_scale,
                )
                mean_pair_cost = None
            else:
                state = train_ottfm_vector_field(
                    train_key,
                    L,
                    float(kappa),
                    n_samples=n_samples,
                    n_steps=n_steps,
                    learning_rate=learning_rate,
                    ot_batch_size=ot_batch_size,
                    sinkhorn_epsilon=sinkhorn_epsilon,
                    mu0_mode=mu0_mode,
                )
                sources, generated, references = generate_ottfm_samples(
                    gen_key,
                    state,
                    n_eval=n_eval,
                    n_steps=rollout_steps,
                    control_scale=control_scale,
                )
                mean_pair_cost = state.mean_pair_cost

            aggregate_metrics = compare_generated_batches_to_reference(
                sources,
                generated,
                references,
                state.evals,
                state.U,
            )
            distances = distribution_distance_summary(
                generated,
                references,
                state.U,
                seed=5000 + 10 * k_idx + m_idx,
            )

            rows.append({
                "kappa": float(kappa),
                "method": method,
                "final_loss": state.final_loss,
                "unnormalized_mse": state.unnormalized_mse,
                "mean_pair_cost": mean_pair_cost,
                "aggregate_metrics": aggregate_metrics,
                "distances": distances,
            })

    return rows
