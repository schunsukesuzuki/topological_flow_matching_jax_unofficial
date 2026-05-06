import jax

from .itfm import train_itfm_vector_field, generate_itfm_samples
from .ottfm import train_ottfm_vector_field, generate_ottfm_samples
from .itfm_metrics import compare_generated_batches_to_reference


def run_mu0_ablation(
    key,
    L,
    kappa: float,
    *,
    n_samples: int = 1024,
    n_steps: int = 1200,
    learning_rate: float = 2e-3,
    n_eval: int = 32,
    rollout_steps: int = 160,
    control_scale: float = 0.92,
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
):
    """Run topology-aware initial distribution ablation.

    The ablation compares the same TFM method under two source distributions:

        standard:
            topology-unaware standard Gaussian μ0

        heat_gp:
            topology-aware heat Gaussian process μ0

    It is run for both I-TFM and OT-TFM:

        I-TFM standard μ0
        I-TFM heat GP μ0
        OT-TFM standard μ0
        OT-TFM heat GP μ0

    The returned aggregate metrics are mean/std over n_eval generated samples.
    """
    configs = [
        ("I-TFM / standard μ0", "itfm", "standard"),
        ("I-TFM / heat GP μ0", "itfm", "heat_gp"),
        ("OT-TFM / standard μ0", "ottfm", "standard"),
        ("OT-TFM / heat GP μ0", "ottfm", "heat_gp"),
    ]

    results = {}

    for idx, (label, method, mu0_mode) in enumerate(configs):
        train_key = jax.random.fold_in(key, 100 + idx)
        gen_key = jax.random.fold_in(key, 10_000 + idx)

        if method == "itfm":
            state = train_itfm_vector_field(
                train_key,
                L,
                kappa,
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
                kappa,
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

        results[label] = {
            "method": method,
            "mu0_mode": mu0_mode,
            "final_loss": state.final_loss,
            "unnormalized_mse": state.unnormalized_mse,
            "mean_pair_cost": mean_pair_cost,
            "aggregate_metrics": aggregate_metrics,
        }

    return results
