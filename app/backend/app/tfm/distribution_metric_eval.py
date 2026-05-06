import jax

from .cfm_distribution import train_cfm_vector_field, generate_cfm_samples
from .itfm import train_itfm_vector_field, generate_itfm_samples
from .ottfm import train_ottfm_vector_field, generate_ottfm_samples
from .itfm_metrics import compare_generated_batches_to_reference
from .distribution_metrics import distribution_distance_summary


def run_distribution_metric_eval(
    key,
    L,
    kappa: float,
    *,
    n_samples: int = 1024,
    n_steps: int = 1200,
    learning_rate: float = 2e-3,
    n_eval: int = 32,
    rollout_steps: int = 160,
    cfm_control_scale: float = 1.0,
    tfm_control_scale: float = 0.92,
    ot_batch_size: int = 64,
    sinkhorn_epsilon: float = 0.75,
):
    """Run four-method evaluation with distribution distance metrics.

    Methods:
        I-CFM
        OT-CFM
        I-TFM
        OT-TFM

    Distances:
        mmd_rbf
        spectral_sliced_wasserstein
        spectral_mode_wasserstein
    """
    configs = [
        ("I-CFM", "icfm"),
        ("OT-CFM", "otcfm"),
        ("I-TFM", "itfm"),
        ("OT-TFM", "ottfm"),
    ]

    results = {}

    for idx, (label, method) in enumerate(configs):
        train_key = jax.random.fold_in(key, 100 + idx)
        gen_key = jax.random.fold_in(key, 10_000 + idx)

        if method == "icfm":
            state = train_cfm_vector_field(
                train_key,
                L,
                kappa,
                coupling="independent",
                n_samples=n_samples,
                n_steps=n_steps,
                learning_rate=learning_rate,
                ot_batch_size=ot_batch_size,
                sinkhorn_epsilon=sinkhorn_epsilon,
            )
            sources, generated, references = generate_cfm_samples(
                gen_key,
                state,
                n_eval=n_eval,
                n_steps=rollout_steps,
                control_scale=cfm_control_scale,
            )
            mean_pair_cost = state.mean_pair_cost

        elif method == "otcfm":
            state = train_cfm_vector_field(
                train_key,
                L,
                kappa,
                coupling="ot",
                n_samples=n_samples,
                n_steps=n_steps,
                learning_rate=learning_rate,
                ot_batch_size=ot_batch_size,
                sinkhorn_epsilon=sinkhorn_epsilon,
            )
            sources, generated, references = generate_cfm_samples(
                gen_key,
                state,
                n_eval=n_eval,
                n_steps=rollout_steps,
                control_scale=cfm_control_scale,
            )
            mean_pair_cost = state.mean_pair_cost

        elif method == "itfm":
            state = train_itfm_vector_field(
                train_key,
                L,
                kappa,
                n_samples=n_samples,
                n_steps=n_steps,
                learning_rate=learning_rate,
                mu0_mode="heat_gp",
            )
            sources, generated, references = generate_itfm_samples(
                gen_key,
                state,
                n_eval=n_eval,
                n_steps=rollout_steps,
                control_scale=tfm_control_scale,
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
                mu0_mode="heat_gp",
            )
            sources, generated, references = generate_ottfm_samples(
                gen_key,
                state,
                n_eval=n_eval,
                n_steps=rollout_steps,
                control_scale=tfm_control_scale,
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
            seed=777 + idx,
        )

        results[label] = {
            "method": method,
            "final_loss": state.final_loss,
            "unnormalized_mse": state.unnormalized_mse,
            "mean_pair_cost": mean_pair_cost,
            "aggregate_metrics": aggregate_metrics,
            "distances": distances,
        }

    return results
