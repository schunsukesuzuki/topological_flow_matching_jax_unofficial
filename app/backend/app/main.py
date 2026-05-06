from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import jax
import jax.numpy as jnp

from .schemas import (
    ComputeRequest,
    ComputeResponse,
    DistributionMetricEvalRequest,
    DistributionMetricEvalResponse,
    GenerateCFMRequest,
    GenerateCFMResponse,
    GenerateITFMRequest,
    GenerateITFMResponse,
    GenerateOTTFMRequest,
    GenerateOTTFMResponse,
    KappaSweepRequest,
    KappaSweepResponse,
    Mu0AblationRequest,
    Mu0AblationResponse,
    OTCouplingVisualizationRequest,
    OTCouplingVisualizationResponse,
    PresetsResponse,
    TrainCFMRequest,
    TrainCFMResponse,
    TrainITFMRequest,
    TrainITFMResponse,
    TrainOTTFMRequest,
    TrainOTTFMResponse,
)
from .tfm.simplicial_complex import annulus_complex, annulus_layout
from .tfm.boundary import build_boundary_matrix
from .tfm.hodge import hodge_laplacian
from .tfm.bridge import (
    cfm_path,
    cfm_velocity,
    cfm_transport_cost,
    tfm_path_spectral,
    tfm_bridge_control_spectral,
    tfm_transport_cost,
)
from .tfm.neural_bridge import fit_bridge_control_mlp
from .tfm.cfm_distribution import train_cfm_vector_field, generate_cfm_sample, generate_cfm_samples
from .tfm.itfm import train_itfm_vector_field, generate_itfm_sample, generate_itfm_samples
from .tfm.ottfm import train_ottfm_vector_field, generate_ottfm_sample, generate_ottfm_samples
from .tfm.mu0_ablation import run_mu0_ablation
from .tfm.distribution_metric_eval import run_distribution_metric_eval
from .tfm.distribution_metrics import distribution_distance_summary
from .tfm.kappa_sweep import run_kappa_sweep
from .tfm.coupling_visualization import build_ot_coupling_visualization
from .tfm.itfm_metrics import compare_generated_to_reference, compare_generated_batches_to_reference
from .tfm.presets import PRESETS
from .tfm.figure1 import build_figure1_payload

app = FastAPI(title="Topological Flow Matching Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ICFM_STATE = None
OTCFM_STATE = None
ITFM_STATE = None
OTTFM_STATE = None


def arr(x):
    return jnp.asarray(x, dtype=jnp.float32)


def to_list(x):
    return jnp.asarray(x).tolist()


def build_annulus_operators(with_face: bool):
    nodes, edges, faces = annulus_complex(n=18, with_faces=with_face)
    B1 = build_boundary_matrix(nodes, edges)
    B2 = build_boundary_matrix(edges, faces) if faces else jnp.zeros((len(edges), 0), dtype=jnp.float32)
    L1 = hodge_laplacian(B_k=B1, B_k_plus_1=B2)
    return nodes, edges, faces, B1, B2, L1


@app.get("/api/presets", response_model=PresetsResponse)
def get_presets():
    return {"presets": PRESETS}


@app.post("/api/compute", response_model=ComputeResponse)
def compute(req: ComputeRequest):
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)

    x0 = arr(req.x0)
    x1 = arr(req.x1)
    if x0.shape[0] != len(edges) or x1.shape[0] != len(edges):
        raise ValueError(f"x0/x1 must have length {len(edges)} for the annulus edge complex")

    t = float(req.t)
    kappa = float(req.kappa)

    evals, U = jnp.linalg.eigh(L1)
    cfm_xt = cfm_path(x0, x1, t)
    tfm_xt = tfm_path_spectral(x0, x1, L1, kappa, t)
    cfm_u = cfm_velocity(x0, x1)
    analytical_tfm_u = tfm_bridge_control_spectral(x0, x1, L1, kappa, t)

    nn_loss = None
    nn_unnormalized_mse = None
    nn_steps = None

    if req.bridge_mode == "neural":
        neural = fit_bridge_control_mlp(
            x0,
            x1,
            L1,
            kappa,
            t,
            n_steps=req.nn_steps,
            learning_rate=3e-3,
            n_time_samples=33,
        )
        tfm_u = neural.pred_u
        nn_loss = neural.final_loss
        nn_unnormalized_mse = neural.unnormalized_final_mse
        nn_steps = req.nn_steps
    else:
        tfm_u = analytical_tfm_u

    cfm_cost = cfm_transport_cost(x0, x1)
    tfm_cost = tfm_transport_cost(x0, x1, L1, kappa)
    B1B2 = B1 @ B2 if B2.size > 0 else jnp.zeros((len(nodes), 0), dtype=jnp.float32)

    return {
        "nodes": annulus_layout(n=18),
        "edges": [list(e) for e in edges],
        "faces": [list(f) for f in faces],
        "B1": to_list(B1),
        "B2": to_list(B2),
        "B1B2": to_list(B1B2),
        "L1": to_list(L1),
        "eigenvalues": to_list(evals),
        "eigenvectors": to_list(U),
        "cfm_xt": to_list(cfm_xt),
        "tfm_xt": to_list(tfm_xt),
        "cfm_u": to_list(cfm_u),
        "tfm_u": to_list(tfm_u),
        "analytical_tfm_u": to_list(analytical_tfm_u),
        "cfm_cost": float(cfm_cost),
        "tfm_cost": float(tfm_cost),
        "bridge_mode": req.bridge_mode,
        "nn_loss": nn_loss,
        "nn_unnormalized_mse": nn_unnormalized_mse,
        "nn_steps": nn_steps,
        "figure1": build_figure1_payload(kappa=kappa, n=18),
    }


@app.post("/api/train_cfm", response_model=TrainCFMResponse)
def train_cfm(req: TrainCFMRequest):
    global ICFM_STATE, OTCFM_STATE
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    state = train_cfm_vector_field(
        jax.random.PRNGKey(req.seed),
        L1,
        float(req.kappa),
        coupling=req.coupling,
        n_samples=req.n_samples,
        n_steps=req.n_steps,
        learning_rate=req.learning_rate,
        ot_batch_size=req.ot_batch_size,
        sinkhorn_epsilon=req.sinkhorn_epsilon,
    )

    if req.coupling == "independent":
        ICFM_STATE = state
    else:
        OTCFM_STATE = state

    return {
        "status": "trained",
        "coupling": req.coupling,
        "n_samples": req.n_samples,
        "n_steps": req.n_steps,
        "kappa": req.kappa,
        "final_loss": state.final_loss,
        "unnormalized_mse": state.unnormalized_mse,
        "mean_pair_cost": state.mean_pair_cost,
        "ot_batch_size": state.ot_batch_size,
        "sinkhorn_epsilon": state.sinkhorn_epsilon,
    }


@app.post("/api/generate_cfm", response_model=GenerateCFMResponse)
def generate_cfm(req: GenerateCFMRequest):
    global ICFM_STATE, OTCFM_STATE
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)

    state = ICFM_STATE if req.coupling == "independent" else OTCFM_STATE
    if state is None:
        state = train_cfm_vector_field(
            jax.random.PRNGKey(21 if req.coupling == "independent" else 22),
            L1,
            float(req.kappa),
            coupling=req.coupling,
            n_samples=2048,
            n_steps=2000,
            learning_rate=2e-3,
            ot_batch_size=64,
            sinkhorn_epsilon=0.75,
        )
        if req.coupling == "independent":
            ICFM_STATE = state
        else:
            OTCFM_STATE = state

    source, generated, target_ref = generate_cfm_sample(
        jax.random.PRNGKey(req.seed),
        state,
        n_steps=req.rollout_steps,
        control_scale=req.control_scale,
    )
    sources_b, generated_b, target_refs_b = generate_cfm_samples(
        jax.random.PRNGKey(req.seed + 30_000),
        state,
        n_eval=req.n_eval,
        n_steps=req.rollout_steps,
        control_scale=req.control_scale,
    )

    metrics = compare_generated_to_reference(source, generated, target_ref, state.evals, state.U)
    aggregate_metrics = compare_generated_batches_to_reference(sources_b, generated_b, target_refs_b, state.evals, state.U)
    distances = distribution_distance_summary(generated_b, target_refs_b, state.U, seed=req.seed + 300)

    return {
        "nodes": annulus_layout(n=18),
        "edges": [list(e) for e in edges],
        "faces": [list(f) for f in faces],
        "source_signal": to_list(source),
        "generated_signal": to_list(generated),
        "target_reference_signal": to_list(target_ref),
        "trained_final_loss": state.final_loss,
        "trained_unnormalized_mse": state.unnormalized_mse,
        "mean_pair_cost": state.mean_pair_cost,
        "metrics": metrics,
        "aggregate_metrics": aggregate_metrics,
        "distances": distances,
        "n_eval": req.n_eval,
        "rollout_steps": req.rollout_steps,
        "control_scale": req.control_scale,
        "coupling": req.coupling,
        "ot_batch_size": state.ot_batch_size,
        "sinkhorn_epsilon": state.sinkhorn_epsilon,
    }


@app.post("/api/train_itfm", response_model=TrainITFMResponse)
def train_itfm(req: TrainITFMRequest):
    global ITFM_STATE
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    ITFM_STATE = train_itfm_vector_field(
        jax.random.PRNGKey(req.seed),
        L1,
        float(req.kappa),
        n_samples=req.n_samples,
        n_steps=req.n_steps,
        learning_rate=req.learning_rate,
        mu0_mode=req.mu0_mode,
    )
    return {
        "status": "trained",
        "n_samples": req.n_samples,
        "n_steps": req.n_steps,
        "kappa": req.kappa,
        "final_loss": ITFM_STATE.final_loss,
        "unnormalized_mse": ITFM_STATE.unnormalized_mse,
        "mu0_mode": ITFM_STATE.mu0_mode,
    }


@app.post("/api/generate_itfm", response_model=GenerateITFMResponse)
def generate_itfm(req: GenerateITFMRequest):
    global ITFM_STATE
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)

    if ITFM_STATE is None:
        ITFM_STATE = train_itfm_vector_field(
            jax.random.PRNGKey(0),
            L1,
            float(req.kappa),
            n_samples=2048,
            n_steps=2000,
            learning_rate=2e-3,
            mu0_mode="heat_gp",
        )

    source, generated, target_ref = generate_itfm_sample(
        jax.random.PRNGKey(req.seed),
        ITFM_STATE,
        n_steps=req.rollout_steps,
        control_scale=req.control_scale,
    )
    sources_b, generated_b, target_refs_b = generate_itfm_samples(
        jax.random.PRNGKey(req.seed + 10_000),
        ITFM_STATE,
        n_eval=req.n_eval,
        n_steps=req.rollout_steps,
        control_scale=req.control_scale,
    )

    metrics = compare_generated_to_reference(source, generated, target_ref, ITFM_STATE.evals, ITFM_STATE.U)
    aggregate_metrics = compare_generated_batches_to_reference(sources_b, generated_b, target_refs_b, ITFM_STATE.evals, ITFM_STATE.U)
    distances = distribution_distance_summary(generated_b, target_refs_b, ITFM_STATE.U, seed=req.seed + 100)

    return {
        "nodes": annulus_layout(n=18),
        "edges": [list(e) for e in edges],
        "faces": [list(f) for f in faces],
        "source_signal": to_list(source),
        "generated_signal": to_list(generated),
        "target_reference_signal": to_list(target_ref),
        "trained_final_loss": ITFM_STATE.final_loss,
        "trained_unnormalized_mse": ITFM_STATE.unnormalized_mse,
        "metrics": metrics,
        "aggregate_metrics": aggregate_metrics,
        "distances": distances,
        "n_eval": req.n_eval,
        "rollout_steps": req.rollout_steps,
        "control_scale": req.control_scale,
        "mu0_mode": ITFM_STATE.mu0_mode,
    }


@app.post("/api/train_ot_tfm", response_model=TrainOTTFMResponse)
def train_ot_tfm(req: TrainOTTFMRequest):
    global OTTFM_STATE
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    OTTFM_STATE = train_ottfm_vector_field(
        jax.random.PRNGKey(req.seed),
        L1,
        float(req.kappa),
        n_samples=req.n_samples,
        n_steps=req.n_steps,
        learning_rate=req.learning_rate,
        ot_batch_size=req.ot_batch_size,
        sinkhorn_epsilon=req.sinkhorn_epsilon,
        mu0_mode=req.mu0_mode,
    )
    return {
        "status": "trained",
        "n_samples": req.n_samples,
        "n_steps": req.n_steps,
        "kappa": req.kappa,
        "final_loss": OTTFM_STATE.final_loss,
        "unnormalized_mse": OTTFM_STATE.unnormalized_mse,
        "mean_pair_cost": OTTFM_STATE.mean_pair_cost,
        "ot_batch_size": OTTFM_STATE.ot_batch_size,
        "sinkhorn_epsilon": OTTFM_STATE.sinkhorn_epsilon,
        "mu0_mode": OTTFM_STATE.mu0_mode,
    }


@app.post("/api/generate_ot_tfm", response_model=GenerateOTTFMResponse)
def generate_ot_tfm(req: GenerateOTTFMRequest):
    global OTTFM_STATE
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)

    if OTTFM_STATE is None:
        OTTFM_STATE = train_ottfm_vector_field(
            jax.random.PRNGKey(0),
            L1,
            float(req.kappa),
            n_samples=2048,
            n_steps=2000,
            learning_rate=2e-3,
            ot_batch_size=64,
            sinkhorn_epsilon=0.75,
            mu0_mode="heat_gp",
        )

    source, generated, target_ref = generate_ottfm_sample(
        jax.random.PRNGKey(req.seed),
        OTTFM_STATE,
        n_steps=req.rollout_steps,
        control_scale=req.control_scale,
    )
    sources_b, generated_b, target_refs_b = generate_ottfm_samples(
        jax.random.PRNGKey(req.seed + 20_000),
        OTTFM_STATE,
        n_eval=req.n_eval,
        n_steps=req.rollout_steps,
        control_scale=req.control_scale,
    )

    metrics = compare_generated_to_reference(source, generated, target_ref, OTTFM_STATE.evals, OTTFM_STATE.U)
    aggregate_metrics = compare_generated_batches_to_reference(sources_b, generated_b, target_refs_b, OTTFM_STATE.evals, OTTFM_STATE.U)
    distances = distribution_distance_summary(generated_b, target_refs_b, OTTFM_STATE.U, seed=req.seed + 200)

    return {
        "nodes": annulus_layout(n=18),
        "edges": [list(e) for e in edges],
        "faces": [list(f) for f in faces],
        "source_signal": to_list(source),
        "generated_signal": to_list(generated),
        "target_reference_signal": to_list(target_ref),
        "trained_final_loss": OTTFM_STATE.final_loss,
        "trained_unnormalized_mse": OTTFM_STATE.unnormalized_mse,
        "mean_pair_cost": OTTFM_STATE.mean_pair_cost,
        "metrics": metrics,
        "aggregate_metrics": aggregate_metrics,
        "distances": distances,
        "n_eval": req.n_eval,
        "rollout_steps": req.rollout_steps,
        "control_scale": req.control_scale,
        "ot_batch_size": OTTFM_STATE.ot_batch_size,
        "sinkhorn_epsilon": OTTFM_STATE.sinkhorn_epsilon,
        "mu0_mode": OTTFM_STATE.mu0_mode,
    }


@app.post("/api/run_mu0_ablation", response_model=Mu0AblationResponse)
def run_initial_distribution_ablation(req: Mu0AblationRequest):
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    results = run_mu0_ablation(
        jax.random.PRNGKey(req.seed),
        L1,
        float(req.kappa),
        n_samples=req.n_samples,
        n_steps=req.n_steps,
        learning_rate=req.learning_rate,
        n_eval=req.n_eval,
        rollout_steps=req.rollout_steps,
        control_scale=req.control_scale,
        ot_batch_size=req.ot_batch_size,
        sinkhorn_epsilon=req.sinkhorn_epsilon,
    )
    return {
        "n_samples": req.n_samples,
        "n_steps": req.n_steps,
        "n_eval": req.n_eval,
        "rollout_steps": req.rollout_steps,
        "control_scale": req.control_scale,
        "results": results,
    }


@app.post("/api/run_distribution_metrics", response_model=DistributionMetricEvalResponse)
def run_distribution_metrics(req: DistributionMetricEvalRequest):
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    results = run_distribution_metric_eval(
        jax.random.PRNGKey(req.seed),
        L1,
        float(req.kappa),
        n_samples=req.n_samples,
        n_steps=req.n_steps,
        learning_rate=req.learning_rate,
        n_eval=req.n_eval,
        rollout_steps=req.rollout_steps,
        cfm_control_scale=req.cfm_control_scale,
        tfm_control_scale=req.tfm_control_scale,
        ot_batch_size=req.ot_batch_size,
        sinkhorn_epsilon=req.sinkhorn_epsilon,
    )
    return {
        "n_samples": req.n_samples,
        "n_steps": req.n_steps,
        "n_eval": req.n_eval,
        "rollout_steps": req.rollout_steps,
        "results": results,
    }


@app.post("/api/run_kappa_sweep", response_model=KappaSweepResponse)
def run_kappa_sweep_endpoint(req: KappaSweepRequest):
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    rows = run_kappa_sweep(
        jax.random.PRNGKey(req.seed),
        L1,
        req.kappas,
        n_samples=req.n_samples,
        n_steps=req.n_steps,
        learning_rate=req.learning_rate,
        n_eval=req.n_eval,
        rollout_steps=req.rollout_steps,
        control_scale=req.control_scale,
        ot_batch_size=req.ot_batch_size,
        sinkhorn_epsilon=req.sinkhorn_epsilon,
        mu0_mode=req.mu0_mode,
    )
    return {
        "kappas": req.kappas,
        "n_samples": req.n_samples,
        "n_steps": req.n_steps,
        "n_eval": req.n_eval,
        "rollout_steps": req.rollout_steps,
        "control_scale": req.control_scale,
        "mu0_mode": req.mu0_mode,
        "rows": rows,
    }


@app.post("/api/ot_coupling_visualization", response_model=OTCouplingVisualizationResponse)
def ot_coupling_visualization(req: OTCouplingVisualizationRequest):
    nodes, edges, faces, B1, B2, L1 = build_annulus_operators(req.with_face)
    evals, U = jnp.linalg.eigh(L1)

    return build_ot_coupling_visualization(
        jax.random.PRNGKey(req.seed),
        evals,
        U,
        kappa=float(req.kappa),
        method=req.method,
        batch_size=req.batch_size,
        sinkhorn_epsilon=req.sinkhorn_epsilon,
        mu0_mode=req.mu0_mode,
        top_k=req.top_k,
    )
