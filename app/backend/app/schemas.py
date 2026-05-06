from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal


class Preset(BaseModel):
    name: str
    description: str
    x0: List[float]
    x1: List[float]


class PresetsResponse(BaseModel):
    presets: List[Preset]


class ComputeRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    t: float = Field(default=0.5, ge=0.0, le=1.0)
    x0: List[float]
    x1: List[float]
    bridge_mode: Literal["neural", "analytical"] = "neural"
    nn_steps: int = Field(default=1500, ge=50, le=5000)


class FigureOnePayload(BaseModel):
    labels: List[str]
    nodes: List[List[float]]
    edges: List[List[int]]
    faces: List[List[int]]
    graph_signals: List[List[float]]
    edge_signals: List[List[float]]
    graph_eigenvalues: List[float]
    edge_eigenvalues: List[float]


class ComputeResponse(BaseModel):
    nodes: List[List[float]]
    edges: List[List[int]]
    faces: List[List[int]]
    B1: List[List[float]]
    B2: List[List[float]]
    B1B2: List[List[float]]
    L1: List[List[float]]
    eigenvalues: List[float]
    eigenvectors: List[List[float]]
    cfm_xt: List[float]
    tfm_xt: List[float]
    cfm_u: List[float]
    tfm_u: List[float]
    analytical_tfm_u: List[float]
    cfm_cost: float
    tfm_cost: float
    bridge_mode: Literal["neural", "analytical"]
    nn_loss: Optional[float] = None
    nn_unnormalized_mse: Optional[float] = None
    nn_steps: Optional[int] = None
    figure1: FigureOnePayload


MetricValue = Dict[str, float]
MetricSummary = Dict[str, MetricValue]
DistributionDistances = Dict[str, float]


class TrainITFMRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    n_samples: int = Field(default=2048, ge=128, le=8192)
    n_steps: int = Field(default=2000, ge=100, le=10000)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1e-1)
    seed: int = 0
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class TrainITFMResponse(BaseModel):
    status: str
    n_samples: int
    n_steps: int
    kappa: float
    final_loss: float
    unnormalized_mse: float
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class GenerateITFMRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    rollout_steps: int = Field(default=160, ge=20, le=1000)
    seed: int = 1
    n_eval: int = Field(default=32, ge=1, le=256)
    control_scale: float = Field(default=0.92, gt=0.0, le=1.5)


class GenerateITFMResponse(BaseModel):
    nodes: List[List[float]]
    edges: List[List[int]]
    faces: List[List[int]]
    source_signal: List[float]
    generated_signal: List[float]
    target_reference_signal: List[float]
    trained_final_loss: float
    trained_unnormalized_mse: float
    metrics: Dict[str, Dict[str, float]]
    aggregate_metrics: Dict[str, MetricSummary]
    distances: Optional[DistributionDistances] = None
    n_eval: int
    rollout_steps: int
    control_scale: float
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class TrainOTTFMRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    n_samples: int = Field(default=2048, ge=128, le=8192)
    n_steps: int = Field(default=2000, ge=100, le=10000)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1e-1)
    seed: int = 11
    ot_batch_size: int = Field(default=64, ge=16, le=256)
    sinkhorn_epsilon: float = Field(default=0.75, gt=0.01, le=20.0)
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class TrainOTTFMResponse(BaseModel):
    status: str
    n_samples: int
    n_steps: int
    kappa: float
    final_loss: float
    unnormalized_mse: float
    mean_pair_cost: float
    ot_batch_size: int
    sinkhorn_epsilon: float
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class GenerateOTTFMRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    rollout_steps: int = Field(default=160, ge=20, le=1000)
    seed: int = 7
    n_eval: int = Field(default=32, ge=1, le=256)
    control_scale: float = Field(default=0.92, gt=0.0, le=1.5)


class GenerateOTTFMResponse(BaseModel):
    nodes: List[List[float]]
    edges: List[List[int]]
    faces: List[List[int]]
    source_signal: List[float]
    generated_signal: List[float]
    target_reference_signal: List[float]
    trained_final_loss: float
    trained_unnormalized_mse: float
    mean_pair_cost: float
    metrics: Dict[str, Dict[str, float]]
    aggregate_metrics: Dict[str, MetricSummary]
    distances: Optional[DistributionDistances] = None
    n_eval: int
    rollout_steps: int
    control_scale: float
    ot_batch_size: int
    sinkhorn_epsilon: float
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class TrainCFMRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    coupling: Literal["independent", "ot"] = "independent"
    n_samples: int = Field(default=2048, ge=128, le=8192)
    n_steps: int = Field(default=2000, ge=100, le=10000)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1e-1)
    seed: int = 21
    ot_batch_size: int = Field(default=64, ge=16, le=256)
    sinkhorn_epsilon: float = Field(default=0.75, gt=0.01, le=20.0)


class TrainCFMResponse(BaseModel):
    status: str
    coupling: Literal["independent", "ot"]
    n_samples: int
    n_steps: int
    kappa: float
    final_loss: float
    unnormalized_mse: float
    mean_pair_cost: float
    ot_batch_size: int
    sinkhorn_epsilon: float


class GenerateCFMRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    coupling: Literal["independent", "ot"] = "independent"
    rollout_steps: int = Field(default=160, ge=20, le=1000)
    seed: int = 31
    n_eval: int = Field(default=32, ge=1, le=256)
    control_scale: float = Field(default=1.0, gt=0.0, le=1.5)


class GenerateCFMResponse(BaseModel):
    nodes: List[List[float]]
    edges: List[List[int]]
    faces: List[List[int]]
    source_signal: List[float]
    generated_signal: List[float]
    target_reference_signal: List[float]
    trained_final_loss: float
    trained_unnormalized_mse: float
    mean_pair_cost: float
    metrics: Dict[str, Dict[str, float]]
    aggregate_metrics: Dict[str, MetricSummary]
    distances: Optional[DistributionDistances] = None
    n_eval: int
    rollout_steps: int
    control_scale: float
    coupling: Literal["independent", "ot"]
    ot_batch_size: int
    sinkhorn_epsilon: float


class Mu0AblationRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    n_samples: int = Field(default=1024, ge=128, le=4096)
    n_steps: int = Field(default=1200, ge=100, le=5000)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1e-1)
    seed: int = 101
    n_eval: int = Field(default=32, ge=1, le=256)
    rollout_steps: int = Field(default=160, ge=20, le=1000)
    control_scale: float = Field(default=0.92, gt=0.0, le=1.5)
    ot_batch_size: int = Field(default=64, ge=16, le=256)
    sinkhorn_epsilon: float = Field(default=0.75, gt=0.01, le=20.0)


class Mu0AblationMethodResult(BaseModel):
    method: Literal["itfm", "ottfm"]
    mu0_mode: Literal["standard", "heat_gp"]
    final_loss: float
    unnormalized_mse: float
    mean_pair_cost: Optional[float] = None
    aggregate_metrics: Dict[str, MetricSummary]


class Mu0AblationResponse(BaseModel):
    n_samples: int
    n_steps: int
    n_eval: int
    rollout_steps: int
    control_scale: float
    results: Dict[str, Mu0AblationMethodResult]


class DistributionMetricEvalRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    n_samples: int = Field(default=1024, ge=128, le=4096)
    n_steps: int = Field(default=1200, ge=100, le=5000)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1e-1)
    seed: int = 202
    n_eval: int = Field(default=32, ge=8, le=256)
    rollout_steps: int = Field(default=160, ge=20, le=1000)
    cfm_control_scale: float = Field(default=1.0, gt=0.0, le=1.5)
    tfm_control_scale: float = Field(default=0.92, gt=0.0, le=1.5)
    ot_batch_size: int = Field(default=64, ge=16, le=256)
    sinkhorn_epsilon: float = Field(default=0.75, gt=0.01, le=20.0)


class DistributionMetricMethodResult(BaseModel):
    method: Literal["icfm", "otcfm", "itfm", "ottfm"]
    final_loss: float
    unnormalized_mse: float
    mean_pair_cost: Optional[float] = None
    aggregate_metrics: Dict[str, MetricSummary]
    distances: DistributionDistances


class DistributionMetricEvalResponse(BaseModel):
    n_samples: int
    n_steps: int
    n_eval: int
    rollout_steps: int
    results: Dict[str, DistributionMetricMethodResult]


class KappaSweepRequest(BaseModel):
    with_face: bool = True
    kappas: List[float] = Field(default=[0.25, 0.5, 1.0, 2.0, 4.0])
    n_samples: int = Field(default=768, ge=128, le=4096)
    n_steps: int = Field(default=900, ge=100, le=5000)
    learning_rate: float = Field(default=2e-3, gt=0.0, le=1e-1)
    seed: int = 303
    n_eval: int = Field(default=32, ge=8, le=256)
    rollout_steps: int = Field(default=160, ge=20, le=1000)
    control_scale: float = Field(default=0.92, gt=0.0, le=1.5)
    ot_batch_size: int = Field(default=64, ge=16, le=256)
    sinkhorn_epsilon: float = Field(default=0.75, gt=0.01, le=20.0)
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"


class KappaSweepRow(BaseModel):
    kappa: float
    method: Literal["I-TFM", "OT-TFM"]
    final_loss: float
    unnormalized_mse: float
    mean_pair_cost: Optional[float] = None
    aggregate_metrics: Dict[str, MetricSummary]
    distances: DistributionDistances


class KappaSweepResponse(BaseModel):
    kappas: List[float]
    n_samples: int
    n_steps: int
    n_eval: int
    rollout_steps: int
    control_scale: float
    mu0_mode: Literal["standard", "heat_gp"]
    rows: List[KappaSweepRow]


class CouplingPair(BaseModel):
    source_index: int
    target_index: int
    plan_mass: float
    cost: float


class CouplingSummary(BaseModel):
    expected_cost: float
    plan_entropy: float
    mean_row_entropy: float
    mean_row_max_mass: float
    mean_col_max_mass: float
    min_cost: float
    max_cost: float
    mean_cost: float


class OTCouplingVisualizationRequest(BaseModel):
    with_face: bool = True
    kappa: float = Field(default=2.0, ge=0.0, le=20.0)
    method: Literal["ot_cfm", "ot_tfm"] = "ot_tfm"
    batch_size: int = Field(default=16, ge=4, le=32)
    sinkhorn_epsilon: float = Field(default=0.75, gt=0.01, le=20.0)
    seed: int = 404
    mu0_mode: Literal["standard", "heat_gp"] = "heat_gp"
    top_k: int = Field(default=12, ge=4, le=32)


class OTCouplingVisualizationResponse(BaseModel):
    method: Literal["ot_cfm", "ot_tfm"]
    label: str
    batch_size: int
    sinkhorn_epsilon: float
    mu0_mode: Literal["standard", "heat_gp"]
    kappa: float
    cost_matrix: List[List[float]]
    plan_matrix: List[List[float]]
    sampled_pairs: List[CouplingPair]
    top_pairs: List[CouplingPair]
    row_argmax_pairs: List[CouplingPair]
    summary: CouplingSummary
