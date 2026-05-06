export type BridgeMode = "neural" | "analytical";
export type Mu0Mode = "standard" | "heat_gp";

export type Preset = {
  name: string;
  description: string;
  x0: number[];
  x1: number[];
};

export type ComputeRequest = {
  with_face: boolean;
  kappa: number;
  t: number;
  x0: number[];
  x1: number[];
  bridge_mode: BridgeMode;
  nn_steps: number;
};

export type FigureOnePayload = {
  labels: string[];
  nodes: number[][];
  edges: number[][];
  faces: number[][];
  graph_signals: number[][];
  edge_signals: number[][];
  graph_eigenvalues: number[];
  edge_eigenvalues: number[];
};

export type ComputeResponse = {
  nodes: number[][];
  edges: number[][];
  faces: number[][];
  B1: number[][];
  B2: number[][];
  B1B2: number[][];
  L1: number[][];
  eigenvalues: number[];
  eigenvectors: number[][];
  cfm_xt: number[];
  tfm_xt: number[];
  cfm_u: number[];
  tfm_u: number[];
  analytical_tfm_u: number[];
  cfm_cost: number;
  tfm_cost: number;
  bridge_mode: BridgeMode;
  nn_loss: number | null;
  nn_unnormalized_mse: number | null;
  nn_steps: number | null;
  figure1: FigureOnePayload;
};

export type SpectralMetrics = {
  harmonic_energy: number;
  low_frequency_energy: number;
  high_frequency_energy: number;
  hodge_energy: number;
  l2_norm: number;
};

export type MetricMeanStd = {
  mean: number;
  std: number;
};

export type SpectralMetricSummary = {
  harmonic_energy: MetricMeanStd;
  low_frequency_energy: MetricMeanStd;
  high_frequency_energy: MetricMeanStd;
  hodge_energy: MetricMeanStd;
  l2_norm: MetricMeanStd;
};

export type DistributionDistances = {
  mmd_rbf: number;
  spectral_sliced_wasserstein: number;
  spectral_mode_wasserstein: number;
};

export type TrainITFMRequest = {
  with_face: boolean;
  kappa: number;
  n_samples: number;
  n_steps: number;
  learning_rate: number;
  seed: number;
  mu0_mode?: Mu0Mode;
};

export type TrainITFMResponse = {
  status: string;
  n_samples: number;
  n_steps: number;
  kappa: number;
  final_loss: number;
  unnormalized_mse: number;
  mu0_mode?: Mu0Mode;
};

export type GenerateITFMRequest = {
  with_face: boolean;
  kappa: number;
  rollout_steps: number;
  seed: number;
  n_eval: number;
  control_scale: number;
};

export type GenerateITFMResponse = {
  nodes: number[][];
  edges: number[][];
  faces: number[][];
  source_signal: number[];
  generated_signal: number[];
  target_reference_signal: number[];
  trained_final_loss: number;
  trained_unnormalized_mse: number;
  metrics: {
    source: SpectralMetrics;
    generated: SpectralMetrics;
    reference: SpectralMetrics;
  };
  aggregate_metrics: {
    source: SpectralMetricSummary;
    generated: SpectralMetricSummary;
    reference: SpectralMetricSummary;
  };
  distances?: DistributionDistances;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  mu0_mode?: Mu0Mode;
};

export type TrainOTTFMRequest = {
  with_face: boolean;
  kappa: number;
  n_samples: number;
  n_steps: number;
  learning_rate: number;
  seed: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
  mu0_mode?: Mu0Mode;
};

export type TrainOTTFMResponse = {
  status: string;
  n_samples: number;
  n_steps: number;
  kappa: number;
  final_loss: number;
  unnormalized_mse: number;
  mean_pair_cost: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
  mu0_mode?: Mu0Mode;
};

export type GenerateOTTFMRequest = {
  with_face: boolean;
  kappa: number;
  rollout_steps: number;
  seed: number;
  n_eval: number;
  control_scale: number;
};

export type GenerateOTTFMResponse = {
  nodes: number[][];
  edges: number[][];
  faces: number[][];
  source_signal: number[];
  generated_signal: number[];
  target_reference_signal: number[];
  trained_final_loss: number;
  trained_unnormalized_mse: number;
  mean_pair_cost: number;
  metrics: {
    source: SpectralMetrics;
    generated: SpectralMetrics;
    reference: SpectralMetrics;
  };
  aggregate_metrics: {
    source: SpectralMetricSummary;
    generated: SpectralMetricSummary;
    reference: SpectralMetricSummary;
  };
  distances?: DistributionDistances;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
  mu0_mode?: Mu0Mode;
};

export type CFMCoupling = "independent" | "ot";

export type TrainCFMRequest = {
  with_face: boolean;
  kappa: number;
  coupling: CFMCoupling;
  n_samples: number;
  n_steps: number;
  learning_rate: number;
  seed: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
};

export type TrainCFMResponse = {
  status: string;
  coupling: CFMCoupling;
  n_samples: number;
  n_steps: number;
  kappa: number;
  final_loss: number;
  unnormalized_mse: number;
  mean_pair_cost: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
};

export type GenerateCFMRequest = {
  with_face: boolean;
  kappa: number;
  coupling: CFMCoupling;
  rollout_steps: number;
  seed: number;
  n_eval: number;
  control_scale: number;
};

export type GenerateCFMResponse = {
  nodes: number[][];
  edges: number[][];
  faces: number[][];
  source_signal: number[];
  generated_signal: number[];
  target_reference_signal: number[];
  trained_final_loss: number;
  trained_unnormalized_mse: number;
  mean_pair_cost: number;
  metrics: {
    source: SpectralMetrics;
    generated: SpectralMetrics;
    reference: SpectralMetrics;
  };
  aggregate_metrics: {
    source: SpectralMetricSummary;
    generated: SpectralMetricSummary;
    reference: SpectralMetricSummary;
  };
  distances?: DistributionDistances;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  coupling: CFMCoupling;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
};

export type Mu0AblationRequest = {
  with_face: boolean;
  kappa: number;
  n_samples: number;
  n_steps: number;
  learning_rate: number;
  seed: number;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
};

export type Mu0AblationMethodResult = {
  method: "itfm" | "ottfm";
  mu0_mode: Mu0Mode;
  final_loss: number;
  unnormalized_mse: number;
  mean_pair_cost: number | null;
  aggregate_metrics: {
    source: SpectralMetricSummary;
    generated: SpectralMetricSummary;
    reference: SpectralMetricSummary;
  };
};

export type Mu0AblationResponse = {
  n_samples: number;
  n_steps: number;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  results: Record<string, Mu0AblationMethodResult>;
};

export type DistributionMetricEvalRequest = {
  with_face: boolean;
  kappa: number;
  n_samples: number;
  n_steps: number;
  learning_rate: number;
  seed: number;
  n_eval: number;
  rollout_steps: number;
  cfm_control_scale: number;
  tfm_control_scale: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
};

export type DistributionMetricMethodResult = {
  method: "icfm" | "otcfm" | "itfm" | "ottfm";
  final_loss: number;
  unnormalized_mse: number;
  mean_pair_cost: number | null;
  aggregate_metrics: {
    source: SpectralMetricSummary;
    generated: SpectralMetricSummary;
    reference: SpectralMetricSummary;
  };
  distances: DistributionDistances;
};

export type DistributionMetricEvalResponse = {
  n_samples: number;
  n_steps: number;
  n_eval: number;
  rollout_steps: number;
  results: Record<string, DistributionMetricMethodResult>;
};

export type KappaSweepRequest = {
  with_face: boolean;
  kappas: number[];
  n_samples: number;
  n_steps: number;
  learning_rate: number;
  seed: number;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  ot_batch_size: number;
  sinkhorn_epsilon: number;
  mu0_mode: Mu0Mode;
};

export type KappaSweepRow = {
  kappa: number;
  method: "I-TFM" | "OT-TFM";
  final_loss: number;
  unnormalized_mse: number;
  mean_pair_cost: number | null;
  aggregate_metrics: {
    source: SpectralMetricSummary;
    generated: SpectralMetricSummary;
    reference: SpectralMetricSummary;
  };
  distances: DistributionDistances;
};

export type KappaSweepResponse = {
  kappas: number[];
  n_samples: number;
  n_steps: number;
  n_eval: number;
  rollout_steps: number;
  control_scale: number;
  mu0_mode: Mu0Mode;
  rows: KappaSweepRow[];
};

export type CouplingMethod = "ot_cfm" | "ot_tfm";

export type CouplingPair = {
  source_index: number;
  target_index: number;
  plan_mass: number;
  cost: number;
};

export type CouplingSummary = {
  expected_cost: number;
  plan_entropy: number;
  mean_row_entropy: number;
  mean_row_max_mass: number;
  mean_col_max_mass: number;
  min_cost: number;
  max_cost: number;
  mean_cost: number;
};

export type OTCouplingVisualizationRequest = {
  with_face: boolean;
  kappa: number;
  method: CouplingMethod;
  batch_size: number;
  sinkhorn_epsilon: number;
  seed: number;
  mu0_mode: Mu0Mode;
  top_k: number;
};

export type OTCouplingVisualizationResponse = {
  method: CouplingMethod;
  label: string;
  batch_size: number;
  sinkhorn_epsilon: number;
  mu0_mode: Mu0Mode;
  kappa: number;
  cost_matrix: number[][];
  plan_matrix: number[][];
  sampled_pairs: CouplingPair[];
  top_pairs: CouplingPair[];
  row_argmax_pairs: CouplingPair[];
  summary: CouplingSummary;
};
