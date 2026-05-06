import type {
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
  Preset,
  TrainCFMRequest,
  TrainCFMResponse,
  TrainITFMRequest,
  TrainITFMResponse,
  TrainOTTFMRequest,
  TrainOTTFMResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function fetchPresets(): Promise<Preset[]> {
  const res = await fetch(`${API_BASE}/api/presets`);
  if (!res.ok) throw new Error(`Failed to fetch presets: ${res.status}`);
  const data = await res.json();
  return data.presets;
}

export async function compute(req: ComputeRequest): Promise<ComputeResponse> {
  const res = await fetch(`${API_BASE}/api/compute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to compute: ${res.status}`);
  return res.json();
}

export async function trainCFM(req: TrainCFMRequest): Promise<TrainCFMResponse> {
  const res = await fetch(`${API_BASE}/api/train_cfm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to train CFM baseline: ${res.status}`);
  return res.json();
}

export async function generateCFM(req: GenerateCFMRequest): Promise<GenerateCFMResponse> {
  const res = await fetch(`${API_BASE}/api/generate_cfm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to generate CFM sample: ${res.status}`);
  return res.json();
}

export async function trainITFM(req: TrainITFMRequest): Promise<TrainITFMResponse> {
  const res = await fetch(`${API_BASE}/api/train_itfm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to train I-TFM: ${res.status}`);
  return res.json();
}

export async function generateITFM(req: GenerateITFMRequest): Promise<GenerateITFMResponse> {
  const res = await fetch(`${API_BASE}/api/generate_itfm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to generate I-TFM sample: ${res.status}`);
  return res.json();
}

export async function trainOTTFM(req: TrainOTTFMRequest): Promise<TrainOTTFMResponse> {
  const res = await fetch(`${API_BASE}/api/train_ot_tfm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to train OT-TFM: ${res.status}`);
  return res.json();
}

export async function generateOTTFM(req: GenerateOTTFMRequest): Promise<GenerateOTTFMResponse> {
  const res = await fetch(`${API_BASE}/api/generate_ot_tfm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to generate OT-TFM sample: ${res.status}`);
  return res.json();
}

export async function runMu0Ablation(req: Mu0AblationRequest): Promise<Mu0AblationResponse> {
  const res = await fetch(`${API_BASE}/api/run_mu0_ablation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to run μ0 ablation: ${res.status}`);
  return res.json();
}

export async function runDistributionMetrics(
  req: DistributionMetricEvalRequest,
): Promise<DistributionMetricEvalResponse> {
  const res = await fetch(`${API_BASE}/api/run_distribution_metrics`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to run distribution metrics: ${res.status}`);
  return res.json();
}

export async function runKappaSweep(req: KappaSweepRequest): Promise<KappaSweepResponse> {
  const res = await fetch(`${API_BASE}/api/run_kappa_sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to run κ sweep: ${res.status}`);
  return res.json();
}

export async function fetchOTCouplingVisualization(
  req: OTCouplingVisualizationRequest,
): Promise<OTCouplingVisualizationResponse> {
  const res = await fetch(`${API_BASE}/api/ot_coupling_visualization`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Failed to fetch OT coupling visualization: ${res.status}`);
  return res.json();
}
