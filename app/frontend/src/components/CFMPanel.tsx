import { useState } from "react";
import { generateCFM, trainCFM } from "../api";
import type {
  CFMCoupling,
  GenerateCFMResponse,
  SpectralMetrics,
  SpectralMetricSummary,
  TrainCFMResponse,
} from "../types";
import { ComplexView } from "./ComplexView";

type Props = {
  withFace: boolean;
  kappa: number;
};

function fmt(x: number) {
  if (!Number.isFinite(x)) return "NaN";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.001) return x.toExponential(3);
  return x.toFixed(4);
}

function fmtMeanStd(v: { mean: number; std: number }) {
  return `${fmt(v.mean)} ± ${fmt(v.std)}`;
}

const rows: Array<[keyof SpectralMetrics, string]> = [
  ["harmonic_energy", "harmonic energy"],
  ["low_frequency_energy", "low-frequency energy"],
  ["high_frequency_energy", "high-frequency energy"],
  ["hodge_energy", "xᵀL₁x"],
  ["l2_norm", "||x||²"],
];

function MetricTable({ metrics }: { metrics: GenerateCFMResponse["metrics"] }) {
  return (
    <div className="table-scroll">
      <table className="matrix metric-table">
        <thead>
          <tr>
            <th>metric</th>
            <th>source μ0</th>
            <th>generated x̂1</th>
            <th>reference μ1</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => (
            <tr key={key}>
              <th>{label}</th>
              <td>{fmt(metrics.source[key])}</td>
              <td>{fmt(metrics.generated[key])}</td>
              <td>{fmt(metrics.reference[key])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AggregateMetricTable({ metrics }: { metrics: GenerateCFMResponse["aggregate_metrics"] }) {
  return (
    <div className="table-scroll">
      <table className="matrix metric-table">
        <thead>
          <tr>
            <th>metric</th>
            <th>source μ0 mean ± std</th>
            <th>generated mean ± std</th>
            <th>reference μ1 mean ± std</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => (
            <tr key={key}>
              <th>{label}</th>
              <td>{fmtMeanStd(metrics.source[key as keyof SpectralMetricSummary])}</td>
              <td>{fmtMeanStd(metrics.generated[key as keyof SpectralMetricSummary])}</td>
              <td>{fmtMeanStd(metrics.reference[key as keyof SpectralMetricSummary])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function couplingLabel(coupling: CFMCoupling) {
  return coupling === "independent" ? "I-CFM" : "OT-CFM";
}

export function CFMPanel({ withFace, kappa }: Props) {
  const [coupling, setCoupling] = useState<CFMCoupling>("independent");
  const [training, setTraining] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainCFMResponse | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateCFMResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleTrain() {
    setTraining(true);
    setError(null);
    try {
      const result = await trainCFM({
        with_face: withFace,
        kappa,
        coupling,
        n_samples: 2048,
        n_steps: 2000,
        learning_rate: 0.002,
        seed: coupling === "independent" ? 21 : 22,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
      });
      setTrainResult(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setTraining(false);
    }
  }

  async function handleGenerate() {
    setGenerating(true);
    setError(null);
    try {
      const result = await generateCFM({
        with_face: withFace,
        kappa,
        coupling,
        rollout_steps: 160,
        seed: coupling === "independent" ? 31 : 32,
        n_eval: 32,
        control_scale: 1.0,
      });
      setGenerateResult(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setGenerating(false);
    }
  }

  const label = couplingLabel(coupling);

  return (
    <div className="card wide cfm-baseline-card">
      <h3>I-CFM / OT-CFM distribution-level baselines</h3>
      <p className="muted">
        This trains ordinary CFM baselines at the same distribution level as I-TFM / OT-TFM.
        CFM uses straight interpolation x_t = (1 - t)x0 + tx1 and target velocity x1 - x0.
        Generation uses dx/dt = uθ(t, x), without the Hodge heat drift -κLx.
      </p>

      <label>
        CFM coupling
        <select value={coupling} onChange={(e) => setCoupling(e.target.value as CFMCoupling)}>
          <option value="independent">I-CFM: independent μ0 ⊗ μ1 pairs</option>
          <option value="ot">OT-CFM: Euclidean Sinkhorn coupling</option>
        </select>
      </label>

      <div className="button-row">
        <button onClick={handleTrain} disabled={training}>
          {training ? `Training ${label}...` : `Train ${label}`}
        </button>
        <button onClick={handleGenerate} disabled={generating}>
          {generating ? "Generating..." : `Generate ${label} sample`}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {trainResult && (
        <p className="muted">
          {label}: trained on {trainResult.n_samples} pairs for {trainResult.n_steps} steps.
          normalized MSE = {trainResult.final_loss.toExponential(3)},
          displayed-scale MSE = {trainResult.unnormalized_mse.toExponential(3)},
          mean pair cost = {fmt(trainResult.mean_pair_cost)}.
        </p>
      )}

      {generateResult && (
        <>
          <p className="muted">
            Generated by CFM rollout with dx/dt = uθ(t, x). Rollout steps =
            {generateResult.rollout_steps}; distribution diagnostics use n_eval =
            {generateResult.n_eval}. Coupling = {generateResult.coupling}.
          </p>

          <div className="itfm-generated-grid">
            <ComplexView
              title={`${label} source x0 ~ μ0`}
              nodes={generateResult.nodes}
              edges={generateResult.edges}
              faces={generateResult.faces}
              signal={generateResult.source_signal}
            />
            <ComplexView
              title={`${label} generated x̂1`}
              nodes={generateResult.nodes}
              edges={generateResult.edges}
              faces={generateResult.faces}
              signal={generateResult.generated_signal}
            />
            <ComplexView
              title="Reference target x1 ~ μ1"
              nodes={generateResult.nodes}
              edges={generateResult.edges}
              faces={generateResult.faces}
              signal={generateResult.target_reference_signal}
            />
          </div>

          <h4>{label} spectral diagnostics: single displayed samples</h4>
          <MetricTable metrics={generateResult.metrics} />

          <h4>{label} spectral diagnostics: distribution mean over n_eval</h4>
          <AggregateMetricTable metrics={generateResult.aggregate_metrics} />
        </>
      )}
    </div>
  );
}
