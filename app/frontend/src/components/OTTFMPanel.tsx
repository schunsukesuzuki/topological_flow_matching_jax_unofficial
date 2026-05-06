import { useState } from "react";
import { generateOTTFM, trainOTTFM } from "../api";
import type {
  GenerateOTTFMResponse,
  SpectralMetrics,
  SpectralMetricSummary,
  TrainOTTFMResponse,
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

function MetricTable({ metrics }: { metrics: GenerateOTTFMResponse["metrics"] }) {
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

function AggregateMetricTable({ metrics }: { metrics: GenerateOTTFMResponse["aggregate_metrics"] }) {
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

export function OTTFMPanel({ withFace, kappa }: Props) {
  const [training, setTraining] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainOTTFMResponse | null>(null);
  const [generateResult, setGenerateResult] = useState<GenerateOTTFMResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleTrain() {
    setTraining(true);
    setError(null);
    try {
      const result = await trainOTTFM({
        with_face: withFace,
        kappa,
        n_samples: 2048,
        n_steps: 2000,
        learning_rate: 0.002,
        seed: 11,
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
      const result = await generateOTTFM({
        with_face: withFace,
        kappa,
        rollout_steps: 160,
        seed: 7,
        n_eval: 32,
        control_scale: 0.92,
      });
      setGenerateResult(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="card wide ot-tfm-card">
      <h3>OT-TFM distribution-level training</h3>
      <p className="muted">
        This trains uθ(t, x_t, κ) from OT-biased pairs. Each minibatch samples source and target
        pools, computes the TFM transport-cost matrix, builds an entropic Sinkhorn plan, and samples
        training pairs from that plan. As in I-TFM, x1 is not part of the neural input.
      </p>

      <div className="button-row">
        <button onClick={handleTrain} disabled={training}>
          {training ? "Training OT-TFM..." : "Train OT-TFM"}
        </button>
        <button onClick={handleGenerate} disabled={generating}>
          {generating ? "Generating..." : "Generate OT sample"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {trainResult && (
        <p className="muted">
          Trained on {trainResult.n_samples} OT-biased pairs for {trainResult.n_steps} steps.
          normalized MSE = {trainResult.final_loss.toExponential(3)},
          displayed-scale MSE = {trainResult.unnormalized_mse.toExponential(3)},
          mean pair cost = {fmt(trainResult.mean_pair_cost)}.
          minibatch = {trainResult.ot_batch_size}, ε = {trainResult.sinkhorn_epsilon}.
        </p>
      )}

      {generateResult && (
        <>
          <p className="muted">
            Generated by ODE rollout with dx/dt = -κLx + {generateResult.control_scale.toFixed(2)}·uθ(t, x).
            Rollout steps = {generateResult.rollout_steps}; distribution diagnostics use n_eval = {generateResult.n_eval}.
            Mean OT pair cost during training = {fmt(generateResult.mean_pair_cost)}.
          </p>

          <div className="itfm-generated-grid">
            <ComplexView
              title="OT-TFM source x0 ~ μ0"
              nodes={generateResult.nodes}
              edges={generateResult.edges}
              faces={generateResult.faces}
              signal={generateResult.source_signal}
            />
            <ComplexView
              title="OT generated x̂1"
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

          <h4>OT-TFM spectral diagnostics: single displayed samples</h4>
          <p className="muted">
            These are the three samples shown above. Pointwise matching is not expected.
          </p>
          <MetricTable metrics={generateResult.metrics} />

          <h4>OT-TFM spectral diagnostics: distribution mean over n_eval</h4>
          <p className="muted">
            This is the more important comparison. OT-TFM changes the training coupling, so generated
            statistics should be compared distributionally against μ1.
          </p>
          <AggregateMetricTable metrics={generateResult.aggregate_metrics} />
        </>
      )}
    </div>
  );
}
