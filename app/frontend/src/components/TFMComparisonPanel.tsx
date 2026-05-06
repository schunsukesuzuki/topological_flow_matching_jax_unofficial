import { useState } from "react";
import { generateITFM, generateOTTFM, trainITFM, trainOTTFM } from "../api";
import type {
  GenerateITFMResponse,
  GenerateOTTFMResponse,
  SpectralMetricSummary,
  SpectralMetrics,
  TrainITFMResponse,
  TrainOTTFMResponse,
} from "../types";

type Props = {
  withFace: boolean;
  kappa: number;
};

type ComparisonState = {
  itfmTrain: TrainITFMResponse;
  otTrain: TrainOTTFMResponse;
  itfmGen: GenerateITFMResponse;
  otGen: GenerateOTTFMResponse;
};

function fmt(x: number) {
  if (!Number.isFinite(x)) return "NaN";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.001) return x.toExponential(3);
  return x.toFixed(4);
}

function meanStd(x: { mean: number; std: number }) {
  return `${fmt(x.mean)} ± ${fmt(x.std)}`;
}

function absGap(a: number, b: number) {
  return Math.abs(a - b);
}

function winner(
  itfm: SpectralMetricSummary[keyof SpectralMetricSummary],
  ot: SpectralMetricSummary[keyof SpectralMetricSummary],
  ref: SpectralMetricSummary[keyof SpectralMetricSummary],
) {
  const itGap = absGap(itfm.mean, ref.mean);
  const otGap = absGap(ot.mean, ref.mean);

  if (Math.abs(itGap - otGap) < 1e-9) return "tie";
  return otGap < itGap ? "OT-TFM" : "I-TFM";
}

const rows: Array<[keyof SpectralMetrics, string]> = [
  ["harmonic_energy", "harmonic energy"],
  ["low_frequency_energy", "low-frequency energy"],
  ["high_frequency_energy", "high-frequency energy"],
  ["hodge_energy", "xᵀL₁x"],
  ["l2_norm", "||x||²"],
];

export function TFMComparisonPanel({ withFace, kappa }: Props) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<ComparisonState | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRunComparison() {
    setRunning(true);
    setError(null);

    try {
      const commonTrain = {
        with_face: withFace,
        kappa,
        n_samples: 2048,
        n_steps: 2000,
        learning_rate: 0.002,
      };

      const commonGenerate = {
        with_face: withFace,
        kappa,
        rollout_steps: 160,
        n_eval: 32,
        control_scale: 0.92,
      };

      const itfmTrain = await trainITFM({
        ...commonTrain,
        seed: 0,
      });

      const otTrain = await trainOTTFM({
        ...commonTrain,
        seed: 11,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
      });

      const itfmGen = await generateITFM({
        ...commonGenerate,
        seed: 1,
      });

      const otGen = await generateOTTFM({
        ...commonGenerate,
        seed: 7,
      });

      setResult({ itfmTrain, otTrain, itfmGen, otGen });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="card wide comparison-card">
      <h3>I-TFM vs OT-TFM distribution comparison</h3>
      <p className="muted">
        This panel trains both I-TFM and OT-TFM with the same high-level settings, then compares
        generated distribution statistics against the same target distribution μ1. The key question is
        whether OT-biased coupling moves generated samples closer to μ1 than independent coupling.
      </p>

      <div className="button-row">
        <button onClick={handleRunComparison} disabled={running}>
          {running ? "Running comparison..." : "Run I-TFM vs OT-TFM comparison"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <p className="muted">
            I-TFM: normalized MSE = {result.itfmTrain.final_loss.toExponential(3)},
            displayed-scale MSE = {result.itfmTrain.unnormalized_mse.toExponential(3)}.
            OT-TFM: normalized MSE = {result.otTrain.final_loss.toExponential(3)},
            displayed-scale MSE = {result.otTrain.unnormalized_mse.toExponential(3)},
            mean pair cost = {fmt(result.otTrain.mean_pair_cost)}.
          </p>

          <div className="table-scroll">
            <table className="matrix metric-table comparison-table">
              <thead>
                <tr>
                  <th>metric</th>
                  <th>I-TFM generated mean ± std</th>
                  <th>OT-TFM generated mean ± std</th>
                  <th>reference μ1 mean ± std</th>
                  <th>closer to μ1</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([key, label]) => {
                  const it = result.itfmGen.aggregate_metrics.generated[key];
                  const ot = result.otGen.aggregate_metrics.generated[key];
                  const ref = result.otGen.aggregate_metrics.reference[key];
                  const win = winner(it, ot, ref);

                  return (
                    <tr key={key}>
                      <th>{label}</th>
                      <td>{meanStd(it)}</td>
                      <td>{meanStd(ot)}</td>
                      <td>{meanStd(ref)}</td>
                      <td className={win === "OT-TFM" ? "winner-ot" : win === "I-TFM" ? "winner-it" : ""}>
                        {win}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="muted">
            The winner column compares absolute distance between each generated mean and the μ1 mean
            for each metric. It is intentionally simple and diagnostic, not a formal statistical test.
          </p>
        </>
      )}
    </div>
  );
}
