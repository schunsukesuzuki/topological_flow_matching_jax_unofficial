import { useState } from "react";
import {
  generateCFM,
  generateITFM,
  generateOTTFM,
  trainCFM,
  trainITFM,
  trainOTTFM,
} from "../api";
import type {
  GenerateCFMResponse,
  GenerateITFMResponse,
  GenerateOTTFMResponse,
  SpectralMetricSummary,
  SpectralMetrics,
  TrainCFMResponse,
  TrainITFMResponse,
  TrainOTTFMResponse,
} from "../types";

type Props = {
  withFace: boolean;
  kappa: number;
};

type MethodName = "I-CFM" | "OT-CFM" | "I-TFM" | "OT-TFM";

type FourWayState = {
  icfmTrain: TrainCFMResponse;
  otcfmTrain: TrainCFMResponse;
  itfmTrain: TrainITFMResponse;
  ottfmTrain: TrainOTTFMResponse;
  icfmGen: GenerateCFMResponse;
  otcfmGen: GenerateCFMResponse;
  itfmGen: GenerateITFMResponse;
  ottfmGen: GenerateOTTFMResponse;
};

const rows: Array<[keyof SpectralMetrics, string]> = [
  ["harmonic_energy", "harmonic energy"],
  ["low_frequency_energy", "low-frequency energy"],
  ["high_frequency_energy", "high-frequency energy"],
  ["hodge_energy", "xᵀL₁x"],
  ["l2_norm", "||x||²"],
];

function fmt(x: number) {
  if (!Number.isFinite(x)) return "NaN";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.001) return x.toExponential(3);
  return x.toFixed(4);
}

function fmtMeanStd(v: { mean: number; std: number }) {
  return `${fmt(v.mean)} ± ${fmt(v.std)}`;
}

function absGap(a: number, b: number) {
  return Math.abs(a - b);
}

function pickBest(
  values: Record<MethodName, SpectralMetricSummary[keyof SpectralMetricSummary]>,
  ref: SpectralMetricSummary[keyof SpectralMetricSummary],
): MethodName {
  const methods = Object.keys(values) as MethodName[];
  let best = methods[0];
  let bestGap = absGap(values[best].mean, ref.mean);

  for (const m of methods.slice(1)) {
    const gap = absGap(values[m].mean, ref.mean);
    if (gap < bestGap) {
      best = m;
      bestGap = gap;
    }
  }

  return best;
}

function methodClass(method: MethodName) {
  switch (method) {
    case "I-CFM":
      return "winner-icfm";
    case "OT-CFM":
      return "winner-otcfm";
    case "I-TFM":
      return "winner-it";
    case "OT-TFM":
      return "winner-ot";
    default:
      return "";
  }
}

export function FourWayComparisonPanel({ withFace, kappa }: Props) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<FourWayState | null>(null);
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
      };

      const icfmTrain = await trainCFM({
        ...commonTrain,
        coupling: "independent",
        seed: 21,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
      });

      const otcfmTrain = await trainCFM({
        ...commonTrain,
        coupling: "ot",
        seed: 22,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
      });

      const itfmTrain = await trainITFM({
        ...commonTrain,
        seed: 0,
      });

      const ottfmTrain = await trainOTTFM({
        ...commonTrain,
        seed: 11,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
      });

      const icfmGen = await generateCFM({
        ...commonGenerate,
        coupling: "independent",
        seed: 31,
        control_scale: 1.0,
      });

      const otcfmGen = await generateCFM({
        ...commonGenerate,
        coupling: "ot",
        seed: 32,
        control_scale: 1.0,
      });

      const itfmGen = await generateITFM({
        ...commonGenerate,
        seed: 1,
        control_scale: 0.92,
      });

      const ottfmGen = await generateOTTFM({
        ...commonGenerate,
        seed: 7,
        control_scale: 0.92,
      });

      setResult({
        icfmTrain,
        otcfmTrain,
        itfmTrain,
        ottfmTrain,
        icfmGen,
        otcfmGen,
        itfmGen,
        ottfmGen,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="card wide four-way-comparison-card">
      <h3>Four-way FM comparison: I-CFM / OT-CFM / I-TFM / OT-TFM</h3>
      <p className="muted">
        This panel runs the paper-style comparison structure on the current annulus edge-signal toy
        dataset. CFM methods use straight interpolation and dx/dt = uθ(t, x). TFM methods use the
        Hodge heat drift dx/dt = -κLx + uθ(t, x). OT-CFM uses Euclidean Sinkhorn coupling, while
        OT-TFM uses TFM transport-cost Sinkhorn coupling.
      </p>

      <div className="button-row">
        <button onClick={handleRunComparison} disabled={running}>
          {running ? "Running four-way comparison..." : "Run four-way comparison"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <p className="muted">
            Training MSEs — I-CFM: {result.icfmTrain.unnormalized_mse.toExponential(3)};
            OT-CFM: {result.otcfmTrain.unnormalized_mse.toExponential(3)};
            I-TFM: {result.itfmTrain.unnormalized_mse.toExponential(3)};
            OT-TFM: {result.ottfmTrain.unnormalized_mse.toExponential(3)}.
            OT-CFM mean pair cost = {fmt(result.otcfmTrain.mean_pair_cost)};
            OT-TFM mean pair cost = {fmt(result.ottfmTrain.mean_pair_cost)}.
          </p>

          <div className="table-scroll">
            <table className="matrix metric-table four-way-table">
              <thead>
                <tr>
                  <th>metric</th>
                  <th>I-CFM generated</th>
                  <th>OT-CFM generated</th>
                  <th>I-TFM generated</th>
                  <th>OT-TFM generated</th>
                  <th>reference μ1</th>
                  <th>best</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([key, label]) => {
                  const values: Record<MethodName, SpectralMetricSummary[keyof SpectralMetricSummary]> = {
                    "I-CFM": result.icfmGen.aggregate_metrics.generated[key],
                    "OT-CFM": result.otcfmGen.aggregate_metrics.generated[key],
                    "I-TFM": result.itfmGen.aggregate_metrics.generated[key],
                    "OT-TFM": result.ottfmGen.aggregate_metrics.generated[key],
                  };
                  const ref = result.ottfmGen.aggregate_metrics.reference[key];
                  const best = pickBest(values, ref);

                  return (
                    <tr key={key}>
                      <th>{label}</th>
                      <td>{fmtMeanStd(values["I-CFM"])}</td>
                      <td>{fmtMeanStd(values["OT-CFM"])}</td>
                      <td>{fmtMeanStd(values["I-TFM"])}</td>
                      <td>{fmtMeanStd(values["OT-TFM"])}</td>
                      <td>{fmtMeanStd(ref)}</td>
                      <td className={methodClass(best)}>{best}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="muted">
            The best column uses absolute distance between each generated mean and the reference μ1
            mean for each metric. This is a compact diagnostic, not a formal hypothesis test.
          </p>
        </>
      )}
    </div>
  );
}
