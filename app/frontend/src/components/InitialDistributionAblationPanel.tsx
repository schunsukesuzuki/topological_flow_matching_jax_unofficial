import { useState } from "react";
import { runMu0Ablation } from "../api";
import type {
  Mu0AblationMethodResult,
  Mu0AblationResponse,
  SpectralMetricSummary,
  SpectralMetrics,
} from "../types";

type Props = {
  withFace: boolean;
  kappa: number;
};

const rows: Array<[keyof SpectralMetrics, string]> = [
  ["harmonic_energy", "harmonic energy"],
  ["low_frequency_energy", "low-frequency energy"],
  ["high_frequency_energy", "high-frequency energy"],
  ["hodge_energy", "xᵀL₁x"],
  ["l2_norm", "||x||²"],
];

const methodOrder = [
  "I-TFM / standard μ0",
  "I-TFM / heat GP μ0",
  "OT-TFM / standard μ0",
  "OT-TFM / heat GP μ0",
];

function fmt(x: number | null | undefined) {
  if (x === null || x === undefined) return "—";
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
  results: Record<string, Mu0AblationMethodResult>,
  key: keyof SpectralMetricSummary,
) {
  const ref = results["OT-TFM / heat GP μ0"].aggregate_metrics.reference[key];
  let bestLabel = methodOrder[0];
  let bestGap = Number.POSITIVE_INFINITY;

  for (const label of methodOrder) {
    const value = results[label].aggregate_metrics.generated[key];
    const gap = absGap(value.mean, ref.mean);
    if (gap < bestGap) {
      bestGap = gap;
      bestLabel = label;
    }
  }

  return bestLabel;
}

function labelClass(label: string) {
  if (label.includes("OT-TFM") && label.includes("heat GP")) return "winner-ot";
  if (label.includes("I-TFM") && label.includes("heat GP")) return "winner-it";
  if (label.includes("OT-TFM")) return "winner-otcfm";
  return "winner-icfm";
}

export function InitialDistributionAblationPanel({ withFace, kappa }: Props) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Mu0AblationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);

    try {
      const res = await runMu0Ablation({
        with_face: withFace,
        kappa,
        n_samples: 1024,
        n_steps: 1200,
        learning_rate: 0.002,
        seed: 101,
        n_eval: 32,
        rollout_steps: 160,
        control_scale: 0.92,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="card wide mu0-ablation-card">
      <h3>Topology-aware initial distribution ablation</h3>
      <p className="muted">
        This compares standard Gaussian μ0 against topology-aware heat GP μ0 for I-TFM and OT-TFM.
        Heat GP initialization damps high-frequency Hodge modes before generation starts, while
        standard Gaussian μ0 is topology-unaware.
      </p>

      <div className="button-row">
        <button onClick={handleRun} disabled={running}>
          {running ? "Running μ0 ablation..." : "Run μ0 ablation"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <p className="muted">
            Ablation settings: n_samples = {result.n_samples}, n_steps = {result.n_steps},
            n_eval = {result.n_eval}, rollout_steps = {result.rollout_steps},
            control_scale = {result.control_scale.toFixed(2)}.
          </p>

          <div className="table-scroll">
            <table className="matrix metric-table mu0-summary-table">
              <thead>
                <tr>
                  <th>method / μ0</th>
                  <th>training MSE</th>
                  <th>mean pair cost</th>
                  <th>generated high-frequency</th>
                  <th>generated xᵀL₁x</th>
                  <th>generated ||x||²</th>
                </tr>
              </thead>
              <tbody>
                {methodOrder.map((label) => {
                  const item = result.results[label];
                  return (
                    <tr key={label}>
                      <th>{label}</th>
                      <td>{item.unnormalized_mse.toExponential(3)}</td>
                      <td>{fmt(item.mean_pair_cost)}</td>
                      <td>{fmtMeanStd(item.aggregate_metrics.generated.high_frequency_energy)}</td>
                      <td>{fmtMeanStd(item.aggregate_metrics.generated.hodge_energy)}</td>
                      <td>{fmtMeanStd(item.aggregate_metrics.generated.l2_norm)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <h4>μ0 ablation: generated distribution vs reference μ1</h4>
          <p className="muted">
            The best column is selected by absolute distance between the generated mean and the
            reference μ1 mean. Reference μ1 is sampled independently for each method, so this is a
            diagnostic comparison rather than a formal statistical test.
          </p>

          <div className="table-scroll">
            <table className="matrix metric-table mu0-ablation-table">
              <thead>
                <tr>
                  <th>metric</th>
                  <th>I-TFM standard μ0</th>
                  <th>I-TFM heat GP μ0</th>
                  <th>OT-TFM standard μ0</th>
                  <th>OT-TFM heat GP μ0</th>
                  <th>reference μ1</th>
                  <th>best</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([key, label]) => {
                  const best = pickBest(result.results, key);
                  const ref = result.results["OT-TFM / heat GP μ0"].aggregate_metrics.reference[key];

                  return (
                    <tr key={key}>
                      <th>{label}</th>
                      {methodOrder.map((methodLabel) => (
                        <td key={methodLabel}>
                          {fmtMeanStd(result.results[methodLabel].aggregate_metrics.generated[key])}
                        </td>
                      ))}
                      <td>{fmtMeanStd(ref)}</td>
                      <td className={labelClass(best)}>{best}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
