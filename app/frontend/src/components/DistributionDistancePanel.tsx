import { useState } from "react";
import { runDistributionMetrics } from "../api";
import type {
  DistributionDistances,
  DistributionMetricEvalResponse,
} from "../types";

type Props = {
  withFace: boolean;
  kappa: number;
};

const methodOrder = ["I-CFM", "OT-CFM", "I-TFM", "OT-TFM"];

const distanceRows: Array<[keyof DistributionDistances, string]> = [
  ["mmd_rbf", "RBF MMD²"],
  ["spectral_sliced_wasserstein", "spectral sliced W"],
  ["spectral_mode_wasserstein", "spectral mode W"],
];

function fmt(x: number | null | undefined) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return "NaN";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.001) return x.toExponential(3);
  return x.toFixed(5);
}

function pickBest(result: DistributionMetricEvalResponse, key: keyof DistributionDistances) {
  let best = methodOrder[0];
  let bestValue = result.results[best].distances[key];

  for (const method of methodOrder.slice(1)) {
    const value = result.results[method].distances[key];
    if (value < bestValue) {
      best = method;
      bestValue = value;
    }
  }

  return best;
}

function methodClass(method: string) {
  if (method === "OT-TFM") return "winner-ot";
  if (method === "I-TFM") return "winner-it";
  if (method === "OT-CFM") return "winner-otcfm";
  return "winner-icfm";
}

export function DistributionDistancePanel({ withFace, kappa }: Props) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DistributionMetricEvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);

    try {
      const res = await runDistributionMetrics({
        with_face: withFace,
        kappa,
        n_samples: 1024,
        n_steps: 1200,
        learning_rate: 0.002,
        seed: 202,
        n_eval: 32,
        rollout_steps: 160,
        cfm_control_scale: 1.0,
        tfm_control_scale: 0.92,
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
    <div className="card wide distribution-distance-card">
      <h3>Distribution distance metrics</h3>
      <p className="muted">
        This panel compares generated samples against reference μ1 using distribution-level distances,
        not only hand-designed Hodge energy summaries. Distances are lower-is-better.
      </p>

      <div className="button-row">
        <button onClick={handleRun} disabled={running}>
          {running ? "Running distribution metrics..." : "Run distribution distance metrics"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <p className="muted">
            Settings: n_samples = {result.n_samples}, n_steps = {result.n_steps}, n_eval =
            {result.n_eval}, rollout_steps = {result.rollout_steps}.
          </p>

          <div className="table-scroll">
            <table className="matrix metric-table distribution-distance-table">
              <thead>
                <tr>
                  <th>distance</th>
                  <th>I-CFM</th>
                  <th>OT-CFM</th>
                  <th>I-TFM</th>
                  <th>OT-TFM</th>
                  <th>best</th>
                </tr>
              </thead>
              <tbody>
                {distanceRows.map(([key, label]) => {
                  const best = pickBest(result, key);
                  return (
                    <tr key={key}>
                      <th>{label}</th>
                      {methodOrder.map((method) => (
                        <td key={method}>{fmt(result.results[method].distances[key])}</td>
                      ))}
                      <td className={methodClass(best)}>{best}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <h4>Training and pair diagnostics</h4>
          <div className="table-scroll">
            <table className="matrix metric-table distribution-distance-table">
              <thead>
                <tr>
                  <th>method</th>
                  <th>training MSE</th>
                  <th>mean pair cost</th>
                  <th>generated xᵀL₁x</th>
                  <th>generated ||x||²</th>
                </tr>
              </thead>
              <tbody>
                {methodOrder.map((method) => {
                  const item = result.results[method];
                  return (
                    <tr key={method}>
                      <th>{method}</th>
                      <td>{item.unnormalized_mse.toExponential(3)}</td>
                      <td>{fmt(item.mean_pair_cost)}</td>
                      <td>
                        {fmt(item.aggregate_metrics.generated.hodge_energy.mean)} ±{" "}
                        {fmt(item.aggregate_metrics.generated.hodge_energy.std)}
                      </td>
                      <td>
                        {fmt(item.aggregate_metrics.generated.l2_norm.mean)} ±{" "}
                        {fmt(item.aggregate_metrics.generated.l2_norm.std)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="muted">
            RBF MMD² is computed in signal space. Spectral sliced W and spectral mode W are computed
            after projecting signals into the Hodge eigenbasis, making them more sensitive to
            topological/spectral mismatch.
          </p>
        </>
      )}
    </div>
  );
}
