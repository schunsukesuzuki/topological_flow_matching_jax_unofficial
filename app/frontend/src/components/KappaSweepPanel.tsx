import { useMemo, useState } from "react";
import { runKappaSweep } from "../api";
import type { KappaSweepResponse, KappaSweepRow } from "../types";

type Props = {
  withFace: boolean;
};

const metricRows = [
  ["spectral_sliced_wasserstein", "spectral sliced W"],
  ["spectral_mode_wasserstein", "spectral mode W"],
  ["mmd_rbf", "RBF MMD²"],
] as const;

function fmt(x: number | null | undefined) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return "NaN";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.001) return x.toExponential(3);
  return x.toFixed(5);
}

function byKappaAndMethod(rows: KappaSweepRow[]) {
  const m = new Map<string, KappaSweepRow>();
  for (const row of rows) {
    m.set(`${row.kappa}:${row.method}`, row);
  }
  return m;
}

function bestMethod(it?: KappaSweepRow, ot?: KappaSweepRow, key?: keyof KappaSweepRow["distances"]) {
  if (!it || !ot || !key) return "—";
  return it.distances[key] <= ot.distances[key] ? "I-TFM" : "OT-TFM";
}

export function KappaSweepPanel({ withFace }: Props) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<KappaSweepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);

    try {
      const res = await runKappaSweep({
        with_face: withFace,
        kappas: [0.25, 0.5, 1.0, 2.0, 4.0],
        n_samples: 768,
        n_steps: 900,
        learning_rate: 0.002,
        seed: 303,
        n_eval: 32,
        rollout_steps: 160,
        control_scale: 0.92,
        ot_batch_size: 64,
        sinkhorn_epsilon: 0.75,
        mu0_mode: "heat_gp",
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  const rowMap = useMemo(() => byKappaAndMethod(result?.rows ?? []), [result]);

  return (
    <div className="card wide kappa-sweep-card">
      <h3>κ sweep / Hodge heat-drift sensitivity</h3>
      <p className="muted">
        This panel sweeps κ in dx/dt = -κLx + uθ(t, x) for I-TFM and OT-TFM.
        It measures how stronger or weaker Hodge heat drift changes generated distribution quality.
        Distances are lower-is-better.
      </p>

      <div className="button-row">
        <button onClick={handleRun} disabled={running}>
          {running ? "Running κ sweep..." : "Run κ sweep"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <p className="muted">
            Settings: κ ∈ [{result.kappas.join(", ")}], n_samples = {result.n_samples},
            n_steps = {result.n_steps}, n_eval = {result.n_eval}, rollout_steps =
            {result.rollout_steps}, μ0 = {result.mu0_mode}, control_scale =
            {result.control_scale.toFixed(2)}.
          </p>

          <h4>Distribution distances across κ</h4>
          <div className="table-scroll">
            <table className="matrix metric-table kappa-sweep-table">
              <thead>
                <tr>
                  <th>κ</th>
                  {metricRows.map(([, label]) => (
                    <th key={`it-${label}`}>I-TFM {label}</th>
                  ))}
                  {metricRows.map(([, label]) => (
                    <th key={`ot-${label}`}>OT-TFM {label}</th>
                  ))}
                  <th>best by spectral sliced W</th>
                </tr>
              </thead>
              <tbody>
                {result.kappas.map((kappa) => {
                  const it = rowMap.get(`${kappa}:I-TFM`);
                  const ot = rowMap.get(`${kappa}:OT-TFM`);
                  const best = bestMethod(it, ot, "spectral_sliced_wasserstein");

                  return (
                    <tr key={kappa}>
                      <th>{kappa}</th>
                      {metricRows.map(([key]) => (
                        <td key={`it-${kappa}-${key}`}>{fmt(it?.distances[key])}</td>
                      ))}
                      {metricRows.map(([key]) => (
                        <td key={`ot-${kappa}-${key}`}>{fmt(ot?.distances[key])}</td>
                      ))}
                      <td className={best === "OT-TFM" ? "winner-ot" : best === "I-TFM" ? "winner-it" : ""}>
                        {best}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <h4>Energy diagnostics across κ</h4>
          <div className="table-scroll">
            <table className="matrix metric-table kappa-sweep-table">
              <thead>
                <tr>
                  <th>κ</th>
                  <th>I-TFM xᵀL₁x</th>
                  <th>OT-TFM xᵀL₁x</th>
                  <th>I-TFM high-frequency</th>
                  <th>OT-TFM high-frequency</th>
                  <th>I-TFM ||x||²</th>
                  <th>OT-TFM ||x||²</th>
                </tr>
              </thead>
              <tbody>
                {result.kappas.map((kappa) => {
                  const it = rowMap.get(`${kappa}:I-TFM`);
                  const ot = rowMap.get(`${kappa}:OT-TFM`);

                  return (
                    <tr key={kappa}>
                      <th>{kappa}</th>
                      <td>{fmt(it?.aggregate_metrics.generated.hodge_energy.mean)}</td>
                      <td>{fmt(ot?.aggregate_metrics.generated.hodge_energy.mean)}</td>
                      <td>{fmt(it?.aggregate_metrics.generated.high_frequency_energy.mean)}</td>
                      <td>{fmt(ot?.aggregate_metrics.generated.high_frequency_energy.mean)}</td>
                      <td>{fmt(it?.aggregate_metrics.generated.l2_norm.mean)}</td>
                      <td>{fmt(ot?.aggregate_metrics.generated.l2_norm.mean)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="muted">
            A useful pattern to look for is whether low κ benefits from heat-GP initialization and
            whether high κ makes the dynamics itself strong enough to smooth standard source noise.
            This panel focuses on heat GP μ0 while varying only κ.
          </p>
        </>
      )}
    </div>
  );
}
