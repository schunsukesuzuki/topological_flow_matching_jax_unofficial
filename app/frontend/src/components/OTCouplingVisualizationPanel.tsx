import { useState } from "react";
import { fetchOTCouplingVisualization } from "../api";
import type {
  CouplingMethod,
  OTCouplingVisualizationResponse,
} from "../types";

type Props = {
  withFace: boolean;
  kappa: number;
};

function fmt(x: number | null | undefined) {
  if (x === null || x === undefined) return "—";
  if (!Number.isFinite(x)) return "NaN";
  if (Math.abs(x) >= 1000 || Math.abs(x) < 0.001) return x.toExponential(3);
  return x.toFixed(4);
}

function matrixMinMax(matrix: number[][]) {
  const vals = matrix.flat();
  return {
    min: Math.min(...vals),
    max: Math.max(...vals),
  };
}

function Heatmap({
  title,
  matrix,
  invert = false,
}: {
  title: string;
  matrix: number[][];
  invert?: boolean;
}) {
  const { min, max } = matrixMinMax(matrix);
  const denom = Math.max(max - min, 1e-12);

  return (
    <div className="heatmap-panel">
      <h4>{title}</h4>
      <div
        className="heatmap-grid"
        style={{
          gridTemplateColumns: `repeat(${matrix[0]?.length ?? 1}, 1fr)`,
        }}
      >
        {matrix.map((row, i) =>
          row.map((value, j) => {
            const raw = (value - min) / denom;
            const alpha = invert ? 1 - raw : raw;
            return (
              <div
                key={`${i}-${j}`}
                className="heatmap-cell"
                title={`row ${i}, col ${j}: ${fmt(value)}`}
                style={{
                  opacity: 0.18 + 0.82 * alpha,
                }}
              />
            );
          }),
        )}
      </div>
      <p className="muted">
        min = {fmt(min)}, max = {fmt(max)}
      </p>
    </div>
  );
}

export function OTCouplingVisualizationPanel({ withFace, kappa }: Props) {
  const [method, setMethod] = useState<CouplingMethod>("ot_tfm");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<OTCouplingVisualizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setRunning(true);
    setError(null);

    try {
      const res = await fetchOTCouplingVisualization({
        with_face: withFace,
        kappa,
        method,
        batch_size: 16,
        sinkhorn_epsilon: 0.75,
        seed: method === "ot_tfm" ? 404 : 405,
        mu0_mode: "heat_gp",
        top_k: 12,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="card wide ot-coupling-card">
      <h3>OT coupling visualization</h3>
      <p className="muted">
        This panel visualizes the minibatch OT coupling used by OT-CFM and OT-TFM.
        OT-CFM uses squared Euclidean cost, while OT-TFM uses the TFM transport cost
        in Hodge spectral coordinates.
      </p>

      <label>
        coupling type
        <select value={method} onChange={(e) => setMethod(e.target.value as CouplingMethod)}>
          <option value="ot_cfm">OT-CFM / Euclidean cost</option>
          <option value="ot_tfm">OT-TFM / TFM transport cost</option>
        </select>
      </label>

      <div className="button-row">
        <button onClick={handleRun} disabled={running}>
          {running ? "Building coupling..." : "Build coupling heatmaps"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {result && (
        <>
          <p className="muted">
            {result.label}; batch_size = {result.batch_size}, ε ={" "}
            {result.sinkhorn_epsilon}, μ0 = {result.mu0_mode}, κ = {result.kappa}.
          </p>

          <div className="ot-coupling-summary">
            <div>
              <strong>expected cost</strong>
              <span>{fmt(result.summary.expected_cost)}</span>
            </div>
            <div>
              <strong>plan entropy</strong>
              <span>{fmt(result.summary.plan_entropy)}</span>
            </div>
            <div>
              <strong>mean row max mass</strong>
              <span>{fmt(result.summary.mean_row_max_mass)}</span>
            </div>
            <div>
              <strong>mean cost</strong>
              <span>{fmt(result.summary.mean_cost)}</span>
            </div>
          </div>

          <div className="heatmap-row">
            <Heatmap title="Cost matrix Cᵢⱼ" matrix={result.cost_matrix} invert />
            <Heatmap title="Sinkhorn plan Pᵢⱼ" matrix={result.plan_matrix} />
          </div>

          <h4>Top plan entries</h4>
          <div className="table-scroll">
            <table className="matrix metric-table coupling-table">
              <thead>
                <tr>
                  <th>rank</th>
                  <th>source i</th>
                  <th>target j</th>
                  <th>plan mass</th>
                  <th>cost</th>
                </tr>
              </thead>
              <tbody>
                {result.top_pairs.map((p, idx) => (
                  <tr key={`${p.source_index}-${p.target_index}-${idx}`}>
                    <th>{idx + 1}</th>
                    <td>{p.source_index}</td>
                    <td>{p.target_index}</td>
                    <td>{fmt(p.plan_mass)}</td>
                    <td>{fmt(p.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h4>Most likely target per source row</h4>
          <div className="table-scroll">
            <table className="matrix metric-table coupling-table">
              <thead>
                <tr>
                  <th>source i</th>
                  <th>argmax target j</th>
                  <th>plan mass</th>
                  <th>cost</th>
                </tr>
              </thead>
              <tbody>
                {result.row_argmax_pairs.map((p) => (
                  <tr key={p.source_index}>
                    <th>{p.source_index}</th>
                    <td>{p.target_index}</td>
                    <td>{fmt(p.plan_mass)}</td>
                    <td>{fmt(p.cost)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="muted">
            In the cost heatmap, darker cells indicate lower cost. In the plan heatmap,
            darker cells indicate larger transport mass. OT-TFM should prefer pairs that
            are close under the topology-aware TFM transport cost, not merely under
            Euclidean signal distance.
          </p>
        </>
      )}
    </div>
  );
}
