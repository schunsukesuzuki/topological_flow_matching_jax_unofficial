import type { BridgeMode } from "../types";

type Props = {
  x0: number[];
  x1: number[];
  cfmXt: number[];
  tfmXt: number[];
  cfmU: number[];
  tfmU: number[];
  analyticalTfmU: number[];
  cfmCost: number;
  tfmCost: number;
  bridgeMode: BridgeMode;
  nnLoss: number | null;
  nnUnnormalizedMse: number | null;
  nnSteps: number | null;
};

function row(label: string, vals: number[], maxCols = 18) {
  const shown = vals.slice(0, maxCols);
  return <tr><th>{label}</th>{shown.map((v, i) => <td key={i}>{v.toFixed(3)}</td>)}{vals.length > maxCols && <td>…</td>}</tr>;
}

export function SignalView({
  x0,
  x1,
  cfmXt,
  tfmXt,
  cfmU,
  tfmU,
  analyticalTfmU,
  cfmCost,
  tfmCost,
  bridgeMode,
  nnLoss,
  nnUnnormalizedMse,
  nnSteps,
}: Props) {
  const trainingText = bridgeMode === "neural" && nnLoss !== null && nnUnnormalizedMse !== null && nnSteps !== null
    ? ` Neural fitting: ${nnSteps} steps, normalized MSE = ${nnLoss.toExponential(3)}, displayed-scale MSE = ${nnUnnormalizedMse.toExponential(3)}.`
    : "";

  return (
    <div className="card wide">
      <h3>CFM vs TFM edge-signal values</h3>
      <p className="muted">
        Previewing the first 18 edge components. TFM u is currently displayed in <strong>{bridgeMode}</strong> mode.
        {trainingText}
      </p>
      <div className="table-scroll">
        <table className="matrix signal-table">
          <tbody>
            {row("x0", x0)}
            {row("x1", x1)}
            {row("CFM x_t", cfmXt)}
            {row("TFM x_t", tfmXt)}
            {row("CFM u", cfmU)}
            {row(bridgeMode === "neural" ? "TFM uθ NN" : "TFM u analytical", tfmU)}
            {bridgeMode === "neural" && row("TFM u target", analyticalTfmU)}
          </tbody>
        </table>
      </div>
      <div className="cost-row">
        <p><strong>CFM transport cost:</strong> {cfmCost.toFixed(6)}</p>
        <p><strong>TFM transport cost:</strong> {tfmCost.toFixed(6)}</p>
      </div>
    </div>
  );
}
