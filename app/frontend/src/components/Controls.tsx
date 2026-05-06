import type { BridgeMode, Preset } from "../types";

type Props = {
  presets: Preset[];
  presetIndex: number;
  setPresetIndex: (i: number) => void;
  withFace: boolean;
  setWithFace: (v: boolean) => void;
  kappa: number;
  setKappa: (v: number) => void;
  t: number;
  setT: (v: number) => void;
  bridgeMode: BridgeMode;
  setBridgeMode: (v: BridgeMode) => void;
  nnSteps: number;
  setNnSteps: (v: number) => void;
};

export function Controls({
  presets,
  presetIndex,
  setPresetIndex,
  withFace,
  setWithFace,
  kappa,
  setKappa,
  t,
  setT,
  bridgeMode,
  setBridgeMode,
  nnSteps,
  setNnSteps,
}: Props) {
  const preset = presets[presetIndex];
  return (
    <div className="card controls">
      <h3>Controls</h3>

      <label>
        Preset
        <select value={presetIndex} onChange={(e) => setPresetIndex(Number(e.target.value))}>
          {presets.map((p, i) => <option key={p.name} value={i}>{p.name}</option>)}
        </select>
      </label>
      {preset && <p className="muted">{preset.description}</p>}

      <label>
        Bridge-control mode
        <select value={bridgeMode} onChange={(e) => setBridgeMode(e.target.value as BridgeMode)}>
          <option value="neural">Neural uθ(t, x_t) learned from analytical TFM target</option>
          <option value="analytical">Analytical closed-form TFM control</option>
        </select>
      </label>

      <label>
        NN training steps = {nnSteps}
        <input
          type="range"
          min={300}
          max={5000}
          step={100}
          value={nnSteps}
          disabled={bridgeMode !== "neural"}
          onChange={(e) => setNnSteps(Number(e.target.value))}
        />
      </label>

      <label className="check">
        <input type="checkbox" checked={withFace} onChange={(e) => setWithFace(e.target.checked)} />
        include 2-simplex faces in annulus
      </label>

      <label>
        t = {t.toFixed(2)}
        <input type="range" min={0} max={1} step={0.01} value={t} onChange={(e) => setT(Number(e.target.value))} />
      </label>

      <label>
        κ = {kappa.toFixed(2)}
        <input type="range" min={0} max={5} step={0.05} value={kappa} onChange={(e) => setKappa(Number(e.target.value))} />
      </label>

      <p className="muted">
        Neural mode now uses Adam, feature/target normalization, richer features, and more time samples to fit uθ(t, x_t) more closely.
      </p>
    </div>
  );
}
