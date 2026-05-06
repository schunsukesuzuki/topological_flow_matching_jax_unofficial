import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Network } from "lucide-react";
import { compute, fetchPresets } from "./api";
import type { BridgeMode, ComputeResponse, Preset } from "./types";
import { Controls } from "./components/Controls";
import { ComplexView } from "./components/ComplexView";
import { MatrixView } from "./components/MatrixView";
import { SpectrumView } from "./components/SpectrumView";
import { SignalView } from "./components/SignalView";
import { FigureOneView } from "./components/FigureOneView";
import { CFMPanel } from "./components/CFMPanel";
import { ITFMPanel } from "./components/ITFMPanel";
import { OTTFMPanel } from "./components/OTTFMPanel";
import { TFMComparisonPanel } from "./components/TFMComparisonPanel";
import { FourWayComparisonPanel } from "./components/FourWayComparisonPanel";
import { InitialDistributionAblationPanel } from "./components/InitialDistributionAblationPanel";
import { DistributionDistancePanel } from "./components/DistributionDistancePanel";
import { KappaSweepPanel } from "./components/KappaSweepPanel";
import { OTCouplingVisualizationPanel } from "./components/OTCouplingVisualizationPanel";
import "./style.css";

function App() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetIndex, setPresetIndex] = useState(0);
  const [withFace, setWithFace] = useState(true);
  const [kappa, setKappa] = useState(2.0);
  const [t, setT] = useState(0.5);
  const [bridgeMode, setBridgeMode] = useState<BridgeMode>("neural");
  const [nnSteps, setNnSteps] = useState(1500);

  const [data, setData] = useState<ComputeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPresets().then(setPresets).catch((e) => setError(String(e)));
  }, []);

  const preset = presets[presetIndex];
  const request = useMemo(() => preset ? {
    with_face: withFace,
    kappa,
    t,
    x0: preset.x0,
    x1: preset.x1,
    bridge_mode: bridgeMode,
    nn_steps: nnSteps,
  } : null, [preset, withFace, kappa, t, bridgeMode, nnSteps]);

  useEffect(() => {
    if (!request) return;
    compute(request).then(setData).catch((e) => setError(String(e)));
  }, [request]);

  return (
    <main>
      <header>
        <div className="title-row"><Network size={28} /><h1>Topological Flow Matching Visualizer</h1></div>
        <p>Figure-1-style Hodge spectrum view plus CFM straight interpolation vs TFM Hodge Laplacian-aware bridge interpolation on an annulus-shaped 2-simplicial complex.</p>
      </header>

      {error && <div className="error">{error}</div>}
      {data && <FigureOneView payload={data.figure1} />}

      <section className="grid">
        <Controls
          presets={presets}
          presetIndex={presetIndex}
          setPresetIndex={setPresetIndex}
          withFace={withFace}
          setWithFace={setWithFace}
          kappa={kappa}
          setKappa={setKappa}
          t={t}
          setT={setT}
          bridgeMode={bridgeMode}
          setBridgeMode={setBridgeMode}
          nnSteps={nnSteps}
          setNnSteps={setNnSteps}
        />

        {data && <ComplexView title="CFM edge signal" nodes={data.nodes} edges={data.edges} faces={data.faces} signal={data.cfm_xt} />}
        {data && <ComplexView title="TFM edge signal" nodes={data.nodes} edges={data.edges} faces={data.faces} signal={data.tfm_xt} />}
        {data && <SpectrumView eigenvalues={data.eigenvalues} />}

        {data && preset && (
          <SignalView
            x0={preset.x0}
            x1={preset.x1}
            cfmXt={data.cfm_xt}
            tfmXt={data.tfm_xt}
            cfmU={data.cfm_u}
            tfmU={data.tfm_u}
            analyticalTfmU={data.analytical_tfm_u}
            cfmCost={data.cfm_cost}
            tfmCost={data.tfm_cost}
            bridgeMode={data.bridge_mode}
            nnLoss={data.nn_loss}
            nnUnnormalizedMse={data.nn_unnormalized_mse}
            nnSteps={data.nn_steps}
          />
        )}

        <CFMPanel withFace={withFace} kappa={kappa} />
        <ITFMPanel withFace={withFace} kappa={kappa} />
        <OTTFMPanel withFace={withFace} kappa={kappa} />
        <TFMComparisonPanel withFace={withFace} kappa={kappa} />
        <FourWayComparisonPanel withFace={withFace} kappa={kappa} />
        <InitialDistributionAblationPanel withFace={withFace} kappa={kappa} />
        <DistributionDistancePanel withFace={withFace} kappa={kappa} />
        <KappaSweepPanel withFace={withFace} />
        <OTCouplingVisualizationPanel withFace={withFace} kappa={kappa} />

        {data && <MatrixView title="B1: edge → node" matrix={data.B1} />}
        {data && <MatrixView title="B2: face → edge" matrix={data.B2} />}
        {data && <MatrixView title="B1B2 = 0" matrix={data.B1B2} />}
        {data && <MatrixView title="L1 Hodge Laplacian" matrix={data.L1} />}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
