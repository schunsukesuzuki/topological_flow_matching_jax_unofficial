# App Quick Start and Usage Guide

> This is a companion README for the **unofficial experimental Topological Flow Matching Visualizer**.  
> It focuses on how to start the app, what each panel does, and how to read the main outputs.

---

## 1. Quick start

From the repository root:

```bash
docker compose up --build
```

Then open:

```text
Frontend:
  http://localhost:5173

Backend API docs:
  http://localhost:8000/docs
```

The frontend is a Vite + React + TypeScript app.  
The backend is a FastAPI app using JAX / Flax for numerical computation and neural vector-field training.

---

## 2. Expected repository structure

The project is expected to have roughly the following structure:

```text
topological-flow-matching-visualizer/
  docker-compose.yml

  backend/
    Dockerfile
    requirements.txt
    app/
      main.py
      schemas.py
      tfm/
        boundary.py
        bridge.py
        cfm_distribution.py
        coupling_visualization.py
        datasets.py
        distribution_metric_eval.py
        distribution_metrics.py
        figure1.py
        flax_model.py
        hodge.py
        itfm.py
        itfm_metrics.py
        kappa_sweep.py
        mu0_ablation.py
        neural_bridge.py
        ottfm.py
        presets.py
        simplicial_complex.py

  frontend/
    Dockerfile
    package.json
    src/
      App.tsx
      api.ts
      types.ts
      style.css
      components/
```

If files are missing after applying patches, Vite may show an import error such as:

```text
Failed to resolve import "./components/SomePanel"
```

In that case, confirm that the corresponding component file exists under:

```text
frontend/src/components/
```

---

## 3. Basic app flow

The app is organized as a vertical dashboard.

A typical usage flow is:

```text
1. Start the app with docker compose.
2. Open the frontend.
3. Inspect the Figure-1-style Hodge spectrum view.
4. Adjust κ and t in the control panel.
5. Compare CFM and TFM edge-signal paths.
6. Run I-CFM / OT-CFM / I-TFM / OT-TFM panels.
7. Inspect distribution metrics and energy diagnostics.
8. Run μ0 ablation, κ sweep, and OT coupling visualization as needed.
```

Some panels trigger backend training or evaluation. These may take a little time because JAX / Flax models are trained on demand.

---

## 4. Main controls

### Preset selector

The preset selector chooses a source / target edge-signal pair for the analytical CFM vs TFM visualizer.

The preset affects:

```text
CFM edge signal
TFM edge signal
CFM / TFM bridge control
CFM / TFM transport cost
```

It does not necessarily affect all distribution-level panels, because those panels sample from synthetic μ0 and μ1 distributions.

---

### `with_face`

This toggles whether the annulus includes 2-simplices / faces.

When faces are included:

```text
B2 is non-empty
L1 includes both lower and upper Hodge terms
```

When faces are removed:

```text
B2 may be empty
the Hodge Laplacian structure changes
```

This changes the topology-aware behavior.

---

### `κ`

`κ` controls the strength of the Hodge heat drift:

```text
dx/dt = -κ L x + uθ(t, x)
```

Interpretation:

```text
small κ:
  weaker Hodge smoothing
  rough / high-frequency modes may remain

large κ:
  stronger Hodge smoothing
  high-frequency modes are suppressed more aggressively
```

The κ sweep panel is designed to study this effect systematically.

---

### `t`

`t` controls the interpolation / bridge time in the analytical CFM-vs-TFM visualizer.

```text
t = 0:
  source side

t = 1:
  target side

0 < t < 1:
  bridge state
```

---

### Bridge mode

The bridge-control display can use:

```text
analytical:
  closed-form TFM bridge control

neural:
  Flax MLP approximation of the bridge control
```

The neural mode trains a small bridge-control model for the selected analytical pair.

---

## 5. Panels and what they mean

## 5.1 Figure-1-style Hodge spectrum panel

This panel shows representative graph and edge signals:

```text
normal sample
heat Gaussian process sample
zero eigenfunction
low-frequency eigenfunction
high-frequency eigenfunction
```

Purpose:

```text
visualize the difference between random noise, heat-smoothed signals,
harmonic modes, low-frequency modes, and high-frequency modes
```

Useful interpretation:

```text
zero eigenfunction:
  harmonic / kernel component

low-frequency eigenfunction:
  smooth structural mode

high-frequency eigenfunction:
  rough / locally oscillatory mode
```

---

## 5.2 CFM edge signal / TFM edge signal

These panels compare the bridge state under:

```text
CFM:
  Euclidean straight interpolation

TFM:
  Hodge Laplacian-aware bridge interpolation
```

CFM treats the edge signal as a Euclidean vector.  
TFM uses the Hodge spectrum and heat dynamics.

---

## 5.3 L1 spectrum

This panel displays the eigenvalue spectrum of the L1 Hodge Laplacian.

Interpretation:

```text
near-zero eigenvalues:
  harmonic components / topological cycles

small positive eigenvalues:
  low-frequency smooth modes

large eigenvalues:
  high-frequency rough modes
```

---

## 5.4 CFM vs TFM edge-signal values

This panel shows numerical signal and control values.

It includes:

```text
x0
x1
CFM x_t
TFM x_t
CFM velocity
TFM control
transport costs
neural bridge-control loss if enabled
```

This is useful when the visual edge plot is not enough.

---

## 5.5 Boundary and Hodge matrices

The app displays:

```text
B1: edge -> node
B2: face -> edge
B1B2 = 0
L1 Hodge Laplacian
```

`B1B2 = 0` is the discrete chain-complex identity:

```text
boundary of boundary = 0
```

`L1` is the Hodge Laplacian acting on edge signals.

---

## 5.6 CFM panel

This panel trains and generates from distribution-level CFM baselines.

Supported couplings:

```text
independent:
  I-CFM

ot:
  OT-CFM using Euclidean OT cost
```

CFM generation uses a learned vector field without the Hodge heat drift.

This is useful as a topology-unaware baseline.

---

## 5.7 I-TFM panel

This panel trains and generates from I-TFM:

```text
independent coupling + TFM dynamics
```

Training pairs are sampled independently:

```text
(X0, X1) ~ μ0 ⊗ μ1
```

Generation dynamics:

```text
dx/dt = -κLx + uθ(t, x)
```

Main interpretation:

```text
I-TFM tests how much topology-aware dynamics alone can help,
without topology-aware OT coupling.
```

---

## 5.8 OT-TFM panel

This panel trains and generates from OT-TFM:

```text
Sinkhorn(TFM transport cost) coupling + TFM dynamics
```

Training pairs are chosen using a topology-aware OT cost, then the same TFM-style neural vector field is trained.

Main interpretation:

```text
OT-TFM tests the combination of topology-aware pair selection
and topology-aware dynamics.
```

---

## 5.9 Four-way comparison panel

This compares:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
```

It is useful for observing:

```text
whether CFM baselines produce high Hodge roughness
whether OT-CFM improves training MSE but not spectral quality
whether TFM variants reduce high-frequency energy
whether I-TFM or OT-TFM is better under the current setting
```

---

## 5.10 Topology-aware initial distribution ablation

This panel compares source distributions:

```text
standard μ0:
  topology-unaware standard Gaussian

heat GP μ0:
  topology-aware heat Gaussian process
```

Methods compared:

```text
I-TFM / standard μ0
I-TFM / heat GP μ0
OT-TFM / standard μ0
OT-TFM / heat GP μ0
```

Main interpretation:

```text
heat GP μ0 can make training smoother,
but it is not guaranteed to improve every generated distribution metric.
```

The result depends on:

```text
κ
target spectral profile
coupling strategy
control scale
```

---

## 5.11 Distribution distance metrics

This panel compares generated samples to reference μ1 samples using:

```text
RBF MMD²
spectral sliced Wasserstein
spectral mode Wasserstein
```

Lower is better.

The spectral distances first project signals into the Hodge eigenbasis, so they are more sensitive to topology-aware mismatch.

This panel is often more reliable than single-sample visual inspection.

---

## 5.12 κ sweep / Hodge heat-drift sensitivity

This panel sweeps:

```text
κ = 0.25, 0.5, 1.0, 2.0, 4.0
```

for:

```text
I-TFM
OT-TFM
```

It reports:

```text
distribution distances
generated xᵀL₁x
generated high-frequency energy
generated ||x||²
```

Main interpretation:

```text
κ controls how strongly the Hodge heat drift suppresses roughness.
```

Typical pattern:

```text
low κ:
  weak smoothing

intermediate κ:
  coupling and dynamics can balance well

high κ:
  heat drift may dominate and reduce the extra benefit of OT coupling
```

---

## 5.13 OT coupling visualization

This panel visualizes the OT coupling itself.

Supported modes:

```text
OT-CFM / Euclidean cost
OT-TFM / TFM transport cost
```

Displayed values:

```text
cost matrix C_ij
Sinkhorn plan P_ij
expected cost
plan entropy
mean row max mass
mean cost
top plan entries
most likely target per source row
```

Interpretation:

```text
cost matrix:
  lower means the source-target pair is cheaper under the chosen geometry

plan matrix:
  larger mass means Sinkhorn assigns more transport between that pair

plan entropy:
  larger means more diffuse coupling

mean row max mass:
  larger means more concentrated row-wise matching
```

OT-CFM and OT-TFM use the same Sinkhorn algorithm, but different cost geometry.

```text
OT-CFM:
  Euclidean distance

OT-TFM:
  Hodge spectral / heat semigroup-aware TFM transport cost
```

---

## 6. Metrics guide

### `harmonic_energy`

Energy in near-zero eigenvalue modes.

Interpretation:

```text
represents harmonic / cycle-like components
not penalized by xᵀLx
should match reference μ1, not blindly minimized
```

---

### `low_frequency_energy`

Energy in low positive eigenvalue modes.

Interpretation:

```text
smooth large-scale structure
should match reference μ1
too high can indicate overshoot
too low can indicate loss of target structure
```

---

### `high_frequency_energy`

Energy in high eigenvalue modes.

Interpretation:

```text
rough / noisy / locally oscillatory component
usually should be small for smooth targets
```

---

### `hodge_energy = xᵀLx`

Hodge roughness energy:

```text
xᵀLx = Σ_i λ_i y_i²
```

Interpretation:

```text
smaller means smoother under the Hodge Laplacian
large high-frequency modes are penalized strongly
harmonic modes are not penalized because λ = 0
```

---

### `l2_norm = ||x||²`

Total signal amplitude.

Interpretation:

```text
should match reference μ1 amplitude
not blindly minimized
```

---

## 7. Notes on training time

Some panels train neural vector fields on demand.

The following may take longer than simple visualization:

```text
I-CFM / OT-CFM training
I-TFM training
OT-TFM training
Four-way comparison
μ0 ablation
distribution metrics
κ sweep
```

The κ sweep is especially heavier because it trains multiple models across κ values.

The default settings are intentionally modest to keep local experimentation manageable.

---

## 8. Common troubleshooting

### Frontend import error

Example:

```text
Failed to resolve import "./components/SomePanel"
```

Check that the corresponding file exists:

```text
frontend/src/components/SomePanel.tsx
```

Then rebuild:

```bash
docker compose up --build
```

---

### Backend endpoint not found

If the frontend calls an endpoint that does not exist, confirm that:

```text
backend/app/main.py
backend/app/schemas.py
frontend/src/api.ts
frontend/src/types.ts
```

are all from the same patch version.

---

### CSS not applied

Some patches include a file such as:

```text
frontend/src/style_some_patch.css
```

Append it to:

```bash
cat frontend/src/style_some_patch.css >> frontend/src/style.css
rm frontend/src/style_some_patch.css
```

Then restart the frontend.

---

### Heatmaps look faint

The OT coupling heatmaps may appear faint depending on browser rendering or PDF export.

Use the numerical tables as the source of truth:

```text
top plan entries
row-wise argmax pairs
expected cost
plan entropy
mean row max mass
```

---

## 9. Recommended first experiments

### Experiment 1: CFM vs TFM bridge

```text
1. Select a preset.
2. Move t from 0 to 1.
3. Compare CFM edge signal and TFM edge signal.
4. Adjust κ and observe how the TFM bridge changes.
```

---

### Experiment 2: I-TFM vs OT-TFM

```text
1. Train I-TFM.
2. Generate I-TFM samples.
3. Train OT-TFM.
4. Generate OT-TFM samples.
5. Compare hodge_energy, high_frequency_energy, and distribution distances.
```

---

### Experiment 3: CFM baselines

```text
1. Run I-CFM.
2. Run OT-CFM.
3. Compare with I-TFM / OT-TFM.
4. Watch for excessive xᵀLx, ||x||², or low-frequency energy in OT-CFM.
```

---

### Experiment 4: μ0 ablation

```text
1. Run topology-aware initial distribution ablation.
2. Compare standard μ0 and heat GP μ0.
3. Check both training MSE and generated distribution metrics.
```

---

### Experiment 5: κ sweep

```text
1. Run κ sweep.
2. Observe how high-frequency energy and xᵀLx change.
3. Check whether I-TFM or OT-TFM wins at each κ.
```

---

### Experiment 6: OT coupling visualization

```text
1. Select OT-CFM.
2. Build coupling heatmaps.
3. Select OT-TFM.
4. Build coupling heatmaps.
5. Compare expected cost, entropy, and top plan entries.
```

---

## 10. Practical interpretation summary

A useful way to read the app is:

```text
CFM:
  baseline Euclidean flow matching

OT-CFM:
  Euclidean OT coupling, but still topology-unaware dynamics

I-TFM:
  topology-aware dynamics with independent coupling

OT-TFM:
  topology-aware dynamics and topology-aware coupling
```

Metrics should be interpreted as:

```text
hodge_energy:
  lower roughness is generally better

high_frequency_energy:
  lower noise / roughness is generally better

harmonic_energy:
  match reference μ1

low_frequency_energy:
  match reference μ1

l2_norm:
  match reference μ1

distribution distances:
  lower is better
```

The main experimental question is not simply:

```text
Which method has the lowest training MSE?
```

but rather:

```text
Which method produces generated samples whose Hodge spectral profile
matches the reference distribution while suppressing rough modes?
```

---

## 11. Suggested next additions

Potential next app-level extensions:

```text
sinkhorn_epsilon sweep
side-by-side OT-CFM vs OT-TFM coupling on the same minibatch
interactive spectral cutoff controls
larger simplicial complexes
alternative μ1 distributions
higher-order Hodge Laplacians
better ODE solvers
formal benchmark scripts
```
