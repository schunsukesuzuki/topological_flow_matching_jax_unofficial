# Topological Flow Matching Visualizer



**Unofficial experimental implementation.**  

- cf. Paper: [Topological Flow Matching](https://openreview.net/forum?id=5CM3ax45Ma)

> This repository is **not** the official repository of the Topological Flow Matching paper and is **not maintained by the paper authors**. It is an independent, experimental visualizer and research prototype intended to help inspect, compare, and reason about CFM / TFM-style dynamics on a small synthetic 2-simplicial complex.

This app visualizes and experiments with **Conditional Flow Matching (CFM)**, **Topological Flow Matching (TFM)**, **I-TFM**, **OT-TFM**, **I-CFM**, and **OT-CFM** on an annulus-shaped 2-simplicial complex.

The main goal is to make the difference between Euclidean flow matching and topology-aware flow matching visible and measurable:

```text
CFM:
  Euclidean straight bridge / topology-unaware dynamics

TFM:
  Hodge Laplacian-aware bridge / topology-aware heat drift

I-TFM:
  independent source-target coupling + TFM dynamics

OT-TFM:
  topology-aware OT coupling + TFM dynamics

I-CFM:
  independent source-target coupling + CFM dynamics

OT-CFM:
  Euclidean OT coupling + CFM dynamics
```

The implementation is intentionally compact and educational. It is designed for local experimentation, visual inspection, and exploratory analysis rather than for production training or exact reproduction of the original paper's full experimental suite.

---

## What this repository demonstrates

This app focuses on edge signals on a 2-simplicial annulus. The annulus is useful because it has a central hole, so the Hodge Laplacian exposes nontrivial harmonic structure.

The app lets you inspect:

- graph node signals,
- edge signals on a simplicial complex,
- boundary matrices,
- Hodge Laplacian structure,
- Hodge eigenvalues and eigenmodes,
- CFM and TFM bridge paths,
- neural approximation of TFM bridge controls,
- distribution-level I-TFM / OT-TFM generation,
- distribution-level I-CFM / OT-CFM baselines,
- Hodge spectral diagnostics,
- distribution distance metrics,
- κ sensitivity,
- OT coupling matrices and Sinkhorn plans.

The top Figure-1-style visualization follows the conceptual structure of the TFM paper's introductory figure:

1. normal sample,
2. heat Gaussian process sample,
3. zero eigenfunction / harmonic component,
4. low-frequency eigenfunction,
5. high-frequency eigenfunction.

The first row shows graph node signals. The second row shows edge signals on a 2-simplicial annulus, with edge magnitude encoded by line width and sign/value encoded by color.

---

## Important disclaimer

This repository is a **research-oriented toy implementation**.

It is not:

- the official Topological Flow Matching implementation,
- a verified reproduction of all paper experiments,
- a complete benchmark suite,
- a production-quality generative modeling library.

It is:

- an experimental visualizer,
- a small educational implementation,
- a tool for reasoning about Hodge Laplacian-aware flow matching,
- a sandbox for comparing Euclidean and topology-aware coupling/dynamics.

---

## Implemented features

### 1. Figure-1-style Hodge spectrum visualization

The UI includes a Figure-1-style panel showing representative graph and edge signals:

```text
normal sample
heat Gaussian process sample
zero eigenfunction
low-frequency eigenfunction
high-frequency eigenfunction
```

This is meant to make the Hodge spectral decomposition visually intuitive.

Implemented concepts:

- graph-level signal display,
- edge-signal display,
- Hodge eigenfunction visualization,
- heat GP smoothing,
- low- vs high-frequency contrast,
- annulus topology with a central hole.

---

### 2. Annulus-shaped 2-simplicial complex

The app uses a larger annulus complex rather than a single triangle.

It exposes:

```text
nodes
edges
faces
B1: edge -> node boundary matrix
B2: face -> edge boundary matrix
B1B2 = 0
L1 Hodge Laplacian
```

The central hole makes harmonic edge-flow structure visible.

---

### 3. Analytical CFM / TFM bridge visualizer

The base visualizer compares:

```text
CFM path:
  Euclidean straight interpolation

TFM path:
  Hodge Laplacian-aware spectral bridge
```

It includes:

- `t` slider,
- `κ` slider,
- CFM edge-signal visualization,
- TFM edge-signal visualization,
- CFM transport cost,
- TFM transport cost,
- analytical TFM bridge control,
- optional neural bridge-control approximation.

---

### 4. Neural bridge-control fitting with Flax

The app includes a Flax MLP scaffold and training path for approximating the TFM bridge control:

```text
u_θ(t, x_t, κ)
```

The analytical TFM bridge control is used as a supervised target. This allows switching between:

```text
analytical bridge control
neural bridge-control approximation
```

This is not a full-scale paper reproduction. It is a minimal neural training demo showing how closed-form bridge controls can supervise a neural vector field.

---

### 5. I-TFM: Independent-coupling Topological Flow Matching

The app implements distribution-level I-TFM.

Training pairs are sampled independently:

```text
(X0, X1) ~ μ0 ⊗ μ1
```

The neural model learns:

```text
u_θ(t, x_t, κ)
```

Generation uses:

```text
dx/dt = -κ L x + u_θ(t, x)
```

where `L` is the L1 Hodge Laplacian.

Implemented I-TFM components:

- independent source-target pair sampling,
- TFM bridge path in Hodge spectral coordinates,
- closed-form TFM bridge control targets,
- Flax MLP training,
- Euler rollout,
- single-sample display,
- multi-sample distribution summary,
- source / generated / reference comparison.

---

### 6. OT-TFM: Topology-aware OT-coupled TFM

The app implements OT-TFM, where source-target pairs are selected using a topology-aware transport cost.

The training pipeline is:

```text
x0_pool ~ μ0
x1_pool ~ μ1
C_ij = TFMTransportCost(x0_i, x1_j)
P = Sinkhorn(C)
(x0, x1) ~ P
```

Then the same TFM bridge-control learning is applied.

Implemented OT-TFM components:

- TFM transport-cost matrix,
- entropic Sinkhorn coupling,
- OT pair sampling,
- `mean_pair_cost` diagnostics,
- Flax MLP training,
- Euler rollout,
- multi-sample distribution evaluation.

---

### 7. I-CFM / OT-CFM distribution-level baselines

The app implements distribution-level CFM baselines:

```text
I-CFM:
  independent coupling + straight CFM bridge

OT-CFM:
  Euclidean OT coupling + straight CFM bridge
```

These baselines are useful for comparing topology-aware and topology-unaware methods under similar training and generation settings.

Implemented CFM baseline components:

- independent CFM pair sampling,
- Euclidean OT pair sampling,
- straight bridge path,
- straight bridge velocity target,
- neural vector field training,
- Euler rollout,
- distribution-level comparison.

---

### 8. Four-way comparison panel

The app includes a four-way comparison panel:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
reference μ1
```

It compares training MSE and generated distribution statistics.

This panel is useful for observing patterns such as:

- CFM may fit training targets but produce rougher generated distributions,
- OT-CFM may improve training MSE but still fail topology-aware spectral matching,
- TFM variants strongly reduce Hodge energy and high-frequency energy,
- OT-TFM can improve energy diagnostics under some κ settings,
- I-TFM can be competitive or better under some distribution-distance metrics.

---

### 9. Topology-aware initial distribution ablation

The app includes an ablation over the source distribution `μ0`.

Implemented source modes:

```text
standard:
  topology-unaware standard Gaussian source

heat_gp:
  topology-aware heat Gaussian process source
```

Compared methods:

```text
I-TFM / standard μ0
I-TFM / heat GP μ0
OT-TFM / standard μ0
OT-TFM / heat GP μ0
```

This ablation helps test whether topology-aware initialization improves:

- training stability,
- pair cost,
- generated Hodge energy,
- generated high-frequency energy,
- generated norm,
- closeness to reference μ1.

A key interpretation is that topology-aware initialization is not automatically better in every metric. It interacts with:

```text
κ
target spectral profile
Hodge heat drift
coupling strategy
control scale
```

---

### 10. Distribution distance metrics

The app includes distribution-level distances between generated samples and reference μ1 samples.

Implemented distances:

```text
RBF MMD²
spectral sliced Wasserstein
spectral mode Wasserstein
```

Interpretation:

```text
lower is better
```

The spectral metrics first project signals into the Hodge eigenbasis, making them more sensitive to topological / spectral mismatch than plain signal-space metrics.

These metrics are used to compare:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
```

---

### 11. κ sweep / Hodge heat-drift sensitivity

The app includes a κ sweep panel for I-TFM and OT-TFM.

The swept dynamics are:

```text
dx/dt = -κ L x + u_θ(t, x)
```

Default κ values:

```text
0.25
0.5
1.0
2.0
4.0
```

The panel reports:

```text
RBF MMD²
spectral sliced Wasserstein
spectral mode Wasserstein
generated xᵀL₁x
generated high-frequency energy
generated ||x||²
```

This helps analyze how strongly the Hodge heat drift controls roughness and spectral distribution quality.

---

### 12. OT coupling visualization

The app includes an OT coupling visualization panel.

It can visualize:

```text
OT-CFM / Euclidean cost
OT-TFM / TFM transport cost
```

Displayed data:

```text
cost matrix C_ij
Sinkhorn transport plan P_ij
expected cost
plan entropy
mean row max mass
mean cost
top plan entries
most likely target per source row
```

This panel is useful for inspecting how OT-CFM and OT-TFM construct source-target pairings.

OT-CFM uses:

```text
C_ij = ||x0_i - x1_j||²
```

OT-TFM uses a Hodge spectral / heat semigroup-aware transport cost.

---

## Hodge energy diagnostics

The app evaluates generated edge signals using Hodge spectral metrics.

Given an edge signal:

```text
x ∈ R^{n_edges}
```

and an L1 Hodge Laplacian:

```text
L = U diag(λ) Uᵀ
```

the Hodge spectral coefficients are:

```text
y = Uᵀx
```

The Hodge energy is:

```text
xᵀLx = Σ_i λ_i y_i²
```

Implemented diagnostics:

```text
harmonic_energy
low_frequency_energy
high_frequency_energy
hodge_energy = xᵀLx
l2_norm = ||x||²
```

Interpretation:

```text
high_frequency_energy:
  generally smaller is better for smooth targets

hodge_energy:
  smaller means smoother under the Hodge Laplacian

harmonic_energy:
  should match the target μ1 harmonic profile; not blindly minimized

low_frequency_energy:
  should match the target μ1 structural profile; not blindly minimized

l2_norm:
  should match the target μ1 amplitude; not blindly minimized
```

A trivial zero signal would have low Hodge energy and low norm, but it would not be a good generated sample. Therefore, the goal is not to minimize all metrics to zero. The goal is to suppress rough modes while matching the target spectral profile.

---

## Sinkhorn coupling and `sinkhorn_epsilon`

OT-CFM and OT-TFM use entropic Sinkhorn coupling.

Given a cost matrix:

```text
C_ij = cost(x0_i, x1_j)
```

Sinkhorn computes a soft transport plan:

```text
P_ij
```

with approximately uniform row and column marginals.

The entropic OT objective is:

```text
min_P Σ_ij P_ij C_ij - ε H(P)
```

where:

```text
H(P) = -Σ_ij P_ij log P_ij
```

The parameter `sinkhorn_epsilon` acts like a temperature:

```text
small ε:
  sharper plan
  stronger preference for low-cost pairs
  closer to hard matching
  potentially less stable and less diverse

large ε:
  softer plan
  more diffuse coupling
  closer to independent pairing
  more stable but less cost-aware
```

The implementation uses the standard scaling form:

```text
K_ij = exp(-C_ij / ε)
P = diag(u) K diag(v)
```

with repeated updates:

```text
u = a / (K v)
v = b / (Kᵀ u)
```

---

## Backend implementation

The backend is implemented with:

```text
FastAPI
JAX
Flax
```

Main responsibilities:

- build the annulus complex,
- construct boundary matrices,
- compute the Hodge Laplacian,
- compute CFM / TFM bridge paths,
- train neural vector fields,
- run distribution-level generation,
- evaluate Hodge spectral metrics,
- compute distribution distances,
- run ablations and sweeps,
- expose JSON APIs to the frontend.

---

## Frontend implementation

The frontend is implemented with:

```text
React
TypeScript
Vite
```

Main responsibilities:

- render the annulus complex,
- display edge signals,
- display matrices,
- display spectra,
- control sliders and settings,
- trigger training / evaluation API calls,
- render comparison tables,
- render OT coupling heatmaps.

---

## Run locally

```bash
docker compose up --build
```

Frontend:

```text
http://localhost:5173
```

Backend docs:

```text
http://localhost:8000/docs
```

---

## API overview

### GET `/api/presets`

Returns preset source/target edge signals on the annulus complex.

---

### POST `/api/compute`

Computes:

```text
boundary matrices
Hodge Laplacian
eigenvalues / eigenvectors
CFM path
TFM path
CFM velocity
TFM bridge control
CFM / TFM transport costs
Figure-1-style visualization payload
```

---

### POST `/api/train_cfm`

Trains a distribution-level CFM baseline.

Supports:

```text
independent coupling
OT coupling
```

---

### POST `/api/generate_cfm`

Generates CFM samples and returns:

```text
source signal
generated signal
reference signal
single-sample metrics
batch aggregate metrics
distribution distances
```

---

### POST `/api/train_itfm`

Trains I-TFM:

```text
independent coupling + TFM dynamics
```

---

### POST `/api/generate_itfm`

Generates I-TFM samples and returns single-sample and batch diagnostics.

---

### POST `/api/train_ot_tfm`

Trains OT-TFM:

```text
Sinkhorn(TFM transport cost) coupling + TFM dynamics
```

---

### POST `/api/generate_ot_tfm`

Generates OT-TFM samples and returns diagnostics.

---

### POST `/api/run_mu0_ablation`

Runs the topology-aware initial distribution ablation:

```text
I-TFM / standard μ0
I-TFM / heat GP μ0
OT-TFM / standard μ0
OT-TFM / heat GP μ0
```

---

### POST `/api/run_distribution_metrics`

Runs four-way distribution distance evaluation:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
```

with:

```text
RBF MMD²
spectral sliced Wasserstein
spectral mode Wasserstein
```

---

### POST `/api/run_kappa_sweep`

Runs κ sensitivity analysis for:

```text
I-TFM
OT-TFM
```

---

### POST `/api/ot_coupling_visualization`

Builds OT coupling visualization payloads for:

```text
OT-CFM / Euclidean cost
OT-TFM / TFM transport cost
```

---

## Project layout

```text
topological-flow-matching-visualizer/
  backend/
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
    src/
      App.tsx
      api.ts
      types.ts
      style.css
      components/
        CFMPanel.tsx
        ComplexView.tsx
        Controls.tsx
        DistributionDistancePanel.tsx
        FigureOneView.tsx
        FourWayComparisonPanel.tsx
        InitialDistributionAblationPanel.tsx
        ITFMPanel.tsx
        KappaSweepPanel.tsx
        MatrixView.tsx
        OTCouplingVisualizationPanel.tsx
        OTTFMPanel.tsx
        SignalView.tsx
        SpectrumView.tsx
        TFMComparisonPanel.tsx

  tests/
```

---

## Conceptual summary

This repository is built around the following contrast:

```text
Euclidean flow matching:
  learns motion in signal space without explicitly respecting topology

Topological flow matching:
  uses Hodge Laplacian structure to define topology-aware smoothing,
  bridge dynamics, transport costs, and diagnostics
```

The core experimental question is:

```text
When does topology-aware geometry improve flow matching on edge signals?
```

This app provides several ways to inspect that question:

```text
visual inspection:
  edge signal plots and Figure-1-style panels

energy diagnostics:
  harmonic / low / high / Hodge / norm metrics

distribution diagnostics:
  MMD and spectral Wasserstein distances

ablation:
  standard μ0 vs heat GP μ0

sensitivity analysis:
  κ sweep

coupling inspection:
  OT-CFM vs OT-TFM cost and plan heatmaps
```

---

## Known limitations

- This is a synthetic annulus demo, not a full-scale benchmark.
- The neural models are intentionally small.
- Euler rollout is simple and may introduce discretization error.
- Sinkhorn coupling is minibatch-based.
- Several cutoff choices, such as low-frequency mode count, are diagnostic choices rather than universal definitions.
- Hyperparameters are tuned for local interpretability rather than exhaustive performance.
- The app prioritizes transparency and experimentation over computational efficiency.

---

## Suggested next experiments

Possible extensions include:

```text
sinkhorn_epsilon sweep
side-by-side OT-CFM vs OT-TFM coupling comparison on the same minibatch
larger / different simplicial complexes
alternative μ1 target distributions
higher-order Hodge Laplacians
better ODE solvers
learned score or drift variants
formal quantitative benchmark scripts
```

---

## License / citation note

If you use this repository as a reference, please cite the original Topological Flow Matching paper separately. This repository itself is an unofficial implementation and should not be cited as the official source of the paper's code.
