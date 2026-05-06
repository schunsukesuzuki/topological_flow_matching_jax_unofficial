# Remaining Implementation Roadmap for the TFM Visualizer

This document summarizes the remaining implementation candidates for the current Topological Flow Matching Visualizer.

The current App already covers the core path:

```text
closed-form conditional TFM
→ neural single-pair fitting
→ I-TFM distribution-level training
→ OT-TFM distribution-level training
→ I-TFM vs OT-TFM comparison
```

The next goal is to expand the App from a TFM-focused implementation demo into a more complete miniature reproduction of the paper’s experimental comparison structure.

---

## 1. I-CFM / OT-CFM Distribution-Level Baselines

### Priority

Highest.

### Motivation

The current App already has CFM single-pair visualization, but it does **not** yet have distribution-level CFM baselines equivalent to the I-TFM / OT-TFM implementations.

The paper compares methods such as:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
TSBM
```

Therefore, to make the App match the paper’s comparison structure, the next most important implementation is:

```text
Train I-CFM
Train OT-CFM
Generate CFM samples
Compare against I-TFM / OT-TFM / reference μ1
```

### Implementation outline

Add:

```text
backend/app/tfm/cfm_distribution.py
frontend/src/components/CFMPanel.tsx
```

Core logic:

```text
I-CFM:
  (X0, X1) ~ μ0 ⊗ μ1
  x_t = (1 - t)x0 + tx1
  target_u = x1 - x0

OT-CFM:
  pair X0, X1 using Euclidean transport cost
  x_t = (1 - t)x0 + tx1
  target_u = x1 - x0
```

Then train:

```text
uθ(t, x_t, κ) -> target_u
```

Unlike TFM, CFM has no Hodge heat drift in the rollout:

```text
dx/dt = uθ(t, x)
```

### Expected result

This will allow a direct test of:

```text
Does topology-aware TFM outperform ordinary CFM on structured edge signals?
```

---

## 2. Four-Way Method Comparison Table

### Priority

Very high.

### Motivation

The current App already compares:

```text
I-TFM generated
OT-TFM generated
reference μ1
```

The next step is to extend this to:

```text
I-CFM generated
OT-CFM generated
I-TFM generated
OT-TFM generated
reference μ1
```

This would make the App much closer to the paper’s experimental layout.

### Implementation outline

Add a new comparison panel:

```text
frontend/src/components/FMMethodComparisonPanel.tsx
```

Display:

```text
metric
I-CFM generated mean ± std
OT-CFM generated mean ± std
I-TFM generated mean ± std
OT-TFM generated mean ± std
reference μ1 mean ± std
best method
```

Metrics:

```text
harmonic energy
low-frequency energy
high-frequency energy
xᵀL₁x
||x||²
spectral Wasserstein or MMD, if implemented
```

### Expected result

This would show whether:

```text
OT-TFM > I-TFM > OT-CFM / I-CFM
```

or whether some metrics favor different couplings.

---

## 3. Topology-Aware Initial Distribution Ablation

### Priority

High.

### Motivation

The paper emphasizes the role of topology-aware initial distributions, especially heat Gaussian processes.

The current App already uses a heat-smoothed Gaussian-like μ0, but it does not compare against standard Gaussian noise.

### Implementation outline

Add a μ0 mode switch:

```text
μ0 mode:
  Standard Gaussian
  Heat GP
```

Then compare:

```text
I-TFM with standard Gaussian μ0
I-TFM with heat GP μ0
OT-TFM with standard Gaussian μ0
OT-TFM with heat GP μ0
```

### Backend changes

Extend:

```text
sample_mu0_heat_noise(...)
```

with a mode argument:

```text
sample_mu0(key, n_samples, evals, U, mode)
```

Modes:

```text
standard:
  y_i ~ N(0, 1)

heat_gp:
  y_i ~ exp(-0.5 κ_heat λ_i) N(0, 1)
```

### Expected result

This directly tests whether topology-aware initialization improves generation quality.

---

## 4. Distribution Distance Metrics

### Priority

High.

### Motivation

Current metrics are interpretable but hand-designed:

```text
harmonic energy
low-frequency energy
high-frequency energy
xᵀL₁x
||x||²
```

The paper uses distribution-level metrics such as Wasserstein-style distances. To make the App more paper-like, add distribution distances.

### Candidate metrics

Recommended:

```text
spectral sliced Wasserstein
```

Other options:

```text
MMD in signal space
MMD in Hodge spectral space
mean nearest-neighbor distance
1D Wasserstein over selected spectral coefficients
```

### Implementation outline

Add:

```text
backend/app/tfm/distribution_metrics.py
```

Implement:

```text
spectral_sliced_wasserstein(generated_batch, reference_batch, U, n_projections)
mmd_rbf(generated_batch, reference_batch)
```

### Expected result

The comparison table can include one or more scalar distribution distances:

```text
lower is better
```

This gives a more formal view than energy summaries alone.

---

## 5. κ Sweep / Sensitivity Panel

### Priority

Medium-high.

### Motivation

The TFM heat drift is:

```text
dx/dt = -κLx + uθ(t, x)
```

κ controls the strength of topology-aware smoothing. The current App has a κ slider, but no systematic comparison.

### Implementation outline

Add a panel that trains or evaluates across:

```text
κ = 0.0
κ = 0.5
κ = 1.0
κ = 2.0
κ = 4.0
```

Show:

```text
κ vs harmonic energy
κ vs low-frequency energy
κ vs high-frequency energy
κ vs xᵀLx
κ vs spectral Wasserstein / MMD
```

### Expected result

This will show how topology-aware heat drift affects:

```text
high-frequency suppression
harmonic component preservation
overall distribution matching
```

---

## 6. OT Coupling Visualization

### Priority

Medium-high.

### Motivation

OT-TFM now works numerically, but the coupling itself is not visible. The App currently shows only:

```text
mean pair cost
ot_batch_size
sinkhorn_epsilon
```

To explain OT-TFM more clearly, visualize the minibatch coupling.

### Implementation outline

For a sampled OT batch, return:

```text
cost matrix C_ij = c_TFM(x0_i, x1_j)
Sinkhorn plan P_ij
sampled pair indices
```

Frontend visualizations:

```text
cost matrix heatmap
Sinkhorn plan heatmap
sampled pair list or connecting lines
```

### Expected result

This would make the difference between I-TFM and OT-TFM visually obvious:

```text
I-TFM:
  random independent pairing

OT-TFM:
  pair selection biased by low topological transport cost
```

---

## 7. Node Signal / Edge Signal Mode Switch

### Priority

Medium.

### Motivation

The current App focuses mainly on edge signals on an annulus-shaped 2-simplicial complex. The paper covers both graph node signals and simplicial signals.

Add a switch between:

```text
k = 0:
  node signals
  L0 = B1 B1ᵀ

k = 1:
  edge signals
  L1 = B1ᵀ B1 + B2 B2ᵀ
```

### Implementation outline

Add signal order mode:

```text
signal_order:
  node
  edge
```

Backend:

```text
build_laplacian(k=0 or k=1)
sample_mu0 for node/edge
sample_mu1 for node/edge
train/generate for selected order
```

Frontend:

```text
Node TFM
Edge TFM
```

### Expected result

This would clarify the distinction between ordinary graph TFM and higher-order simplicial TFM.

---

## 8. Loss Curves and Training Diagnostics

### Priority

Medium.

### Motivation

The current App displays only final loss values:

```text
normalized MSE
displayed-scale MSE
mean pair cost
```

For training behavior, add loss curves and richer diagnostics.

### Implementation outline

Backend should return:

```text
loss_history
pair_cost_mean
pair_cost_std
pair_cost_min
pair_cost_max
```

Frontend should display:

```text
loss curve
pair cost histogram or summary table
```

For OT-TFM, especially useful diagnostics are:

```text
Sinkhorn ε
mean pair cost
pair cost variance
```

### Expected result

This makes it easier to debug and explain:

```text
why OT-TFM learns differently from I-TFM
whether training has converged
whether pair costs are reasonable
```

---

## 9. Model Persistence / Reuse / Export

### Priority

Medium-low.

### Motivation

The current App uses in-memory model states. This is fine for a local demo, but not convenient for repeated experiments.

### Implementation outline

Add:

```text
Save model
Load model
Export metrics JSON
Export generated samples
Export coupling matrix
```

Backend:

```text
save Flax params
load Flax params
save normalization stats
save training config
```

Possible files:

```text
models/itfm_state.pkl
models/ottfm_state.pkl
exports/metrics.json
exports/generated_samples.npy
```

### Expected result

This turns the App from a pure visual demo into a small experimental tool.

---

## 10. Real-World-Like Synthetic Dataset Modes

### Priority

Medium-low, but useful for presentation.

### Motivation

The current dataset is a synthetic annulus. The paper discusses structured datasets such as traffic flows, ocean currents, brain signals, and seismic events.

Before using real datasets, the App can add realistic synthetic presets.

### Candidate presets

```text
traffic-like edge flow on a road graph
ocean-like vector field on a mesh
brain-like node signal on a graph
seismic-like event intensity on a spatial graph
```

### Implementation outline

Add dataset modes:

```text
annulus_harmonic
traffic_grid
ocean_vortex
brain_graph
seismic_radial
```

Each mode should define:

```text
complex / graph
node or edge signal domain
μ0
μ1
visual layout
metrics
```

### Expected result

This makes the App easier to connect to the paper’s application domains while still remaining self-contained.

---

# Recommended Implementation Order

The most natural next implementation path is:

```text
1. I-CFM / OT-CFM distribution-level baselines
2. Four-way comparison table:
   I-CFM / OT-CFM / I-TFM / OT-TFM / reference
3. Distribution distance metrics:
   spectral sliced Wasserstein / MMD
4. Topology-aware μ0 ablation:
   standard Gaussian vs heat GP
5. κ sweep panel
6. OT coupling visualization:
   cost matrix + Sinkhorn plan
7. Node signal / edge signal mode switch
8. Loss curves and training diagnostics
9. Model persistence and export
10. Real-world-like synthetic dataset modes
```

The most important next step is:

```text
I-CFM / OT-CFM distribution-level baseline
```

because this completes the paper’s central comparison structure:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
```

Once this is implemented, the App will be able to demonstrate not only that OT-TFM improves over I-TFM, but also that topology-aware TFM improves over ordinary CFM on structured simplicial signals.
