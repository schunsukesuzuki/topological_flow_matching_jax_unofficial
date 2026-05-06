# Topology-aware initial distribution ablation patch

This patch implements roadmap item 3:

```text
Topology-Aware Initial Distribution Ablation
```

## What it adds

Two μ0 modes:

```text
standard:
  topology-unaware standard Gaussian source

heat_gp:
  topology-aware heat Gaussian process source
```

The ablation runs:

```text
I-TFM / standard μ0
I-TFM / heat GP μ0
OT-TFM / standard μ0
OT-TFM / heat GP μ0
```

and compares generated distribution statistics against reference μ1.

## New backend endpoint

```text
POST /api/run_mu0_ablation
```

## Files included

```text
backend/app/tfm/datasets.py
backend/app/tfm/itfm.py
backend/app/tfm/ottfm.py
backend/app/tfm/mu0_ablation.py
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/InitialDistributionAblationPanel.tsx
frontend/src/App.tsx
frontend/src/style_mu0_ablation_patch.css
```

## Apply

Copy files into the same project paths.

Append:

```text
frontend/src/style_mu0_ablation_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## Notes

The ablation uses a lighter default training setup than the full panels:

```text
n_samples = 1024
n_steps = 1200
n_eval = 32
rollout_steps = 160
```

This keeps the four-model ablation manageable in the local demo.
