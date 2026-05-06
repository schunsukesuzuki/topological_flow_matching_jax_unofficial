# κ Sweep / Hodge Heat-Drift Sensitivity Patch

This patch implements the κ sweep panel.

## Purpose

It evaluates how the Hodge heat drift strength:

```text
dx/dt = -κLx + uθ(t, x)
```

affects generated distribution quality.

## Methods compared

```text
I-TFM
OT-TFM
```

## Default κ values

```text
0.25, 0.5, 1.0, 2.0, 4.0
```

## New backend file

```text
backend/app/tfm/kappa_sweep.py
```

## New backend endpoint

```text
POST /api/run_kappa_sweep
```

## Updated files

```text
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/KappaSweepPanel.tsx
frontend/src/App.tsx
frontend/src/style_kappa_sweep_patch.css
```

## Apply

Copy files into the same project paths.

Append:

```text
frontend/src/style_kappa_sweep_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## Default panel settings

```text
n_samples = 768
n_steps = 900
n_eval = 32
rollout_steps = 160
μ0 = heat_gp
```
