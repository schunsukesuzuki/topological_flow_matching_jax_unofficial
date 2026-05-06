# Distribution Distance Metrics Patch

This patch implements roadmap item 4:

```text
Distribution Distance Metrics
```

## Added metrics

```text
RBF MMD²
spectral sliced Wasserstein
spectral mode Wasserstein
```

## Interpretation

All three distances compare generated samples against reference μ1.

```text
lower is better
```

The spectral metrics first project signals into the Hodge eigenbasis, so they are more sensitive to
topological / spectral mismatch than a plain signal-space metric.

## New backend files

```text
backend/app/tfm/distribution_metrics.py
backend/app/tfm/distribution_metric_eval.py
```

## New backend endpoint

```text
POST /api/run_distribution_metrics
```

## Updated files

```text
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/DistributionDistancePanel.tsx
frontend/src/App.tsx
frontend/src/style_distribution_metrics_patch.css
```

## Apply

Copy files into the same project paths.

Append:

```text
frontend/src/style_distribution_metrics_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## Default panel settings

```text
n_samples = 1024
n_steps = 1200
n_eval = 32
rollout_steps = 160
```
