# I-TFM metrics patch

This patch adds quantitative spectral diagnostics for the I-TFM generated sample.

## Files included

```text
backend/app/tfm/itfm_metrics.py
backend/app/schemas.py
backend/app/main.py
frontend/src/types.ts
frontend/src/components/ITFMPanel.tsx
frontend/src/style_itfm_metrics_patch.css
```

## Apply

Copy files into the project with the same paths.

Append:

```text
frontend/src/style_itfm_metrics_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## Metrics

The I-TFM panel now shows:

```text
harmonic energy
low-frequency energy
high-frequency energy
xᵀL₁x
||x||²
```

for:

```text
source μ0
generated x̂1
reference μ1
```

This is more meaningful than pointwise comparison because I-TFM generation targets the μ1 distribution, not a specific reference sample.
