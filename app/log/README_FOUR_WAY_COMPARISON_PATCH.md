# Four-way FM comparison patch

This patch adds a paper-style four-way comparison panel:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
reference μ1
```

## What the panel does

It runs:

```text
Train I-CFM
Train OT-CFM
Train I-TFM
Train OT-TFM
Generate each method with n_eval = 32
Compare generated distribution mean ± std against reference μ1
```

for the following metrics:

```text
harmonic energy
low-frequency energy
high-frequency energy
xᵀL₁x
||x||²
```

It also marks which method is closest to μ1 for each metric.

## Files included

```text
frontend/src/components/FourWayComparisonPanel.tsx
frontend/src/App.tsx
frontend/src/style_four_way_comparison_patch.css
```

## Apply

Copy files into the same project paths.

Append:

```text
frontend/src/style_four_way_comparison_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.
