# OT Coupling Visualization Patch

This patch implements roadmap item 6:

```text
OT Coupling Visualization
```

## What it adds

A new UI panel:

```text
OT coupling visualization
```

It can visualize:

```text
OT-CFM / Euclidean cost
OT-TFM / TFM transport cost
```

## New backend file

```text
backend/app/tfm/coupling_visualization.py
```

## New backend endpoint

```text
POST /api/ot_coupling_visualization
```

## Displayed data

```text
cost matrix C_ij
Sinkhorn transport plan P_ij
expected cost
plan entropy
mean row max mass
top plan entries
most likely target per source row
```

## Updated files

```text
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/OTCouplingVisualizationPanel.tsx
frontend/src/App.tsx
frontend/src/style_ot_coupling_visualization_patch.css
```

## Apply

Copy files into the same project paths.

Append:

```text
frontend/src/style_ot_coupling_visualization_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## Notes

The default batch size is 16 to keep the heatmaps readable.
