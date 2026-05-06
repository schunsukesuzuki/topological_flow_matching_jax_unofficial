# I-TFM App Patch

This patch wires I-TFM into the FastAPI backend and React frontend.

## Files included

```text
backend/app/tfm/datasets.py
backend/app/tfm/itfm.py
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/ITFMPanel.tsx
frontend/src/App.tsx
frontend/src/style_itfm_patch.css
```

## How to apply

Copy files into the existing project using the same paths.

Also append the contents of:

```text
frontend/src/style_itfm_patch.css
```

to your existing:

```text
frontend/src/style.css
```

## New backend endpoints

```text
POST /api/train_itfm
POST /api/generate_itfm
```

## What the UI adds

A new card:

```text
I-TFM distribution-level training
```

with:

```text
Train I-TFM
Generate sample
```

The generated visualization shows:

```text
source x0 ~ μ0
generated x̂1
reference target x1 ~ μ1
```

## Key difference from the single-pair fitting demo

The I-TFM model input does not include x1:

```text
uθ(t, x_t, κ)
```

This makes it usable at generation time, where the target sample is unknown.
