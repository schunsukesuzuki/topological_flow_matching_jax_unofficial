# I-CFM / OT-CFM Distribution-Level Baselines Patch

This patch adds ordinary CFM baselines at the same distribution level as the existing I-TFM / OT-TFM panels.

## Implemented

```text
I-CFM:
  (X0, X1) ~ μ0 ⊗ μ1
  x_t = (1 - t)x0 + tx1
  target_u = x1 - x0
  rollout: dx/dt = uθ(t, x)

OT-CFM:
  source / target pools are coupled with entropic Sinkhorn using Euclidean cost
  x_t = (1 - t)x0 + tx1
  target_u = x1 - x0
  rollout: dx/dt = uθ(t, x)
```

Unlike TFM, CFM generation does **not** use the Hodge heat drift term `-κLx`.

## New backend endpoints

```text
POST /api/train_cfm
POST /api/generate_cfm
```

## Files included

```text
backend/app/tfm/cfm_distribution.py
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/CFMPanel.tsx
frontend/src/App.tsx
frontend/src/style_cfm_baselines_patch.css
```

## Apply

Copy files into the same project paths.

Append:

```text
frontend/src/style_cfm_baselines_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## Next step

After this patch, the App has the methods needed for the four-way comparison:

```text
I-CFM
OT-CFM
I-TFM
OT-TFM
```

The next roadmap item should be the unified four-way comparison panel.
