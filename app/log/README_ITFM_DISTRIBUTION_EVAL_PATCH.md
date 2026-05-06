# I-TFM distribution evaluation patch

This patch implements the requested changes:

1. Lower μ1 harmonic amplitude slightly.
2. Increase rollout steps from 80 to 160 by default.
3. Apply mild learned-control scaling and componentwise clipping during rollout.
4. Add n_eval=32 distribution-level mean/std diagnostics for source / generated / reference.

## Files included

```text
backend/app/tfm/datasets.py
backend/app/tfm/itfm.py
backend/app/tfm/itfm_metrics.py
backend/app/schemas.py
backend/app/main.py

frontend/src/types.ts
frontend/src/api.ts
frontend/src/components/ITFMPanel.tsx
frontend/src/style_itfm_distribution_eval_patch.css
```

## Apply

Copy files into the same paths.

Append:

```text
frontend/src/style_itfm_distribution_eval_patch.css
```

to:

```text
frontend/src/style.css
```

Then delete the patch CSS file if desired.

## New generation behavior

```text
rollout_steps = 160
n_eval = 32
control_scale = 0.92
```

The UI now shows:

```text
single displayed sample metrics
distribution mean ± std over n_eval samples
```
