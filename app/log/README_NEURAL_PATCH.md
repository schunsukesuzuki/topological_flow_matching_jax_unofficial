# Neural bridge-control patch

This patch adds a switch between:

- analytical closed-form TFM bridge control;
- neural BridgeMLP approximation u_theta(t, x_t).

Default mode is neural.

## Files

```text
backend/app/tfm/flax_model.py
backend/app/tfm/neural_bridge.py
backend/app/schemas.py
backend/app/main.py
frontend/src/types.ts
frontend/src/components/Controls.tsx
frontend/src/components/SignalView.tsx
frontend/src/App.tsx
```

Copy these files into the existing project, preserving the paths.

## Behavior

In neural mode, the backend:

1. computes analytical TFM bridge paths and controls;
2. builds a small deterministic supervised dataset over time samples;
3. trains BridgeMLP with SGD;
4. returns u_theta(t, x_t) as `tfm_u`;
5. also returns `analytical_tfm_u` as the teacher target for comparison.

In analytical mode, `tfm_u` is the closed-form TFM control directly.
