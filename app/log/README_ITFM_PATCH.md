# I-TFM distribution-level patch

This patch adds the first distribution-level I-TFM implementation.

Core additions:

- `backend/app/tfm/datasets.py`
- `backend/app/tfm/itfm.py`

These implement:

```text
(X0, X1) ~ μ0 ⊗ μ1
t ~ Uniform(0, 1)
x_t = TFM bridge path
target_u = analytical conditional TFM bridge control
uθ(t, x_t, κ) ≈ target_u
```

Important distinction from the single-pair fitting demo:

```text
Single-pair demo:
  input includes x0, x1
  purpose = overfit one conditional bridge

I-TFM:
  input does NOT include x1
  purpose = learn a distribution-level vector field usable for generation
```

Also included in the patch are updated schemas, API functions, and a frontend `ITFMPanel`.
