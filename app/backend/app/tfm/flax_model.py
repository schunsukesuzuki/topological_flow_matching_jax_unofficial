from typing import Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp


class BridgeMLP(nn.Module):
    """MLP for learning bridge-control targets in the current annulus app.

    This version is slightly larger than the first neural patch because the
    current app predicts a 72-dimensional edge-control vector. The model is
    still small enough for an interactive educational demo, but has enough
    capacity to overfit the deterministic analytical bridge-control target for
    a single displayed source/target pair.

    Default annulus dimensions:
        edge signal x_t : 72
        output u_theta : 72

    The actual input dimension is determined at init time by Flax. In the
    improved training patch, features are richer than concat([t], x_t):

        concat([t, kappa], x_t, x0, x1, x1 - x0)

    For the default annulus demo this gives:
        2 + 72 * 4 = 290 input dimensions.
    """

    hidden_dims: Sequence[int] = (256, 256, 128)
    output_dim: int = 72

    @nn.compact
    def __call__(self, x):
        """Run the MLP forward pass."""
        h = x
        for dim in self.hidden_dims:
            h = nn.Dense(dim)(h)
            h = nn.silu(h)
        return nn.Dense(self.output_dim)(h)


def make_bridge_features(
    t: float,
    x_t: jnp.ndarray,
    x0: Optional[jnp.ndarray] = None,
    x1: Optional[jnp.ndarray] = None,
    kappa: Optional[float] = None,
):
    """Create neural input features for bridge-control fitting.

    The original minimal neural demo used only concat([t], x_t). That is enough
    to demonstrate the idea, but it makes the regression unnecessarily hard.
    This improved version optionally includes x0, x1, x1 - x0, and kappa.

    In the current app, x0/x1/kappa are fixed within one request, so this richer
    feature vector lets the MLP learn the analytical control much more accurately
    with a small number of deterministic training samples.

    Args:
        t: Scalar interpolation time in [0, 1].
        x_t: Current edge signal.
        x0: Optional source edge signal.
        x1: Optional target edge signal.
        kappa: Optional heat-drift strength.

    Returns:
        A 1D feature vector.
    """
    parts = [jnp.asarray([t], dtype=x_t.dtype)]

    if kappa is not None:
        parts.append(jnp.asarray([kappa], dtype=x_t.dtype))

    parts.append(x_t)

    if x0 is not None and x1 is not None:
        parts.extend([x0, x1, x1 - x0])

    return jnp.concatenate(parts, axis=0)


def bridge_mse_loss(pred_u: jnp.ndarray, target_u: jnp.ndarray):
    """Mean-squared error loss for bridge-control fitting."""
    return jnp.mean((pred_u - target_u) ** 2)
