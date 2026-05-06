import jax.numpy as jnp


def cfm_path(x0, x1, t):
    """Compute the Conditional Flow Matching (CFM) straight-line path.

    CFM treats x0 and x1 as points in a Euclidean vector space and connects them
    by linear interpolation:

        x_t = (1 - t) * x0 + t * x1

    This path does not use any graph, simplicial-complex, or Hodge-Laplian
    structure. In the visualizer, this is the baseline path against which the
    topology-aware TFM path is compared.
    """
    return (1.0 - t) * x0 + t * x1


def cfm_velocity(x0, x1):
    """Compute the constant CFM velocity.

    Since the CFM path is

        x_t = (1 - t) * x0 + t * x1,

    differentiating with respect to t gives

        d x_t / dt = x1 - x0.

    Therefore the conditional vector field for CFM is constant in time and does
    not depend on the intermediate point x_t.
    """
    return x1 - x0


def cfm_transport_cost(x0, x1):
    """Squared L2 transport cost for the constant-velocity CFM bridge.

    For x_t = (1 - t) x0 + t x1, the velocity is constant:
        u_t = x1 - x0
    Therefore the action / transport cost is:
        int_0^1 ||u_t||^2 dt = ||x1 - x0||^2

    In other words, the CFM transport cost is just the squared Euclidean
    distance between x0 and x1.
    """
    u = cfm_velocity(x0, x1)
    return jnp.sum(u**2)


def tfm_path_spectral(x0, x1, L, kappa, t, eps=1e-8):
    """Compute the deterministic Topological Flow Matching bridge path.

    This function implements the spectral TFM bridge path for the heat-drift
    reference process

        dX_t / dt = -kappa * L * X_t + control,

    where L is a Hodge Laplacian acting on the signal space, for example the
    1-Hodge Laplacian L1 acting on edge signals.

    The key idea is to diagonalize L and compute the bridge independently in
    each Hodge eigenmode. If

        L = U diag(lambda_i) U.T,

    then x0 and x1 are transformed into spectral coordinates:

        y0 = U.T @ x0
        y1 = U.T @ x1

    In each mode, omega_i = kappa * lambda_i controls the strength of the heat
    drift. Zero modes and kappa = 0 reduce to the ordinary CFM linear path.
    Nonzero modes use the hyperbolic-sine bridge formula:

        y_t^i =
            sinh(omega_i * (1 - t)) / sinh(omega_i) * y0^i
          + sinh(omega_i * t)       / sinh(omega_i) * y1^i

    The result is transformed back to the original signal coordinates by U @ y_t.
    """
    # Diagonalize the Hodge Laplacian. Its eigenvectors define the spectral
    # coordinates where heat drift acts independently per mode.
    evals, U = jnp.linalg.eigh(L)

    # Project source and target signals into the Hodge eigenbasis.
    y0 = U.T @ x0
    y1 = U.T @ x1

    # Per-mode heat-drift strength. Large eigenvalues correspond to
    # high-frequency modes and are affected more strongly by the heat reference.
    omega = kappa * evals

    # Linear path used for zero / near-zero modes, including harmonic
    # topological components where lambda_i = 0.
    linear = (1.0 - t) * y0 + t * y1

    # TFM bridge path for nonzero modes under heat drift.
    denom = jnp.sinh(omega)
    coeff0 = jnp.sinh(omega * (1.0 - t)) / (denom + eps)
    coeff1 = jnp.sinh(omega * t) / (denom + eps)
    sinh_path = coeff0 * y0 + coeff1 * y1

    # Avoid numerical instability when omega is close to zero. The theoretical
    # limit is exactly the linear CFM path.
    use_linear = jnp.abs(omega) < eps
    y_t = jnp.where(use_linear, linear, sinh_path)

    # Return from spectral coordinates to the original signal coordinates.
    return U @ y_t


def tfm_bridge_control_spectral(x0, x1, L, kappa, t, eps=1e-8):
    """Compute the spectral TFM bridge control target.

    TFM augments the flow ODE with a topology-aware heat drift:

        dX_t / dt = -kappa * L * X_t + u_t(X_t)

    The role of u_t is the corrective bridge control that steers the system
    from x0 to x1 while the reference dynamics naturally follow heat diffusion.

    For zero / near-zero modes, the control reduces to the CFM velocity:

        u_y^i = y1^i - y0^i

    For nonzero modes, the heat-drift-aware control is:

        u_y^i =
            [2 * omega_i * exp(-omega_i * (1 - t))
             / (1 - exp(-2 * omega_i))]
            * [y1^i - exp(-omega_i) * y0^i]

    where omega_i = kappa * lambda_i. The residual term measures how far the
    target is from where the source would end up under the uncontrolled heat
    reference process.
    """
    # Spectral decomposition of the Hodge Laplacian.
    evals, U = jnp.linalg.eigh(L)

    # Move source and target into the eigenbasis of L.
    y0 = U.T @ x0
    y1 = U.T @ x1
    omega = kappa * evals

    # For zero modes or kappa = 0, the topology-aware drift vanishes and the
    # bridge control is the same as the CFM constant velocity.
    linear_u = y1 - y0

    # Heat evolution factors. exp_full corresponds to evolution from t=0 to t=1;
    # exp_remaining corresponds to evolution over the remaining interval [t, 1].
    exp_full = jnp.exp(-omega)
    exp_remaining = jnp.exp(-omega * (1.0 - t))

    # Spectral bridge-control coefficient for nonzero heat-drift modes.
    numerator = 2.0 * omega * exp_remaining
    denominator = 1.0 - jnp.exp(-2.0 * omega)
    coeff = numerator / (denominator + eps)

    # Difference between the target and the source evolved by the uncontrolled
    # heat reference process.
    residual = y1 - exp_full * y0

    # Mode-wise TFM bridge control.
    spectral_u = coeff * residual

    # Use the stable linear limit for zero / near-zero modes.
    use_linear = jnp.abs(omega) < eps
    u_y = jnp.where(use_linear, linear_u, spectral_u)

    # Return to the original signal coordinates.
    return U @ u_y


def tfm_transport_cost(x0, x1, L, kappa, eps=1e-8):
    """Compute the TFM transport cost induced by the heat reference process.

    Unlike CFM, whose transport cost is the Euclidean squared distance
    ||x1 - x0||^2, TFM measures the cost relative to the topology-aware heat
    reference dynamics.

    After diagonalizing the Hodge Laplacian, each spectral mode contributes:

        c_i = (y1^i - y0^i)^2

    when omega_i = kappa * lambda_i is zero or near zero, and otherwise:

        c_i =
            [2 * omega_i / (1 - exp(-2 * omega_i))]
            * [y1^i - exp(-omega_i) * y0^i]^2

    The total TFM transport cost is the sum over all modes.
    """
    # Diagonalize L and move signals into Hodge spectral coordinates.
    evals, U = jnp.linalg.eigh(L)
    y0 = U.T @ x0
    y1 = U.T @ x1
    omega = kappa * evals

    # Linear CFM-like cost for zero / near-zero modes.
    linear_cost = (y1 - y0) ** 2

    # Residual relative to the uncontrolled heat evolution from x0 to time 1.
    exp_full = jnp.exp(-omega)
    residual = y1 - exp_full * y0

    # Spectral TFM cost coefficient for nonzero modes.
    coeff = 2.0 * omega / (1.0 - jnp.exp(-2.0 * omega) + eps)
    spectral_cost = coeff * residual**2

    # Use the stable linear limit for zero / near-zero modes.
    use_linear = jnp.abs(omega) < eps
    c_i = jnp.where(use_linear, linear_cost, spectral_cost)

    # Sum all modal contributions into a scalar cost.
    return jnp.sum(c_i)
