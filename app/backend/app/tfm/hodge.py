from typing import Optional
import jax.numpy as jnp


def hodge_laplacian(B_k: Optional[jnp.ndarray] = None, B_k_plus_1: Optional[jnp.ndarray] = None):
    """Build a Hodge Laplacian from boundary matrices.

    This function implements the standard k-Hodge Laplacian:

        L_k = B_k.T @ B_k + B_{k+1} @ B_{k+1}.T

    where:
        - B_k maps k-simplices to (k-1)-simplices.
        - B_{k+1} maps (k+1)-simplices to k-simplices.

    The two terms have different structural meanings:

        B_k.T @ B_k:
            The "down" Laplacian term. It captures how k-simplex signals
            interact through shared (k-1)-simplices.

        B_{k+1} @ B_{k+1}.T:
            The "up" Laplacian term. It captures how k-simplex signals
            interact through shared (k+1)-simplices.

    Examples:
        For node signals:
            L0 = hodge_laplacian(B_k=None, B_k_plus_1=B1)
               = B1 @ B1.T

        For edge signals:
            L1 = hodge_laplacian(B_k=B1, B_k_plus_1=B2)
               = B1.T @ B1 + B2 @ B2.T

        For top-dimensional face signals without higher simplices:
            L2 = hodge_laplacian(B_k=B2, B_k_plus_1=None)
               = B2.T @ B2

    In the TFM visualizer, this function is mainly used to construct L1,
    the Hodge Laplacian acting on edge signals. Its spectrum is then used for
    heat smoothing, zero/low/high-frequency modes, and TFM bridge paths.
    """
    # Collect the available Laplacian terms. Some dimensions only have a
    # lower-boundary term or an upper-boundary term.
    terms = []

    # Down Laplacian term:
    #   k-simplex signal -> boundary on (k-1)-simplices -> back to k-simplices.
    #
    # For edge signals, this is B1.T @ B1 and captures node-mediated structure.
    if B_k is not None:
        terms.append(B_k.T @ B_k)

    # Up Laplacian term:
    #   k-simplex signal -> lifted through (k+1)-simplices -> back to k-simplices.
    #
    # For edge signals, this is B2 @ B2.T and captures face-mediated structure.
    # The size check skips empty boundary matrices, such as B2 with zero faces.
    if B_k_plus_1 is not None and B_k_plus_1.size > 0:
        terms.append(B_k_plus_1 @ B_k_plus_1.T)

    # A Hodge Laplacian cannot be built if neither boundary matrix is provided.
    if not terms:
        raise ValueError("At least one boundary matrix is required.")

    # Initialize with the first available term.
    L = terms[0]

    # Add any remaining term. In the usual edge-signal case, this produces:
    #   L1 = B1.T @ B1 + B2 @ B2.T
    for term in terms[1:]:
        L = L + term

    return L
