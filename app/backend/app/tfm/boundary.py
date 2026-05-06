from typing import Sequence, Tuple
import jax.numpy as jnp

# A simplex is represented as a tuple of integer node IDs.
# Examples:
#   (0,)       -> 0-simplex / node
#   (0, 1)    -> 1-simplex / edge
#   (0, 1, 2) -> 2-simplex / triangular face
Simplex = Tuple[int, ...]


def build_boundary_matrix(lower_simplices: Sequence[Simplex], higher_simplices: Sequence[Simplex]):
    """Build the boundary matrix from higher-dimensional simplices to lower ones.

    The returned matrix represents the boundary operator

        ∂_k : C_k -> C_{k-1}

    where:
        - lower_simplices are the (k-1)-simplices and become matrix rows.
        - higher_simplices are the k-simplices and become matrix columns.

    Examples:
        build_boundary_matrix(nodes, edges) returns B1: edge -> node.
        build_boundary_matrix(edges, faces) returns B2: face -> edge.

    Each column stores the signed boundary of one higher simplex using the
    standard oriented-simplex formula:

        ∂[v0, ..., vk] =
            sum_j (-1)^j [v0, ..., v_{j-1}, v_{j+1}, ..., vk]

    The output is a JAX array so it can be used directly in downstream Hodge
    Laplacian computations such as:

        L1 = B1.T @ B1 + B2 @ B2.T
    """
    # Map each lower simplex to its row index.
    # This allows us to place each boundary face in the correct matrix row.
    lower_index = {tuple(s): i for i, s in enumerate(lower_simplices)}

    # Rows correspond to lower simplices.
    # Columns correspond to higher simplices.
    B = jnp.zeros((len(lower_simplices), len(higher_simplices)), dtype=jnp.float32)

    # Each higher simplex contributes one column to the boundary matrix.
    for col, simplex in enumerate(higher_simplices):
        simplex = tuple(simplex)

        # A simplex with k + 1 vertices has dimension k.
        # Example:
        #   len((0, 1)) = 2 -> k = 1 -> edge
        #   len((0, 1, 2)) = 3 -> k = 2 -> triangle face
        k = len(simplex) - 1

        # Generate all (k-1)-faces by removing one vertex at a time.
        for j in range(k + 1):
            # Remove the j-th vertex from the simplex.
            # Example:
            #   simplex = (0, 1, 2), j = 1 -> face = (0, 2)
            face = simplex[:j] + simplex[j + 1 :]

            # Normalize the face representation to the canonical sorted order.
            # This matches the way simplices are stored elsewhere in the app.
            #
            # Note:
            #   For a fully general oriented simplicial complex, sorting can
            #   require an additional sign correction. In this visualizer, all
            #   simplices are stored with a canonical sorted orientation, so this
            #   simple normalization is sufficient.
            face = tuple(sorted(face))

            # Standard boundary sign from the oriented-simplex formula.
            sign = (-1.0) ** j

            # Locate the row corresponding to this boundary face.
            row = lower_index[face]

            # JAX arrays are immutable, so use the functional update syntax.
            # This sets B[row, col] to either +1 or -1.
            B = B.at[row, col].set(sign)

    return B
