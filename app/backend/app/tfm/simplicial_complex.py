import math
from typing import List, Tuple

# A simplex is represented by a tuple of integer node IDs.
# Examples:
#   (0,)       -> 0-simplex / node
#   (0, 1)    -> 1-simplex / edge
#   (0, 1, 2) -> 2-simplex / triangular face
Simplex = Tuple[int, ...]


def triangle_complex(with_face: bool):
    """Return a minimal triangle-shaped simplicial complex.

    This is the smallest useful toy example for comparing a pure 1-cycle
    against a filled 2-simplex:

    - with_face=False: three edges form a loop, so a 1-dimensional hole exists.
    - with_face=True: the triangle face fills the loop, so the hole disappears.

    Returns:
        nodes: list of 0-simplices.
        edges: list of 1-simplices.
        faces: list of 2-simplices, either empty or containing one triangle.
    """
    nodes = [(0,), (1,), (2,)]
    edges = [(0, 1), (1, 2), (0, 2)]
    faces = [(0, 1, 2)] if with_face else []
    return nodes, edges, faces


def triangle_layout():
    """Return 2D coordinates for drawing the triangle complex.

    The coordinates are only for frontend/SVG visualization. They do not affect
    the boundary matrices, Hodge Laplacian, or any topology-aware computation.
    """
    return [[0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]]


def annulus_complex(n: int = 18, with_faces: bool = True):
    """Construct a small annulus-shaped 2-simplicial complex.

    This complex is used for the Figure-1-style visualization. It consists of
    two circular rings of nodes:

    - nodes 0..n-1 are on the outer ring.
    - nodes n..2n-1 are on the inner ring.

    The edge set contains four edge types per annulus sector:

    - outer ring edges,
    - inner ring edges,
    - radial edges connecting outer and inner rings,
    - diagonal edges used to split each quadrilateral sector into two triangles.

    If with_faces=True, each annulus sector is filled with two triangular faces.
    The central hole is deliberately left unfilled, so the 1-Hodge Laplacian can
    expose a visible harmonic edge mode circulating around the hole.

    Args:
        n: Number of nodes on each ring. The total number of nodes is 2 * n.
        with_faces: Whether to include triangular 2-simplices in the annulus.

    Returns:
        nodes: list of 0-simplices.
        edges: sorted list of 1-simplices.
        faces: sorted list of 2-simplices if with_faces=True; otherwise empty.
    """
    nodes: List[Simplex] = [(i,) for i in range(2 * n)]
    edge_set = set()

    def add_edge(a: int, b: int):
        """Add an undirected edge using canonical sorted orientation.

        Sorting makes the representation deterministic and avoids duplicate
        edges such as (1, 3) and (3, 1). The actual orientation used by the
        boundary matrix is handled later by the boundary construction code.
        """
        edge_set.add(tuple(sorted((a, b))))

    for i in range(n):
        # Wrap around at the final sector so the rings are closed.
        j = (i + 1) % n

        outer_i, outer_j = i, j
        inner_i, inner_j = n + i, n + j

        # Boundary of the outer ring.
        add_edge(outer_i, outer_j)

        # Boundary of the inner ring.
        add_edge(inner_i, inner_j)

        # Radial connector between the outer and inner rings.
        add_edge(outer_i, inner_i)

        # Diagonal edge used to divide one annulus sector into two triangles:
        #
        #   outer_i ---- outer_j
        #      |       /    |
        #      |     /      |
        #   inner_i ---- inner_j
        add_edge(outer_j, inner_i)

    # Convert to a sorted list so edge-indexed signals are stable across runs.
    # This is important because an edge signal x in R^|E| assumes that each
    # component always refers to the same edge.
    edges = sorted(edge_set)

    faces: List[Simplex] = []
    if with_faces:
        for i in range(n):
            j = (i + 1) % n

            outer_i, outer_j = i, j
            inner_i, inner_j = n + i, n + j

            # Each quadrilateral annulus sector is triangulated into two faces.
            # The central annulus hole is not filled.
            faces.append(tuple(sorted((outer_i, outer_j, inner_i))))
            faces.append(tuple(sorted((outer_j, inner_i, inner_j))))

        # Remove any accidental duplicates and stabilize face ordering.
        faces = sorted(set(faces))

    return nodes, edges, faces


def annulus_layout(n: int = 18):
    """Return 2D coordinates for drawing the annulus complex.

    Coordinates follow the same node ordering as annulus_complex:

    - first n coordinates: outer ring, radius 1.0;
    - next n coordinates: inner ring, radius 0.55.

    These coordinates are used only for visualization. They are not involved in
    constructing B1, B2, L0, L1, or the TFM/CFM paths.
    """
    coords = []

    # Draw the outer ring first and the inner ring second to match node IDs.
    for radius in (1.0, 0.55):
        for i in range(n):
            # Equally spaced points around the circle.
            # The -pi/2 shift places the first node near the top of the drawing.
            theta = 2.0 * math.pi * i / n - math.pi / 2.0
            coords.append([radius * math.cos(theta), radius * math.sin(theta)])

    return coords
