import math
from .simplicial_complex import annulus_complex


def _edge_mid_angle(edge, n=18):
    a, b = edge
    ia, ib = a % n, b % n
    # unwrap the final segment so the phase changes smoothly around the ring
    if abs(ia - ib) > n / 2:
        if ia < ib:
            ia += n
        else:
            ib += n
    return 2.0 * math.pi * ((ia + ib) / 2.0) / n


def _edge_signal(kind: str, n=18):
    _, edges, _ = annulus_complex(n=n, with_faces=True)
    vals = []
    for idx, edge in enumerate(edges):
        a, b = edge
        theta = _edge_mid_angle(edge, n)
        is_radial = (a < n <= b) or (b < n <= a)
        if kind == "noisy":
            v = 0.75 * math.sin(7 * theta) + 0.35 * math.cos(11 * theta) + (0.55 if idx % 5 == 0 else -0.15)
        elif kind == "circulation":
            v = 1.0 if not is_radial else 0.05 * math.sin(theta)
        elif kind == "smooth_wave":
            v = math.sin(theta) + 0.35 * math.cos(2 * theta)
        elif kind == "checker":
            v = 1.0 if idx % 2 == 0 else -1.0
        else:
            v = 0.0
        vals.append(round(float(v), 4))
    return vals


PRESETS = [
    {
        "name": "Annulus: noisy edge field → harmonic circulation",
        "description": "Figure-1-like annulus complex. Source contains high-frequency edge noise; target is mostly circulation around the central hole.",
        "x0": _edge_signal("noisy"),
        "x1": _edge_signal("circulation"),
    },
    {
        "name": "Annulus: checkerboard → smooth wave",
        "description": "High-frequency alternating edge signal transported toward a low-frequency smooth mode.",
        "x0": _edge_signal("checker"),
        "x1": _edge_signal("smooth_wave"),
    },
]
