"""Enumeration of all ice-rule edge orientations of a 3-regular cage."""

from __future__ import annotations

import numpy as np

from .geometry import Cage
from .symmetry import _bfs_order


def _edge_order(cage: Cage) -> list[int]:
    """Edges sorted so that vertices along a BFS order get all their edges fixed early."""
    order, _ = _bfs_order(cage.neighbors())
    pos = np.empty(cage.n_vertices, dtype=np.int64)
    pos[order] = np.arange(cage.n_vertices)
    pu, pv = pos[cage.edges[:, 0]], pos[cage.edges[:, 1]]
    hi, lo = np.maximum(pu, pv), np.minimum(pu, pv)
    return np.lexsort((lo, hi)).tolist()


def enumerate_ice_configs(cage: Cage, return_peak: bool = False):
    """All orientations where every vertex has cage in-degree 1 or 2, as sorted uint64 bitmasks.

    The array is grown one edge at a time (both directions), and each vertex whose three
    edges have all been fixed is filtered against the ice rule immediately.
    """
    inc = cage.incident_edges()
    order = _edge_order(cage)
    step_of_edge = {e: k for k, e in enumerate(order)}
    completes_at: list[list[int]] = [[] for _ in order]
    for v, es in enumerate(inc):
        completes_at[max(step_of_edge[e] for e in es)].append(v)

    one = np.uint64(1)
    x = np.zeros(1, dtype=np.uint64)
    peak = 1
    for k, e in enumerate(order):
        x = np.concatenate([x, x | (one << np.uint64(e))])
        peak = max(peak, len(x))
        for v in completes_at[k]:
            ins = []
            for f in inc[v]:
                bit = (x >> np.uint64(f)) & one
                # v receives the bond when bit=1 and v is the smaller endpoint, or bit=0 and v is larger
                larger = np.uint64(1 if cage.edges[f, 1] == v else 0)
                ins.append(bit ^ larger)
            s = ins[0] + ins[1] + ins[2]
            x = x[(s == 1) | (s == 2)]
    x.sort()
    return (x, peak) if return_peak else x
