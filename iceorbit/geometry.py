"""Cage geometry: vertex coordinates and the undirected hydrogen-bond graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Cage:
    """A 3-regular cage graph embedded in 3D.

    `edges[e] = (u, v)` with `u < v`; edges are sorted lexicographically.
    Bit `e` of a configuration is 0 when `u` donates to `v`, 1 when `v` donates to `u`.
    """

    coords: np.ndarray  # (V, 3) float
    edges: np.ndarray  # (E, 2) int, u < v

    @property
    def n_vertices(self) -> int:
        return len(self.coords)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def neighbors(self) -> list[list[int]]:
        nb: list[list[int]] = [[] for _ in range(self.n_vertices)]
        for u, v in self.edges:
            nb[u].append(int(v))
            nb[v].append(int(u))
        return [sorted(x) for x in nb]

    def incident_edges(self) -> list[list[int]]:
        inc: list[list[int]] = [[] for _ in range(self.n_vertices)]
        for e, (u, v) in enumerate(self.edges):
            inc[u].append(e)
            inc[v].append(e)
        return inc


def edges_by_distance(coords: np.ndarray, cutoff: float | None = None) -> np.ndarray:
    """Connect vertex pairs closer than `cutoff` (default: 1.25 x shortest distance)."""
    coords = np.asarray(coords, dtype=float)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    iu, ju = np.triu_indices(len(coords), k=1)
    dist = d[iu, ju]
    if cutoff is None:
        cutoff = 1.25 * dist.min()
    mask = dist < cutoff
    edges = np.stack([iu[mask], ju[mask]], axis=1)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


def make_cage(coords: np.ndarray, edges: np.ndarray | None = None, cutoff: float | None = None) -> Cage:
    coords = np.asarray(coords, dtype=float)
    if edges is None:
        edges = edges_by_distance(coords, cutoff)
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    cage = Cage(coords=coords, edges=edges)
    deg = np.bincount(edges.ravel(), minlength=cage.n_vertices)
    if not np.all(deg == 3):
        bad = np.nonzero(deg != 3)[0]
        raise ValueError(f"cage must be 3-regular; vertices with wrong degree: {bad.tolist()} (deg={deg[bad].tolist()})")
    if cage.n_edges > 64:
        raise ValueError(f"at most 64 edges are supported (got {cage.n_edges})")
    return cage


def dodecahedron_coords() -> np.ndarray:
    phi = (1 + 5**0.5) / 2
    pts = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a in (-1, 1):
        for b in (-1, 1):
            pts.append((0, a / phi, b * phi))
            pts.append((a / phi, b * phi, 0))
            pts.append((a * phi, 0, b / phi))
    return np.array(pts, dtype=float)


def cube_coords() -> np.ndarray:
    return np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


BUILTIN = {
    "dodecahedron": dodecahedron_coords,
    "cube": cube_coords,
}


def builtin_cage(name: str = "dodecahedron") -> Cage:
    try:
        coords = BUILTIN[name]()
    except KeyError:
        raise ValueError(f"unknown builtin cage {name!r}; choose from {sorted(BUILTIN)}") from None
    return make_cage(coords)


def read_xyz(path: str | Path) -> np.ndarray:
    """Read oxygen coordinates from an XYZ file; non-oxygen atoms (e.g. H) are ignored."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    coords = []
    for line in lines[2 : 2 + n]:
        f = line.split()
        if not f:
            continue
        elem = f[0].rstrip("0123456789").upper()
        if elem in ("O", "OW"):
            coords.append([float(f[1]), float(f[2]), float(f[3])])
    if not coords:
        raise ValueError(f"no oxygen atoms found in {path}")
    return np.array(coords)


def cage_from_xyz(path: str | Path, cutoff: float | None = None) -> Cage:
    return make_cage(read_xyz(path), cutoff=cutoff)
