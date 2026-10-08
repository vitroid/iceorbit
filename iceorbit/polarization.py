"""Polarization of a configuration: every hydrogen bond is a unit dipole from donor to acceptor.

Cage edge e=(u, v), u < v, contributes s_e * u_e with u_e the unit vector from u to v and
s_e = +1 if u donates (bit 0), -1 otherwise. The fourth (outer) bond of vertex v points along
d_v = -normalize(sum of unit vectors from v to its three cage neighbors); it is donated by v
(t_v = +1) when v has cage in-degree 2, and accepted (t_v = -1) when the in-degree is 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Cage

BYTE = 8
ZERO_TOL = 1e-8
ROUND_DECIMALS = 9


def bond_directions(cage: Cage) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors of cage edges (u -> v) and of the outer bond direction at each vertex."""
    x = cage.coords
    u, v = cage.edges[:, 0], cage.edges[:, 1]
    edge_dir = x[v] - x[u]
    edge_dir /= np.linalg.norm(edge_dir, axis=1, keepdims=True)

    inward = np.zeros_like(x)
    np.add.at(inward, u, edge_dir)
    np.add.at(inward, v, -edge_dir)
    norm = np.linalg.norm(inward, axis=1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("outer bond direction is undefined at a vertex whose cage bonds sum to zero")
    return edge_dir, -inward / norm


def linear_form(cage: Cage) -> tuple[np.ndarray, np.ndarray]:
    """(c, w) such that P(x) = c + sum_e bit_e(x) * w[e]."""
    edge_dir, outer = bond_directions(cage)
    u, v = cage.edges[:, 0], cage.edges[:, 1]
    w = -2 * edge_dir + 2 * outer[u] - 2 * outer[v]
    n_in0 = np.bincount(v, minlength=cage.n_vertices)
    c = edge_dir.sum(axis=0) + ((2 * n_in0 - 3)[:, None] * outer).sum(axis=0)
    return c, w


def polarization(configs: np.ndarray, cage: Cage, block: int = 1 << 20) -> np.ndarray:
    """Polarization vectors (N, 3) of the given configurations, via per-byte lookup tables."""
    configs = np.asarray(configs, dtype=np.uint64)
    c, w = linear_form(cage)
    n_e = cage.n_edges
    tables = []
    for s in range(0, n_e, BYTE):
        width = min(BYTE, n_e - s)
        t = np.zeros((1 << width, 3))
        for b in range(width):
            t[1 << b : 1 << (b + 1)] = t[: 1 << b] + w[s + b]
        tables.append(t)

    out = np.empty((len(configs), 3))
    mask = np.uint64((1 << BYTE) - 1)
    for i in range(0, len(configs), block):
        x = configs[i : i + block]
        p = np.broadcast_to(c, (len(x), 3)).copy()
        for k, t in enumerate(tables):
            p += t[((x >> np.uint64(BYTE * k)) & mask).astype(np.intp)]
        out[i : i + block] = p
    return out


def polarization_direct(config: int, cage: Cage) -> np.ndarray:
    """Reference implementation summing bond dipoles one by one."""
    edge_dir, outer = bond_directions(cage)
    p = np.zeros(3)
    n_in = np.zeros(cage.n_vertices, dtype=int)
    for e, (u, v) in enumerate(cage.edges):
        if (config >> e) & 1:
            p -= edge_dir[e]
            n_in[u] += 1
        else:
            p += edge_dir[e]
            n_in[v] += 1
    for v in range(cage.n_vertices):
        p += (1 if n_in[v] == 2 else -1) * outer[v]
    return p


@dataclass
class MagnitudeDistribution:
    values: np.ndarray  # (M,) distinct |P| (rounded), ascending
    n_configs: np.ndarray  # (M,) configurations with that |P|
    n_classes: np.ndarray  # (M,) classes whose members take that |P|
    total: int

    @property
    def probability(self) -> np.ndarray:
        return self.n_configs / self.total


def magnitude_distribution(mag: np.ndarray, class_id: np.ndarray) -> MagnitudeDistribution:
    rounded = np.round(mag, ROUND_DECIMALS)
    rounded[np.abs(mag) < ZERO_TOL] = 0.0
    values, inv, counts = np.unique(rounded, return_inverse=True, return_counts=True)
    pairs = np.unique(inv.astype(np.int64) * (int(class_id.max()) + 1) + class_id)
    n_classes = np.bincount(pairs // (int(class_id.max()) + 1), minlength=len(values))
    return MagnitudeDistribution(values, counts, n_classes, len(mag))
