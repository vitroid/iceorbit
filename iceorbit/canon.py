"""Canonical forms under the symmetry group, stabilizers and orbit (class) statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .symmetry import Group

CHUNK_BITS = 16
CHUNK_MASK = np.uint64((1 << CHUNK_BITS) - 1)


def _chunk_tables(edge_perm_g: np.ndarray, n_edges: int) -> list[np.ndarray]:
    """For each 16-bit chunk of the input, a table mapping chunk value -> permuted bits."""
    tables = []
    for c in range(0, n_edges, CHUNK_BITS):
        width = min(CHUNK_BITS, n_edges - c)
        t = np.zeros(1 << width, dtype=np.uint64)
        for b in range(width):
            t[1 << b : 1 << (b + 1)] = t[: 1 << b] | np.uint64(1 << int(edge_perm_g[c + b]))
        tables.append(t)
    return tables


def apply(x: np.ndarray, tables: list[np.ndarray], flip: np.uint64) -> np.ndarray:
    out = tables[0][(x & CHUNK_MASK).astype(np.intp)]
    for i, t in enumerate(tables[1:], start=1):
        out |= t[((x >> np.uint64(CHUNK_BITS * i)) & CHUNK_MASK).astype(np.intp)]
    out ^= flip
    return out


def act(group: Group, g: int, x: np.ndarray) -> np.ndarray:
    """Image of configurations `x` under group element `g`."""
    x = np.asarray(x, dtype=np.uint64)
    return apply(x, _chunk_tables(group.edge_perm[g], group.edge_perm.shape[1]), group.flip[g])


@dataclass
class OrbitResult:
    configs: np.ndarray  # (N,) uint64, all configurations
    canon: np.ndarray  # (N,) uint64, canonical form (min over the orbit)
    class_id: np.ndarray  # (N,) int, index into the class arrays
    representatives: np.ndarray  # (K,) uint64, canonical form of each class
    degeneracy: np.ndarray  # (K,) int, orbit size
    stabilizer_order: np.ndarray  # (K,) int
    stabilizer_improper: np.ndarray  # (K,) bool, stabilizer contains a reflection-type element
    fixed_counts: np.ndarray  # (G,) int, number of configurations fixed by each g
    group_order: int

    @property
    def n_classes(self) -> int:
        return len(self.representatives)

    def burnside_classes(self) -> float:
        return self.fixed_counts.sum() / self.group_order


def classify(configs: np.ndarray, group: Group, block: int = 1 << 20) -> OrbitResult:
    """Canonicalize every configuration as the minimum image over `group` and group into classes."""
    configs = np.asarray(configs, dtype=np.uint64)
    n, n_e = len(configs), group.edge_perm.shape[1]
    canon = configs.copy()
    stab = np.zeros(n, dtype=np.int32)
    improper_fix = np.zeros(n, dtype=bool)
    fixed = np.zeros(group.order, dtype=np.int64)

    for g in range(group.order):
        tables = _chunk_tables(group.edge_perm[g], n_e)
        flip = group.flip[g]
        for s in range(0, n, block):
            x = configs[s : s + block]
            img = apply(x, tables, flip)
            np.minimum(canon[s : s + block], img, out=canon[s : s + block])
            eq = img == x
            stab[s : s + block] += eq
            if not group.proper[g]:
                improper_fix[s : s + block] |= eq
            fixed[g] += int(eq.sum())

    reps, first, class_id, counts = np.unique(canon, return_index=True, return_inverse=True, return_counts=True)
    stab_rep = stab[first]
    if not np.all(counts * stab_rep == group.order):
        raise AssertionError("orbit-stabilizer relation violated; group action is inconsistent")
    return OrbitResult(
        configs=configs,
        canon=canon,
        class_id=class_id,
        representatives=reps,
        degeneracy=counts,
        stabilizer_order=stab_rep,
        stabilizer_improper=improper_fix[first],
        fixed_counts=fixed,
        group_order=group.order,
    )
