"""Graph automorphism group of a cage and its action on edge orientations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import Cage


@dataclass(frozen=True)
class Group:
    """A group acting on configurations by `x -> P_g(x) XOR flip[g]`.

    `edge_perm[g, e]` is the image of edge `e`; `flip[g]` has bit `edge_perm[g, e]` set
    when the image of `(u, v)` is `(a, b)` with `a > b`, i.e. the stored direction reverses.
    """

    vertex_perm: np.ndarray  # (G, V)
    proper: np.ndarray  # (G,) bool
    edge_perm: np.ndarray  # (G, E)
    flip: np.ndarray  # (G,) uint64

    @property
    def order(self) -> int:
        return len(self.vertex_perm)

    def subgroup(self, mask: np.ndarray) -> "Group":
        return Group(self.vertex_perm[mask], self.proper[mask], self.edge_perm[mask], self.flip[mask])

    def rotations(self) -> "Group":
        return self.subgroup(self.proper)


def _bfs_order(nb: list[list[int]], root: int = 0) -> tuple[list[int], list[int]]:
    order, parent = [root], [-1] * len(nb)
    seen = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        for w in nb[u]:
            if w not in seen:
                seen.add(w)
                parent[w] = u
                order.append(w)
                q.append(w)
    if len(order) != len(nb):
        raise ValueError("cage graph is not connected")
    return order, parent


def vertex_automorphisms(nb: list[list[int]]) -> list[list[int]]:
    """All adjacency-preserving vertex permutations, by backtracking along a BFS order."""
    n = len(nb)
    adj = [set(x) for x in nb]
    order, parent = _bfs_order(nb)
    # earlier[i]: neighbors of order[i] that appear before it in the BFS order
    pos = {v: i for i, v in enumerate(order)}
    earlier = [[w for w in nb[v] if pos[w] < i] for i, v in enumerate(order)]

    result: list[list[int]] = []
    sigma = [-1] * n
    used = [False] * n

    def extend(i: int) -> None:
        if i == n:
            result.append(sigma.copy())
            return
        v = order[i]
        candidates = range(n) if parent[v] < 0 else nb[sigma[parent[v]]]
        for c in candidates:
            if used[c]:
                continue
            if all(sigma[w] in adj[c] for w in earlier[i]):
                sigma[v] = c
                used[c] = True
                extend(i + 1)
                used[c] = False
                sigma[v] = -1

    extend(0)
    return result


def _local_chirality(coords: np.ndarray, nb: list[list[int]]) -> np.ndarray:
    s = np.empty(len(nb), dtype=np.int8)
    for v, (a, b, c) in enumerate(nb):
        m = np.stack([coords[a] - coords[v], coords[b] - coords[v], coords[c] - coords[v]])
        det = np.linalg.det(m)
        if abs(det) < 1e-9:
            raise ValueError(f"vertex {v} and its neighbors are coplanar; cannot determine chirality")
        s[v] = 1 if det > 0 else -1
    return s


def _perm_parity(seq: list[int]) -> int:
    seq = list(seq)
    sign = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign


def is_proper(sigma: list[int], nb: list[list[int]], chir: np.ndarray) -> bool:
    """True if `sigma` preserves the local handedness at every vertex (a rotation)."""
    agree = 0
    for v, ns in enumerate(nb):
        mapped = [sigma[w] for w in ns]
        target = nb[sigma[v]]
        parity = _perm_parity([target.index(m) for m in mapped])
        agree += 1 if chir[sigma[v]] * parity == chir[v] else -1
    if abs(agree) != len(nb):
        raise ValueError("automorphism preserves handedness at some vertices but not others; is the cage convex?")
    return agree > 0


def automorphism_group(cage: Cage) -> Group:
    nb = cage.neighbors()
    chir = _local_chirality(cage.coords, nb)
    perms = vertex_automorphisms(nb)

    edge_index = {(int(u), int(v)): e for e, (u, v) in enumerate(cage.edges)}
    n_e = cage.n_edges
    edge_perm = np.empty((len(perms), n_e), dtype=np.int64)
    flip = np.zeros(len(perms), dtype=np.uint64)
    proper = np.empty(len(perms), dtype=bool)
    for g, sigma in enumerate(perms):
        proper[g] = is_proper(sigma, nb, chir)
        mask = 0
        for e, (u, v) in enumerate(cage.edges):
            a, b = sigma[u], sigma[v]
            e2 = edge_index[(min(a, b), max(a, b))]
            edge_perm[g, e] = e2
            if a > b:
                mask |= 1 << e2
        flip[g] = mask
    return Group(np.array(perms, dtype=np.int64), proper, edge_perm, flip)
